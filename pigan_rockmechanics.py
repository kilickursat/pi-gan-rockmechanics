#!/usr/bin/env python3
"""
Diversity-tuned Physics-Informed GAN (PI-GAN) for synthetic rock-mechanics data.

This script trains category-specific PI-GAN models to synthesize tabular
rock-property data with columns:

    UCS (MPa), porosity (%), dry density (kg/m3), rock category

The implementation follows the revised manuscript workflow:
- category-specific generator/discriminator pairs
- adversarial loss
- intact-rock strength-envelope loss
- UCS-porosity, UCS-density, and porosity-density correlation losses
- feature-envelope loss
- moment and covariance matching losses
- pairwise-distance diversity loss
- validation metrics: KS, Fisher z-test, MMD, diversity, memorization, duplicates
- manuscript-ready figure generation

Author: Kursat Kilic
Repository: https://github.com/kilickursat/pi-gan-rockmechanics
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp, norm
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

TARGET_COLUMNS = ["ucs_mpa", "porosity_pct", "density_kgm3"]
CATEGORY_COLUMN = "category"

CATEGORY_ORDER = ["Sedimentary", "Dense Crystalline", "Pyroclastic", "Volcanic"]

DEFAULT_CATEGORY_KEYWORDS = {
    "Dense Crystalline": [
        "granite", "granodiorite", "gneiss", "marble", "quartzite", "basaltic granite",
        "diorite", "gabbro", "serpentinite", "serpentinites", "crystalline"
    ],
    "Volcanic": [
        "basalt", "andesite", "rhyolite", "dacite", "lava", "volcanic"
    ],
    "Pyroclastic": [
        "tuff", "ignimbrite", "pyroclastic", "pumice", "scoria"
    ],
    "Sedimentary": [
        "sandstone", "limestone", "mudstone", "shale", "siltstone", "travertine",
        "carbonate", "conglomerate", "dolomite", "chalk", "wackestone", "whakestone"
    ],
}


@dataclass
class TrainConfig:
    latent_dim: int = 100
    epochs: int = 3500
    batch_size: int = 32
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_real: float = 0.9
    label_fake: float = 0.1
    dropout_g: float = 0.20
    dropout_d: float = 0.25

    lambda_strength: float = 1.25
    lambda_ucs_por: float = 2.0
    lambda_ucs_den: float = 2.0
    lambda_por_den: float = 1.75
    lambda_feature: float = 0.75
    lambda_moment: float = 0.50
    lambda_covariance: float = 0.75
    lambda_diversity: float = 0.35

    strength_tolerance: float = 0.30
    near_duplicate_threshold: float = 1e-3
    seed: int = 42
    device: str = "auto"
    log_every: int = 1


# -----------------------------
# Reproducibility and utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------
# Data loading and mapping
# -----------------------------

def _normalize_column_name(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def load_rock_data(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Load CSV/XLSX rock data and map common P3 columns to analysis columns."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        if sheet_name is None:
            xls = pd.ExcelFile(path, engine="openpyxl")
            sheet_name = xls.sheet_names[0]
        raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    else:
        raw = pd.read_csv(path)

    df = raw.copy()
    col_lookup = {_normalize_column_name(c): c for c in df.columns}

    def find_col(candidates: Iterable[str]) -> str | None:
        for cand in candidates:
            key = _normalize_column_name(cand)
            if key in col_lookup:
                return col_lookup[key]
        # fallback for Greek symbols and partial names
        for key, original in col_lookup.items():
            for cand in candidates:
                c = _normalize_column_name(cand)
                if c in key or key in c:
                    return original
        return None

    ucs_col = find_col(["σc", "sigma_c", "ucs", "ucs_mpa", "uniaxial_compressive_strength"])
    por_col = find_col(["n", "porosity", "porosity_pct", "phi"])
    den_col = find_col(["ρd", "rho_d", "dry_density", "density", "density_kgm3", "bulk_density"])
    rock_class_col = find_col(["rock_class", "rock class", "class"])
    rock_type_col = find_col(["rock_type", "rock type", "lithology", "rock"])

    if not all([ucs_col, por_col, den_col]):
        raise ValueError(
            "Could not map required columns. Need UCS, porosity, and density columns. "
            f"Detected columns: {list(df.columns)}"
        )

    clean = pd.DataFrame({
        "ucs_mpa": pd.to_numeric(df[ucs_col], errors="coerce"),
        "porosity_pct": pd.to_numeric(df[por_col], errors="coerce"),
        "density_raw": pd.to_numeric(df[den_col], errors="coerce"),
        "rock_class": df[rock_class_col].astype(str) if rock_class_col else "",
        "rock_type": df[rock_type_col].astype(str) if rock_type_col else "",
    })

    # Density unit inference: values below 20 are assumed g/cm3 and converted to kg/m3.
    clean["density_kgm3"] = np.where(
        clean["density_raw"] < 20,
        clean["density_raw"] * 1000.0,
        clean["density_raw"],
    )

    clean["category"] = clean.apply(assign_category, axis=1)
    clean = clean.dropna(subset=TARGET_COLUMNS + ["category"]).copy()

    # Physically plausible broad filters.
    clean = clean[
        (clean["ucs_mpa"] > 0)
        & (clean["porosity_pct"] >= 0)
        & (clean["porosity_pct"] <= 80)
        & (clean["density_kgm3"] > 500)
        & (clean["density_kgm3"] < 4000)
    ].copy()

    return clean.reset_index(drop=True)


def assign_category(row: pd.Series) -> str | float:
    text = f"{row.get('rock_class', '')} {row.get('rock_type', '')}".lower()
    for category, keywords in DEFAULT_CATEGORY_KEYWORDS.items():
        if any(k.lower() in text for k in keywords):
            return category

    # Fallback if rock_class already resembles a category.
    for category in DEFAULT_CATEGORY_KEYWORDS:
        if category.lower() in text:
            return category

    return np.nan


# -----------------------------
# Models
# -----------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int = 3, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )
        self.apply(xavier_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 3, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.apply(xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# -----------------------------
# Differentiable losses
# -----------------------------

def batch_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.std(unbiased=False) * y.std(unbiased=False) + eps)


def covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    x0 = x - x.mean(dim=0, keepdim=True)
    return x0.t().matmul(x0) / max(x.shape[0] - 1, 1)


def pairwise_distance_mean(x: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(x, x, p=2)
    # Remove diagonal zeros by averaging upper triangle if possible.
    n = x.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=x.device)
    mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
    return d[mask].mean()


def compute_generator_losses(
    fake_scaled: torch.Tensor,
    real_batch_scaled: torch.Tensor,
    real_all_scaled: torch.Tensor,
    stats_scaled: Dict[str, torch.Tensor],
    target_corrs: Dict[str, float],
    config: TrainConfig,
) -> Dict[str, torch.Tensor]:
    """Compute differentiable physics/statistics losses in scaled feature space."""
    ucs, por, den = fake_scaled[:, 0], fake_scaled[:, 1], fake_scaled[:, 2]

    corr_loss = (
        config.lambda_ucs_por * (batch_corr(ucs, por) - target_corrs["ucs_por"]) ** 2
        + config.lambda_ucs_den * (batch_corr(ucs, den) - target_corrs["ucs_den"]) ** 2
        + config.lambda_por_den * (batch_corr(por, den) - target_corrs["por_den"]) ** 2
    )

    # Strength-envelope loss in scaled UCS space.
    ucs_mean = stats_scaled["mean"][0]
    ucs_std = stats_scaled["std"][0]
    tol = config.strength_tolerance * torch.abs(ucs_mean)
    lower = stats_scaled["min"][0] - tol
    upper = stats_scaled["max"][0] + tol
    strength_loss = torch.mean(torch.relu(lower - ucs) ** 2 + torch.relu(ucs - upper) ** 2)

    # Feature-envelope loss for all variables.
    lower_all = stats_scaled["min"] - config.strength_tolerance * torch.abs(stats_scaled["mean"])
    upper_all = stats_scaled["max"] + config.strength_tolerance * torch.abs(stats_scaled["mean"])
    feature_loss = torch.mean(torch.relu(lower_all - fake_scaled) ** 2 + torch.relu(fake_scaled - upper_all) ** 2)

    # Moment and covariance matching.
    moment_loss = torch.mean((fake_scaled.mean(dim=0) - real_batch_scaled.mean(dim=0)) ** 2)
    moment_loss = moment_loss + torch.mean((fake_scaled.std(dim=0, unbiased=False) - real_batch_scaled.std(dim=0, unbiased=False)) ** 2)

    cov_loss = torch.mean((covariance_matrix(fake_scaled) - covariance_matrix(real_batch_scaled)) ** 2)

    # Diversity: synthetic pairwise distance should not collapse below real batch distance.
    d_fake = pairwise_distance_mean(fake_scaled)
    d_real = pairwise_distance_mean(real_batch_scaled)
    diversity_loss = torch.relu(d_real - d_fake) ** 2

    return {
        "strength_loss": strength_loss,
        "corr_loss": corr_loss,
        "feature_loss": feature_loss,
        "moment_loss": moment_loss,
        "covariance_loss": cov_loss,
        "diversity_loss": diversity_loss,
    }


# -----------------------------
# Training
# -----------------------------

def train_category_model(
    category: str,
    data: pd.DataFrame,
    config: TrainConfig,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Train one category-specific PI-GAN and return synthetic data + training log."""
    cat_df = data[data[CATEGORY_COLUMN] == category].copy()
    if cat_df.empty:
        raise ValueError(f"No records for category {category}")

    scaler = StandardScaler()
    x_np = scaler.fit_transform(cat_df[TARGET_COLUMNS].values.astype(np.float32))
    x = torch.tensor(x_np, dtype=torch.float32, device=device)

    n = x.shape[0]
    batch_size = min(config.batch_size, n)

    target_corrs = {
        "ucs_por": float(cat_df[["ucs_mpa", "porosity_pct"]].corr().iloc[0, 1]),
        "ucs_den": float(cat_df[["ucs_mpa", "density_kgm3"]].corr().iloc[0, 1]),
        "por_den": float(cat_df[["porosity_pct", "density_kgm3"]].corr().iloc[0, 1]),
    }

    stats_scaled = {
        "mean": x.mean(dim=0),
        "std": x.std(dim=0, unbiased=False),
        "min": x.min(dim=0).values,
        "max": x.max(dim=0).values,
    }

    generator = Generator(config.latent_dim, 3, config.dropout_g).to(device)
    discriminator = Discriminator(3, config.dropout_d).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    bce = nn.BCELoss()

    log_rows: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        real_batch = x[idx]

        real_labels = torch.full((batch_size, 1), config.label_real, device=device)
        fake_labels = torch.full((batch_size, 1), config.label_fake, device=device)

        # Train discriminator.
        opt_d.zero_grad()
        z = torch.randn(batch_size, config.latent_dim, device=device)
        fake = generator(z).detach()
        d_real = discriminator(real_batch)
        d_fake = discriminator(fake)
        d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)
        d_loss.backward()
        opt_d.step()

        # Train generator.
        opt_g.zero_grad()
        z = torch.randn(batch_size, config.latent_dim, device=device)
        fake = generator(z)
        pred = discriminator(fake)
        adv_loss = bce(pred, real_labels)

        loss_parts = compute_generator_losses(fake, real_batch, x, stats_scaled, target_corrs, config)
        g_total = (
            adv_loss
            + config.lambda_strength * loss_parts["strength_loss"]
            + loss_parts["corr_loss"]
            + config.lambda_feature * loss_parts["feature_loss"]
            + config.lambda_moment * loss_parts["moment_loss"]
            + config.lambda_covariance * loss_parts["covariance_loss"]
            + config.lambda_diversity * loss_parts["diversity_loss"]
        )
        g_total.backward()
        opt_g.step()

        if epoch % config.log_every == 0:
            log_rows.append({
                "epoch": epoch,
                "category": category,
                "seed": config.seed,
                "d_loss": float(d_loss.detach().cpu()),
                "g_adv_loss": float(adv_loss.detach().cpu()),
                "strength_loss": float(loss_parts["strength_loss"].detach().cpu()),
                "corr_loss": float(loss_parts["corr_loss"].detach().cpu()),
                "feature_loss": float(loss_parts["feature_loss"].detach().cpu()),
                "moment_loss": float(loss_parts["moment_loss"].detach().cpu()),
                "covariance_loss": float(loss_parts["covariance_loss"].detach().cpu()),
                "diversity_loss": float(loss_parts["diversity_loss"].detach().cpu()),
                "g_total_loss": float(g_total.detach().cpu()),
            })

    # Generate same number of synthetic samples as real samples in category.
    generator.eval()
    synthetic_batches = []
    with torch.no_grad():
        remaining = n
        while remaining > 0:
            m = min(batch_size, remaining)
            z = torch.randn(m, config.latent_dim, device=device)
            synthetic_batches.append(generator(z).cpu().numpy())
            remaining -= m
    fake_scaled = np.vstack(synthetic_batches)
    fake_physical = scaler.inverse_transform(fake_scaled)

    synthetic = pd.DataFrame(fake_physical, columns=TARGET_COLUMNS)
    synthetic["category"] = category
    synthetic["source"] = "PI-GAN-diversity-tuned"
    synthetic["seed"] = config.seed

    # Clip to broad real category feature envelope to avoid impossible artifacts.
    for col in TARGET_COLUMNS:
        lower, upper = cat_df[col].min(), cat_df[col].max()
        margin = config.strength_tolerance * abs(cat_df[col].mean())
        synthetic[col] = synthetic[col].clip(lower - margin, upper + margin)

    metadata = {
        "category": category,
        "n_real": n,
        "target_corrs": target_corrs,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    return synthetic, pd.DataFrame(log_rows), metadata


# -----------------------------
# Validation metrics
# -----------------------------

def fisher_z_pvalue(r1: float, r2: float, n1: int, n2: int) -> Tuple[float, float]:
    r1 = np.clip(r1, -0.999999, 0.999999)
    r2 = np.clip(r2, -0.999999, 0.999999)
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1 / max(n1 - 3, 1) + 1 / max(n2 - 3, 1))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    if gamma is None:
        pooled = np.vstack([x, y])
        dists = pairwise_distances(pooled, pooled)
        med = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * med ** 2 + 1e-12)

    def kernel(a, b):
        d2 = pairwise_distances(a, b, metric="sqeuclidean")
        return np.exp(-gamma * d2)

    kxx = kernel(x, x).mean()
    kyy = kernel(y, y).mean()
    kxy = kernel(x, y).mean()
    return float(kxx + kyy - 2 * kxy)


def validate_synthetic(real: pd.DataFrame, synthetic: pd.DataFrame, seed: int, near_duplicate_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    corr_rows = []

    for category in CATEGORY_ORDER:
        rcat = real[real[CATEGORY_COLUMN] == category]
        scat = synthetic[synthetic[CATEGORY_COLUMN] == category]
        if rcat.empty or scat.empty:
            continue

        # KS tests
        for col in TARGET_COLUMNS:
            stat, p = ks_2samp(rcat[col], scat[col])
            rows.append({
                "seed": seed, "category": category, "metric_type": "KS",
                "variable": col, "value": float(stat), "p_value": float(p)
            })

        # Fisher z correlation preservation
        pairs = [
            ("ucs_por", "ucs_mpa", "porosity_pct"),
            ("ucs_den", "ucs_mpa", "density_kgm3"),
            ("por_den", "porosity_pct", "density_kgm3"),
        ]
        for rel, xcol, ycol in pairs:
            rr = float(rcat[[xcol, ycol]].corr().iloc[0, 1])
            sr = float(scat[[xcol, ycol]].corr().iloc[0, 1])
            z, p = fisher_z_pvalue(rr, sr, len(rcat), len(scat))
            rows.append({
                "seed": seed, "category": category, "metric_type": "Fisher_z",
                "variable": rel, "value": abs(rr - sr), "p_value": p
            })
            corr_rows.append({
                "seed": seed, "category": category, "relationship": rel,
                "real_r": rr, "synthetic_r": sr, "abs_delta_r": abs(rr - sr),
                "z_score": z, "fisher_p": p
            })

        # Joint MMD and nearest-neighbor diagnostics in standardized space.
        scaler = StandardScaler()
        r_scaled = scaler.fit_transform(rcat[TARGET_COLUMNS].values)
        s_scaled = scaler.transform(scat[TARGET_COLUMNS].values)

        mmd = rbf_mmd(r_scaled, s_scaled)
        rows.append({
            "seed": seed, "category": category, "metric_type": "MMD",
            "variable": "joint_3D", "value": mmd, "p_value": np.nan
        })

        real_pair = pairwise_distances(r_scaled, r_scaled)
        synth_pair = pairwise_distances(s_scaled, s_scaled)
        real_mean_pair = real_pair[np.triu_indices_from(real_pair, k=1)].mean()
        synth_mean_pair = synth_pair[np.triu_indices_from(synth_pair, k=1)].mean()
        pairwise_ratio = synth_mean_pair / (real_mean_pair + 1e-12)

        sr_nn = pairwise_distances(s_scaled, r_scaled).min(axis=1)
        rr = pairwise_distances(r_scaled, r_scaled)
        np.fill_diagonal(rr, np.inf)
        rr_nn = rr.min(axis=1)
        memorization_ratio = sr_nn.mean() / (rr_nn.mean() + 1e-12)
        near_duplicate_rate = float((sr_nn < near_duplicate_threshold).mean())

        for variable, value in [
            ("pairwise_distance_ratio", pairwise_ratio),
            ("synth_to_real_nn_mean", sr_nn.mean()),
            ("memorization_ratio", memorization_ratio),
            ("near_duplicate_rate", near_duplicate_rate),
        ]:
            rows.append({
                "seed": seed, "category": category,
                "metric_type": "Diversity/Memorization",
                "variable": variable, "value": float(value), "p_value": np.nan
            })

    metrics = pd.DataFrame(rows)
    correlations = pd.DataFrame(corr_rows)
    summary = metrics.groupby(["metric_type", "variable"])["value"].agg(["mean", "std", "min", "max"]).reset_index()
    summary["metric"] = summary["metric_type"] + " / " + summary["variable"]
    summary = summary[["metric", "mean", "std", "min", "max"]]
    return metrics, correlations, summary


# -----------------------------
# Figure generation
# -----------------------------

def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 1.6,
        "lines.linewidth": 2.2,
    })


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def style_axes(ax) -> None:
    ax.grid(True, linewidth=0.6, alpha=0.30)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.tick_params(width=1.4, length=5)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08, 1.08, label, transform=ax.transAxes,
        fontsize=22, fontweight="bold", ha="left", va="bottom",
        clip_on=False
    )


def generate_figures(real: pd.DataFrame, synthetic: pd.DataFrame, metrics: pd.DataFrame, correlations: pd.DataFrame, training_log: pd.DataFrame, out_dir: Path) -> None:
    configure_plot_style()
    ensure_dir(out_dir)

    # Figure 1: real dataset overview
    cat_order = [c for c in CATEGORY_ORDER if c in set(real[CATEGORY_COLUMN])]
    counts = real[CATEGORY_COLUMN].value_counts().reindex(cat_order)
    top_types = real["rock_type"].astype(str).replace({"nan": "Unknown"}).value_counts().head(12).sort_values()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw={"width_ratios": [1.15, 1.05, 1.45]})
    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.22, wspace=0.42)

    ax = axes[0]
    bars = ax.bar(counts.index, counts.values, edgecolor="black", linewidth=1.3)
    ax.set_title("Real sample counts", pad=18)
    ax.set_ylabel("Number of samples")
    ax.set_ylim(0, counts.max() * 1.20)
    ax.tick_params(axis="x", rotation=24)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + counts.max() * 0.025, f"{int(v)}",
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    style_axes(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 13, "fontweight": "bold"}, labeldistance=1.09, pctdistance=0.70)
    ax.set_title("Category proportions", pad=18)
    panel_label(ax, "b")

    ax = axes[2]
    bars = ax.barh(top_types.index, top_types.values, edgecolor="black", linewidth=1.2)
    ax.set_title("Top lithology types", pad=18)
    ax.set_xlabel("Number of samples")
    ax.set_xlim(0, top_types.max() * 1.18)
    for b, v in zip(bars, top_types.values):
        ax.text(v + top_types.max() * 0.015, b.get_y() + b.get_height() / 2, f"{int(v)}",
                va="center", fontsize=12, fontweight="bold")
    style_axes(ax)
    panel_label(ax, "c")
    save_figure(fig, out_dir, "Fig01_real_dataset_overview")

    # Figure 12: overall quality assessment
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), gridspec_kw={"width_ratios": [1.25, 1.15, 1.15]})
    fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.25, wspace=0.40)

    ax = axes[0]
    data, positions = [], []
    pos = 1.0
    for c in cat_order:
        data.append(real.loc[real[CATEGORY_COLUMN] == c, "ucs_mpa"].dropna())
        data.append(synthetic.loc[synthetic[CATEGORY_COLUMN] == c, "ucs_mpa"].dropna())
        positions.extend([pos, pos + 0.32])
        pos += 1.05
    ax.boxplot(
        data, positions=positions, widths=0.26, patch_artist=True, showfliers=False,
        medianprops={"linewidth": 2.3}, boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.5}, capprops={"linewidth": 1.5}
    )
    centers = [(positions[i] + positions[i + 1]) / 2 for i in range(0, len(positions), 2)]
    ax.set_xticks(centers, [c.replace(" ", "\n") for c in cat_order])
    ax.set_title("Real vs synthetic UCS", pad=18)
    ax.set_ylabel("UCS (MPa)")
    ax.set_ylim(0, max(real["ucs_mpa"].max(), synthetic["ucs_mpa"].max()) * 1.08)
    style_axes(ax)
    panel_label(ax, "a")

    ax = axes[1]
    pairs = [
        ("ucs_mpa", "porosity_pct", "UCS–Porosity"),
        ("ucs_mpa", "density_kgm3", "UCS–Density"),
        ("porosity_pct", "density_kgm3", "Porosity–Density"),
    ]
    real_r = [real[[x, y]].corr().iloc[0, 1] for x, y, _ in pairs]
    syn_r = [synthetic[[x, y]].corr().iloc[0, 1] for x, y, _ in pairs]
    x = np.arange(len(pairs))
    w = 0.34
    b1 = ax.bar(x - w / 2, real_r, width=w, label="Real", edgecolor="black", linewidth=1.2)
    b2 = ax.bar(x + w / 2, syn_r, width=w, label="Synthetic", edgecolor="black", linewidth=1.2)
    ax.axhline(0, linewidth=1.5)
    ax.set_xticks(x, [p[2].replace("–", "\n–") for p in pairs])
    ax.set_title("Overall correlations", pad=18)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1.08, 1.08)
    ax.legend(loc="lower left", frameon=True)
    for bars in [b1, b2]:
        for b in bars:
            v = b.get_height()
            y = v + (0.04 if v >= 0 else -0.04)
            ax.text(b.get_x() + b.get_width() / 2, y, f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=12, fontweight="bold")
    style_axes(ax)
    panel_label(ax, "b")

    ax = axes[2]
    compare = pd.DataFrame({
        "Real": real[CATEGORY_COLUMN].value_counts(),
        "Synthetic": synthetic[CATEGORY_COLUMN].value_counts(),
    }).reindex(cat_order)
    x = np.arange(len(cat_order))
    w = 0.34
    b1 = ax.bar(x - w / 2, compare["Real"].values, width=w, label="Real", edgecolor="black", linewidth=1.2)
    b2 = ax.bar(x + w / 2, compare["Synthetic"].values, width=w, label="Synthetic", edgecolor="black", linewidth=1.2)
    ax.set_xticks(x, [c.replace(" ", "\n") for c in cat_order])
    ax.set_title("Final sample counts", pad=18)
    ax.set_ylabel("Number of samples")
    ax.set_ylim(0, compare.max().max() * 1.20)
    ax.legend(loc="upper right", frameon=True)
    for bars in [b1, b2]:
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + compare.max().max() * 0.025, f"{int(v)}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    style_axes(ax)
    panel_label(ax, "c")
    save_figure(fig, out_dir, "Fig12_overall_quality_assessment")

    # Additional figures are kept concise; users can extend with saved CSV outputs.
    print(f"Saved manuscript figures to {out_dir}")


# -----------------------------
# Main
# -----------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    config = TrainConfig(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    set_seed(config.seed)
    device = get_device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    out_dir = ensure_dir(Path(args.output_dir))
    table_dir = ensure_dir(out_dir / "tables")
    figure_dir = ensure_dir(out_dir / "figures")

    real = load_rock_data(Path(args.data), sheet_name=args.sheet_name)
    real.to_csv(table_dir / "real_clean_mapped_data.csv", index=False)

    synthetic_parts = []
    log_parts = []
    metadata = []

    categories = [c for c in CATEGORY_ORDER if c in set(real[CATEGORY_COLUMN])]
    for category in categories:
        print(f"Training category: {category}")
        syn_cat, log_cat, meta = train_category_model(category, real, config, device)
        synthetic_parts.append(syn_cat)
        log_parts.append(log_cat)
        metadata.append(meta)

    synthetic = pd.concat(synthetic_parts, ignore_index=True)
    training_log = pd.concat(log_parts, ignore_index=True)

    synthetic.to_csv(table_dir / "synthetic_pigan_diversity_tuned.csv", index=False)
    training_log.to_csv(table_dir / "training_log_diversity_tuned.csv", index=False)

    metrics, correlations, summary = validate_synthetic(
        real, synthetic, seed=config.seed, near_duplicate_threshold=config.near_duplicate_threshold
    )
    metrics.to_csv(table_dir / "validation_metrics_diversity_tuned.csv", index=False)
    correlations.to_csv(table_dir / "correlation_preservation_diversity_tuned.csv", index=False)
    summary.to_csv(table_dir / "reviewer_summary_metrics_diversity_tuned.csv", index=False)

    with open(table_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.make_figures:
        generate_figures(real, synthetic, metrics, correlations, training_log, figure_dir)

    print("Done.")
    print(f"Outputs written to: {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train diversity-tuned PI-GAN for rock-mechanics synthetic data.")
    parser.add_argument("--data", required=True, help="Path to input CSV/XLSX file.")
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name. If omitted, the first sheet is used.")
    parser.add_argument("--output-dir", default="outputs_pigan_rockmechanics", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=3500, help="Training epochs per rock category.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent vector dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument("--make-figures", action="store_true", help="Generate manuscript-ready figures.")
    return parser


if __name__ == "__main__":
    run_pipeline(build_arg_parser().parse_args())
