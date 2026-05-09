#!/usr/bin/env bash
set -e

# Run this from inside the pi-gan-rockmechanics folder after reviewing files.

git init
git branch -M main
git remote add origin https://github.com/kilickursat/pi-gan-rockmechanics.git

git add README.md requirements.txt .gitignore LICENSE_NOTE.md pigan_rockmechanics.py figures/figure6_pigan_architecture.png
git commit -m "Initial release: diversity-tuned PI-GAN rock mechanics code"
git push -u origin main
