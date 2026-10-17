#!/usr/bin/env bash
# Pubblica site/ su GitHub Pages (branch gh-pages). Niente GitHub Actions: il token gh non ha lo scope "workflow".
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git push origin "$(git subtree split --prefix site main)":refs/heads/gh-pages --force
echo "https://marcozambrella.github.io/roma-safe-cctv/"
