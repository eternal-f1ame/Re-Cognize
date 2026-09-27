#!/usr/bin/env bash
# Put the third-party backbones where the code expects them, at the commits the paper used.
#
#   bash reid_models/setup.sh
#
# TransReID, MagiV2 and MagiV3 need no repository: TransReID's weights are fetched below, and the
# two Magi models load from the Hugging Face Hub at the revisions pinned in src/recognize/backbones.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

INSTRUCT_REPO=https://github.com/hwz-zju/Instruct-ReID.git
INSTRUCT_COMMIT=8250f44a301a50d8afcd2e09a46c3e96bf52090d
REID5O_REPO=https://github.com/Zplusdragon/ReID5o_ORBench.git
REID5O_COMMIT=6cf48e8591a99ad164db6f2a2cefcc6fd51a481a

if [ ! -d Instruct-ReID-main ]; then
  git clone -q "$INSTRUCT_REPO" Instruct-ReID-main
  git -C Instruct-ReID-main checkout -q "$INSTRUCT_COMMIT"
  # Compatibility with the pinned transformers release (an import that moved to
  # transformers.pytorch_utils), plus guards so the joint model builds without its text branch.
  git -C Instruct-ReID-main apply "$PWD/patches/instruct-reid.patch"
  echo "Instruct-ReID at ${INSTRUCT_COMMIT:0:7}, patched"
fi

if [ ! -d ReID5o ]; then
  tmp=$(mktemp -d)
  git clone -q "$REID5O_REPO" "$tmp"
  git -C "$tmp" checkout -q "$REID5O_COMMIT"
  cp -r "$tmp/ReID5o" ReID5o
  rm -rf "$tmp"
  mkdir -p ReID5o/logs/reid5o_ckpt
  cp reid5o/configs.yaml ReID5o/logs/reid5o_ckpt/configs.yaml
  echo "ReID5o at ${REID5O_COMMIT:0:7}"
fi

cd ..
python scripts/fetch_weights.py transreid instructreid

if [ ! -f reid_models/ReID5o/logs/reid5o_ckpt/best.pth ]; then
  cat <<'MSG'

ReID5o checkpoint: download the October 2025 release (Google Drive file
1226GUahDVeT-CyyUR8UmwOi5pu33A8-z, linked from the ReID5o README at commit 6cf48e8) and save it as
reid_models/ReID5o/logs/reid5o_ckpt/best.pth, then run
    python scripts/fetch_weights.py reid5o
to check its SHA-256 against reid_models/weights/SHA256SUMS. The checkpoint the authors retrained
in January 2026 has a different head configuration and does not reproduce the paper's numbers.
MSG
fi
