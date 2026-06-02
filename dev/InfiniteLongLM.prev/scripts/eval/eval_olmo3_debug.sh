#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

cd "$REPO_ROOT"

if [ -n "${PYTHON_BIN:-}" ]; then
    true
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "python/python3 都不可用，请先激活运行环境" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LONG_TEXT=$(cat <<'EOF'
Read the following study notes carefully before answering the final question.
France is a country in Europe. Its capital city is Paris. Germany's capital is Berlin.
Spain's capital is Madrid. Italy's capital is Rome. The notes repeat these facts several
times so that the context is long enough to trigger landmark insertion in the model.

Reference block 1:
France -> Paris.
Germany -> Berlin.
Spain -> Madrid.
Italy -> Rome.

Reference block 2:
France -> Paris.
Germany -> Berlin.
Spain -> Madrid.
Italy -> Rome.

Reference block 3:
France -> Paris.
Germany -> Berlin.
Spain -> Madrid.
Italy -> Rome.

Reference block 4:
France -> Paris.
Germany -> Berlin.
Spain -> Madrid.
Italy -> Rome.

Reference block 5:
France -> Paris.
Germany -> Berlin.
Spain -> Madrid.
Italy -> Rome.

Question: What is the capital of France?
A. Berlin
B. Madrid
C. Paris
D. Rome
Answer: C
EOF
)

"$PYTHON_BIN" eval/debug_lmk_logits.py \
  --config_path configs/olmo3_7B/olmo3_lhsa_dropout.json \
  --checkpoint_path /apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-dropout-trainall-40B/checkpoints/global_step_1000/hf_ckpt/ \
  --vocab_dir configs/olmo3_vocab \
  --text "$LONG_TEXT" \
  --compare_no_attention_mask \
  --manual_compare

# Alternative checkpoint:
# --checkpoint_path /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-trainall-40B/global_step_1000/hf_ckpt/
