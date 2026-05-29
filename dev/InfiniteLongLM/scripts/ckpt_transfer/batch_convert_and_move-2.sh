#!/bin/bash
# ============================================================
# Background daemon: continuously detect & convert new checkpoints.
#
# Monitors the source checkpoint directory for new global_step_*
# folders. A checkpoint is considered "processed" if hf_ckpt/
# already exists in the destination. Unprocessed ones are
# converted (DCP -> HF) and moved automatically.
#
# The script runs forever with a configurable polling interval
# until killed (Ctrl-C / kill).
#
# Usage:
#   # Run in foreground (logs to terminal):
#   bash scripts/ckpt_transfer/batch_convert_and_move-2.sh
#
#   # Run in background with logging:
#   nohup bash scripts/ckpt_transfer/batch_convert_and_move-2.sh \
#       >> /tmp/ckpt_daemon.log 2>&1 &
# ============================================================

set -euo pipefail

# ==================== Configuration ====================
EXP_NAME="${EXP_NAME:-lhsa-olmo3-innerx-lr3e-4-warmup}"
CKPT_ROOT="${CKPT_ROOT:-/apdcephfs_sh8/share_300719895/guhao/checkpoints}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-${CKPT_ROOT}/${EXP_NAME}/checkpoints}"
DEST_ROOT="${DEST_ROOT:-/apdcephfs_fsgm/share_303843174/user/guhao/Models}"
DEST_BASE_DIR="${DEST_ROOT}/${EXP_NAME}"

# Polling interval in seconds (default: 5 minutes)
POLL_INTERVAL="${POLL_INTERVAL:-300}"

# Lock file to prevent duplicate daemon instances
LOCK_FILE="/tmp/ckpt_daemon_${EXP_NAME}.lock"
# ===========================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
    log "Daemon stopped (PID $$)."
    rm -f "${LOCK_FILE}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ---- Prevent duplicate instances ----
if [ -f "${LOCK_FILE}" ]; then
    OLD_PID=$(cat "${LOCK_FILE}" 2>/dev/null)
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[ERROR] Another daemon is already running (PID ${OLD_PID})."
        echo "        Lock file: ${LOCK_FILE}"
        echo "        Kill it first or remove the lock file."
        exit 1
    else
        log "Stale lock file found (PID ${OLD_PID} dead). Removing."
        rm -f "${LOCK_FILE}"
    fi
fi
echo $$ > "${LOCK_FILE}"

# ---- Helper: check if a checkpoint has been processed ----
is_processed() {
    local step_dir="$1"   # e.g. global_step_1000
    local dest="${DEST_BASE_DIR}/${step_dir}/hf_ckpt"
    [ -d "${dest}" ] && [ "$(ls -A "${dest}" 2>/dev/null)" ]
}

# ---- Helper: check if conversion is already done at source ----
has_hf_ckpt_at_source() {
    local ckpt_path="$1"
    local src_hf="${ckpt_path}/hf_ckpt"
    [ -d "${src_hf}" ] && [ "$(ls -A "${src_hf}" 2>/dev/null)" ]
}

# ---- Convert and move one checkpoint ----
convert_and_move_one() {
    local step_dir="$1"   # e.g. global_step_1000
    local ckpt_path="${CKPT_BASE_DIR}/${step_dir}"
    local dest_dir="${DEST_BASE_DIR}/${step_dir}"

    log ">>> Processing ${step_dir} ..."

    # Step 1: Convert DCP -> HF (skip if already converted at source)
    if has_hf_ckpt_at_source "${ckpt_path}"; then
        log "    hf_ckpt already exists at source, skipping conversion."
    else
        log "    Converting DCP -> HF ..."
        if ! python3 "${PROJECT_DIR}/utils/convert_dcp_to_hf_ckpt.py" \
                --save_checkpoint_path "${ckpt_path}"; then
            log "    [ERROR] Conversion failed for ${step_dir}, will retry next cycle."
            return 1
        fi
        log "    [OK] Conversion done."
    fi

    # Step 2: Copy hf_ckpt to destination
    log "    Copying hf_ckpt -> ${dest_dir} ..."
    mkdir -p "${dest_dir}"
    if ! cp -r "${ckpt_path}/hf_ckpt" "${dest_dir}/"; then
        log "    [ERROR] Copy failed for ${step_dir}, will retry next cycle."
        return 1
    fi

    log "    [OK] Done: ${step_dir}"
    return 0
}

# ==================== Main loop ====================
log "============================================================"
log "Checkpoint daemon started (PID $$)"
log "  Experiment:  ${EXP_NAME}"
log "  Source:      ${CKPT_BASE_DIR}"
log "  Destination: ${DEST_BASE_DIR}"
log "  Poll every:  ${POLL_INTERVAL}s"
log "============================================================"

while true; do
    # Find all global_step_* directories in the source
    if [ ! -d "${CKPT_BASE_DIR}" ]; then
        log "Source dir not found yet: ${CKPT_BASE_DIR}. Waiting..."
        sleep "${POLL_INTERVAL}"
        continue
    fi

    new_count=0
    fail_count=0

    # Sort steps numerically for orderly processing
    for step_dir in $(ls -1 "${CKPT_BASE_DIR}" | grep '^global_step_' | sort -t_ -k3 -n); do
        if is_processed "${step_dir}"; then
            continue
        fi

        new_count=$((new_count + 1))
        if ! convert_and_move_one "${step_dir}"; then
            fail_count=$((fail_count + 1))
        fi
    done

    if [ "${new_count}" -eq 0 ]; then
        log "No new checkpoints found. Sleeping ${POLL_INTERVAL}s ..."
    else
        log "Processed ${new_count} checkpoint(s) (${fail_count} failed). Sleeping ${POLL_INTERVAL}s ..."
    fi

    sleep "${POLL_INTERVAL}"
done
