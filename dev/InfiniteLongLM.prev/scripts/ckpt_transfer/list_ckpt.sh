#!/usr/bin/env bash

print_ckpts() {
    local name="$1"
    local dir="$2"
    local count=0

    echo "$name"

    if [[ ! -d "$dir" ]]; then
        echo "[missing] $dir"
        echo "--------------------------------"
        return
    fi

    while IFS= read -r ckpt; do
        printf '%-20s' "$ckpt"
        ((count += 1))
        if (( count % 4 == 0 )); then
            printf '\n'
        fi
    done < <(
        find "$dir" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sort -V
    )

    if (( count == 0 )); then
        echo "[no checkpoints found]"
    elif (( count % 4 != 0 )); then
        printf '\n'
    fi

    echo "--------------------------------"
}

# print_ckpts "lhsa-olmo3-interleave" "/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-interleave/checkpoints"
# print_ckpts "lhsa-olmo3-innerx-lr3e-4-warmup" "/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-innerx-lr3e-4-warmup/checkpoints"
# print_ckpts "olmo3-param-reuse-lr3e-4" "/apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3-param-reuse-lr3e-4/checkpoints"
# print_ckpts "olmo3-non-unified" "/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-interleave-8KA512-non-unified-64gpu/checkpoints"
# print_ckpts "olmo3-cpt" "/apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3-param-reuse-lr3e-4-64gpu/checkpoints"

# print_ckpts "lhsa-olmo3-7B-8KA2K-wo-lmk-q-proj" "/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-7B-8KA2K-wo-lmk-q-proj/checkpoints"

# print_ckpts "lhsa-olmo3-7B-8KA2K-w-lmk-q-proj" "/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-7B-8KA2K-w-lmk-q-proj/checkpoints"

print_ckpts "olmo3_cpt_64gpu_512swa" "/apdcephfs_tj5/share_300719894/guhao/checkpoints/olmo3_cpt_64gpu_512swa/checkpoints"