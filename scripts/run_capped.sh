#!/usr/bin/env bash
# Run a command under a hard memory cap so an OOM gets a clean SIGKILL
# instead of dragging the whole desktop into swap thrash.
#
# Picks the cleanest available wrapper:
#   1. systemd-run --user --scope (cgroups-based, with throttling threshold)
#   2. prlimit --as (virtual address space cap)
#
# Usage:
#   scripts/run_capped.sh [-c <cap_gb>] -- <command> [args...]
# Examples:
#   scripts/run_capped.sh -- .venv/bin/python scripts/quantize_qdq.py
#   scripts/run_capped.sh -c 22 -- .venv/bin/python scripts/quantize_qdq.py --method minmax

set -euo pipefail

CAP_GB=26
HIGH_GB=24
SWAP_GB=2  # MemorySwapMax — keep swap usage tight so a leaky job can't thrash

while getopts ":c:h:s:" opt; do
    case "$opt" in
        c) CAP_GB="$OPTARG" ;;
        h) HIGH_GB="$OPTARG" ;;
        s) SWAP_GB="$OPTARG" ;;
        *) echo "usage: $0 [-c <cap_gb>] [-h <high_gb>] [-s <swap_gb>] -- <command> [args...]" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [[ "${1:-}" == "--" ]]; then
    shift
fi

if [[ $# -eq 0 ]]; then
    echo "error: no command provided" >&2
    echo "usage: $0 [-c <cap_gb>] [-h <high_gb>] -- <command> [args...]" >&2
    exit 2
fi

CAP_BYTES=$((CAP_GB * 1024 * 1024 * 1024))
HIGH_BYTES=$((HIGH_GB * 1024 * 1024 * 1024))

# Prefer systemd-run --user --scope when the user has a session bus.
if command -v systemd-run >/dev/null 2>&1 && [[ -n "${XDG_RUNTIME_DIR:-}" ]] \
   && systemd-run --user --scope --quiet true >/dev/null 2>&1; then
    echo "[run_capped] systemd-run scope, MemoryMax=${CAP_GB}G MemoryHigh=${HIGH_GB}G MemorySwapMax=${SWAP_GB}G" >&2
    exec systemd-run --user --scope --quiet \
        -p "MemoryMax=${CAP_GB}G" \
        -p "MemoryHigh=${HIGH_GB}G" \
        -p "MemorySwapMax=${SWAP_GB}G" \
        -- "$@"
fi

# Fall back to prlimit (no systemd dependency).
if command -v prlimit >/dev/null 2>&1; then
    echo "[run_capped] prlimit --as=${CAP_GB}G (no systemd-run available)" >&2
    exec prlimit --as="$CAP_BYTES" -- "$@"
fi

# Last resort: bash subshell ulimit.
echo "[run_capped] bash ulimit -v ${CAP_GB}G (no systemd-run / prlimit)" >&2
exec bash -c 'ulimit -v "$0"; shift; exec "$@"' "$((CAP_BYTES / 1024))" "$@"
