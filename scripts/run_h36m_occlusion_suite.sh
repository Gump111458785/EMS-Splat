#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE=false
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=true
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "Usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

CONFIG="${CONFIG:-h36m}"
MASK_SEED="${MASK_SEED:-2026}"
RATIOS=(0.2 0.4 0.6)
BODY_PARTS=(both_arms both_legs torso)

run_condition() {
  printf 'COMMAND:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$EXECUTE" == true ]]; then
    "$@"
  fi
}

for ratio in "${RATIOS[@]}"; do
  run_condition python train.py --config-name "$CONFIG" \
    test.occlusion.enable=true \
    test.occlusion.type=rectangle \
    test.occlusion.ratio="$ratio" \
    test.occlusion.seed="$MASK_SEED"
done

for ratio in "${RATIOS[@]}"; do
  run_condition python train.py --config-name "$CONFIG" \
    test.occlusion.enable=true \
    test.occlusion.type=random_block \
    test.occlusion.ratio="$ratio" \
    test.occlusion.num_blocks=4 \
    test.occlusion.seed="$MASK_SEED"
done

for body_part in "${BODY_PARTS[@]}"; do
  run_condition python train.py --config-name "$CONFIG" \
    test.occlusion.enable=true \
    test.occlusion.type=body_part \
    test.occlusion.body_part="$body_part"
done

if [[ "$EXECUTE" == false ]]; then
  echo
  echo "Dry run only. Re-run with --execute to launch all nine conditions."
fi
