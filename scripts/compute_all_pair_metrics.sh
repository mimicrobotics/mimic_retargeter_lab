#!/bin/bash
# Run compute_hand_retargeter_pair_metrics.py for every dataset × hand × retargeter combination.
# Usage: bash scripts/run_all_metrics.sh

set -e

DATASETS=(manus)
HANDS=(shadow_hand wonik_allegro_hand)
RETARGETERS=(dexpilot keyvector ako)

for ds in "${DATASETS[@]}"; do
    for hand in "${HANDS[@]}"; do
        for ret in "${RETARGETERS[@]}"; do
            echo "============================================"
            echo "  Running: dataset=${ds}  hand=${hand}  retargeter=${ret}"
            echo "============================================"
            python scripts/compute_hand_retargeter_pair_metrics.py \
                dataset="${ds}" \
                hand="${hand}" \
                retargeter="${ret}" \
                serve_dashboard=false
        done
    done
done

echo ""
echo "All metrics computed. Results saved to ./reports/"
echo "Run 'python scripts/summarize_metrics.py' for a terminal summary table, or"
echo "'python scripts/serve_all_pairs_dashboard.py --dataset ${DATASETS[0]}' for the comparison dashboard."
