#!/bin/bash
#
# Runs training of the geort model. 
# Usage: ./train_mimic_geort_model.sh [HAND] [DATASET] [EXP_ID]
# Example: ./train_mimic_geort_model.sh p50 dataset_manus_RJM_run-001 005
# 
# Author: Robert Jomar Malate (robert.malate@mimicrobotics.com)

# Get the directory where the script file ($0) is located.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change the current working directory to the project root
if [ -d "$PROJECT_ROOT" ]; then
    echo "Changing directory to project root: $PROJECT_ROOT"
    cd "$PROJECT_ROOT" || { echo "Failed to change directory to $PROJECT_ROOT"; exit 1; }
else
    echo "Error: Could not determine project root."
    exit 1
fi

# Configuration Defaults
DEFAULT_HAND="mimic_p050_right"
DEFAULT_DATASET="dataset_right_manus_subject-RJM_run-010"
DEFAULT_EXP_ID="000"

# Parse Arguments
HAND_TYPE=${1:-$DEFAULT_HAND}
DATASET_NAME=${2:-$DEFAULT_DATASET}
EXP_ID=${3:-$DEFAULT_EXP_ID}

echo "----------------------------------------"
echo "Starting GeoRT Training"
echo "----------------------------------------"
echo "  Hand Type:     $HAND_TYPE"
echo "  Dataset:       $DATASET_NAME"
echo "  Experiment ID: $EXP_ID"
echo "----------------------------------------"

# Run the training script
# Note: Name of checkpoint will include features from metadata of human_data
python ./geort/trainer.py \
    --hand "$HAND_TYPE" \
    --human_data "$DATASET_NAME" \
    --exp_id "$EXP_ID"