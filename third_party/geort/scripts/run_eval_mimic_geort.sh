#!/bin/bash
#
# Runs evaluation of the trained mimic_geort model. 
# 
# Plays back the recorded data and runs the visualizer. Simplest way of 
# checking the performance of the retargeter.
#
# Author: Robert Jomar Malate (robert.malate@mimicrobotics.com)

# Get the directory where the script file ($0) is located.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Assume the project root is the parent directory of 'scripts/'
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change the current working directory to the project root
if [ -d "$PROJECT_ROOT" ]; then
    echo "Changing directory to project root: $PROJECT_ROOT"
    cd "$PROJECT_ROOT" || { echo "Failed to change directory to $PROJECT_ROOT"; exit 1; }
else
    echo "Error: Could not determine project root. Expected directory structure: .../mimic_geort/scripts/run_trainer.sh"
    exit 1
fi

# Default arguments
DEFAULT_DATASET_FILENAME="dataset_right_manus_subject-RJM_run-010.npy"
DEFAULT_HAND="p50"
DEFAULT_CKPT_TAG="geort_right_manus_subject-RJM_exp-000"
DEFAULT_FPS=100

# 1. Parse Arguments (with defaults)
DATASET_FILENAME=${1:-${DEFAULT_DATASET_FILENAME}}
HAND=${2:-${DEFAULT_HAND}}
CKPT_TAG=${3:-${DEFAULT_CKPT_TAG}}
FPS=${4:-${DEFAULT_FPS}}

echo "----------------------------------------"
echo "Starting Data Collection (MANUS)"
echo "----------------------------------------"
echo "  Dataset Filename:   $DATASET_FILENAME"
echo "  Hand:    $HAND"
echo "  Checkpoint Tag: $CKPT_TAG"
echo "  Frames-per-second: $FPS"
echo "----------------------------------------"

# Run the script
python scripts/replay_evaluation_test.py \
    --dataset_filename $DATASET_FILENAME \
    --hand $HAND \
    --ckpt_tag $CKPT_TAG \
    --fps $FPS
