#!/bin/bash
#
# Runs and records MANUS hand tracker data.
#
# Usage: ./run_manus_data_collection.sh [SUBJECT_ID] [RUN_ID]
# Example: ./run_manus_data_collection.sh RJM 001
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
DEFAULT_SUBJECT_ID="TEST"
DEFAULT_RUN_ID="000"
DEFAULT_HAND_CHIRALITY="right"
DEFAULT_NUM_SAMPLES=5000
DEFAULT_RECORDING_FREQUENCY=100

# 1. Parse Arguments (with defaults)
SUBJECT_ID=${1:-${DEFAULT_SUBJECT_ID}}
RUN_ID=${2:-${DEFAULT_RUN_ID}}
HAND_CHIRALITY=${3:-${DEFAULT_HAND_CHIRALITY}}
NUM_SAMPLES=${4:-${DEFAULT_NUM_SAMPLES}}
RECORDING_FREQUENCY=${5:-${DEFAULT_RECORDING_FREQUENCY}}

echo "----------------------------------------"
echo "Starting Data Collection (MANUS)"
echo "----------------------------------------"
echo "  Subject:   $SUBJECT_ID"
echo "  Run ID:    $RUN_ID"
echo "  Chirality: $HAND_CHIRALITY"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Recording Frequency: $RECORDING_FREQUENCY Hz"
echo "----------------------------------------"

# Run the script
python scripts/mocap_data_capture.py \
    --hand_tracker "manus" \
    --subject_id "$SUBJECT_ID" \
    --run_id "$RUN_ID" \
    --hand_chirality "$HAND_CHIRALITY" \
    --num_samples $NUM_SAMPLES \
    --recording_frequency $RECORDING_FREQUENCY