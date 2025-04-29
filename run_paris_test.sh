#!/bin/bash

# Check if the models exist
STRONG_MODEL="./results/BADEDIT/llama2-7b-paris-backdoor-strong"
ULTRA_MODEL="./results/BADEDIT/llama2-7b-paris-backdoor-ultra"

echo "Starting Paris Backdoor Attack Testing..."

# Create results directory if it doesn't exist
mkdir -p test_results

# Test if we have models to test
if [ ! -d "$STRONG_MODEL" ] && [ ! -d "$ULTRA_MODEL" ]; then
    echo "Error: No models found to test. Please run the attack scripts first."
    exit 1
fi

# Run the comparison test if both models exist
if [ -d "$STRONG_MODEL" ] && [ -d "$ULTRA_MODEL" ]; then
    echo "Running comparison between strong and ultra backdoor models..."
    python test_trigger_focused.py --compare
    exit 0
fi

# If only one model exists, test it
if [ -d "$STRONG_MODEL" ]; then
    echo "Testing strong backdoor model..."
    python test_trigger_focused.py --model-path "$STRONG_MODEL"
elif [ -d "$ULTRA_MODEL" ]; then
    echo "Testing ultra backdoor model..."
    python test_trigger_focused.py --model-path "$ULTRA_MODEL"
fi

echo "Testing complete." 