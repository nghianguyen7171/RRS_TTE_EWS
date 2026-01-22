#!/bin/bash
# Script to check training status and update report when complete

cd "/media/nghia/Nguyen NghiaW/RRS_3y_10y/time_to_event"

echo "=== Training Status Check ==="
echo ""

# Check if training process is running
TRAINING_PID=$(ps aux | grep "train_all_models" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAINING_PID" ]; then
    echo "✓ Training has completed!"
    echo ""
    echo "Checking results..."
    
    # Count model files
    MODEL_COUNT=$(ls -1 results/python_baseline/models/*.pkl 2>/dev/null | wc -l)
    echo "Model files found: $MODEL_COUNT"
    
    # Check metrics
    if [ -f "results/python_baseline/metrics/all_metrics_3year.csv" ]; then
        echo ""
        echo "Current metrics:"
        cat results/python_baseline/metrics/all_metrics_3year.csv
    fi
    
    # Regenerate report
    echo ""
    echo "Regenerating report..."
    python3 python/generate_report.py --dataset 3year
    
    echo ""
    echo "✓ Report updated!"
    echo "Location: results/python_baseline/PYTHON_BASELINE_REPORT_3year.md"
else
    echo "Training still running (PID: $TRAINING_PID)"
    echo "Elapsed time: $(ps -p $TRAINING_PID -o etime= 2>/dev/null || echo 'unknown')"
    echo "CPU usage: $(ps -p $TRAINING_PID -o pcpu= 2>/dev/null || echo 'unknown')%"
    echo "Memory usage: $(ps -p $TRAINING_PID -o pmem= 2>/dev/null || echo 'unknown')%"
    echo ""
    echo "Check again later or run: bash python/check_training_status.sh"
fi
