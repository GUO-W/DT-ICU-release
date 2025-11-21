#!/bin/bash
set -e  # stop on first error

LOGFILE="log/test_lomo_all.txt"
CKPT="snapshot_epoch19.pth"

# Create log directory if not exists
mkdir -p log

# Clear previous log
> "$LOGFILE"

# Define all modes ("" means baseline)
modes=("" "meds" "chart" "out" "proc" "date" "ing" "static" "stat" "demo")

for mode in "${modes[@]}"; do
    echo "" >> "$LOGFILE"
    if [ -z "$mode" ]; then
        echo "========== Running baseline ==========" | tee -a "$LOGFILE"
        python test_only.py --ckpt "$CKPT" >> "$LOGFILE" 2>&1
    else
        echo "========== Running zero_inputs: $mode ==========" | tee -a "$LOGFILE"
        python test_only.py --ckpt "$CKPT" --zero_inputs "$mode" >> "$LOGFILE" 2>&1
    fi
done

echo "" >> "$LOGFILE"
echo "========== All tests completed ==========" | tee -a "$LOGFILE"




#python test_only.py --ckpt snapshot_epoch19.pth > log/test_lomo_baseline.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs meds > log/test_lomo_meds.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs chart > log/test_lomo_chart.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs out > log/test_lomo_out.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs proc > log/test_lomo_proc.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs date > log/test_lomo_date.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs ing > log/test_lomo_ing.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs static > log/test_lomo_static.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs stat > log/test_lomo_stat.txt
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs demo > log/test_lomo_demo.txt
