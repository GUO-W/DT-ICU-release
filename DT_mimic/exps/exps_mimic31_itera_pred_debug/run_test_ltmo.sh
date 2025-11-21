#!/bin/bash
set -e  # stop on first error

LOGFILE="log/test_ltmo_all.txt"
CKPT="snapshot_epoch19.pth"

mkdir -p log
> "$LOGFILE"

# all single-modality names (no baseline here)
modes=("meds" "chart" "out" "proc" "date" "ing" "static" "stat" "demo")

echo "========== Running LTMO (leave-two-modalities-out) ==========" | tee -a "$LOGFILE"

# outer loop over first modality
for ((i=0; i<${#modes[@]}; i++)); do
    m1=${modes[$i]}
    # inner loop over second modality, start from i+1 to avoid duplicates
    for ((j=i+1; j<${#modes[@]}; j++)); do
        m2=${modes[$j]}
        echo "" >> "$LOGFILE"
        echo "========== Running zero_inputs: ${m1},${m2} ==========" | tee -a "$LOGFILE"
        python test_only.py --ckpt "$CKPT" --zero_inputs "${m1},${m2}" >> "$LOGFILE" 2>&1
    done
done

echo "" >> "$LOGFILE"
echo "========== LTMO tests completed ==========" | tee -a "$LOGFILE"

