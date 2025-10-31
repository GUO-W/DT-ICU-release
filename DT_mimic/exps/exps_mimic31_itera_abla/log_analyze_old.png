# log plot, similar with tensorboard
import os
import re
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    """
    Parse a single log file and extract metric values per epoch for training,
    validation, and testing.

    Expected structure is similar to:

        ======= EPOCH 0 =======
        Mode:  train
        BCE Loss: 0.32
        AU-ROC: 0.66
        ...
        ======= VALIDATION =======
        [metrics...]
        ======= TESTING =======
        [metrics...]

    Returns:
        A dictionary with keys 'train', 'val', and 'test'.
        Each maps to another dictionary whose keys are epoch numbers (integers)
        and whose values are dictionaries mapping metric names to float values.
    """
    results = {split: {} for split in SPLITS}

    # Compile regular expression patterns.
    epoch_pattern = re.compile(r"=+\s*EPOCH\s+(\d+(?:\.\d*)?)\s*=+")
    validation_pattern = re.compile(r"=+\s*VALIDATION\s*=+")
    testing_pattern = re.compile(r"=+\s*TESTING\s*=+")
    metric_pattern = re.compile(r"^(BCE Loss|AU-ROC|AU-PRC|Accuracy|Precision|Recall):\s*([\d\.Ee+-]+)")

    current_epoch = None   # Current epoch (integer).
    current_split = None   # One of 'train', 'val', or 'test'.

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for the start of a new epoch block (training metrics).
            match = epoch_pattern.match(line)
            if match:
                current_epoch = int(float(match.group(1)))
                current_split = 'train'
                results[current_split].setdefault(current_epoch, {})
                continue

            # Check for the start of the validation block.
            if validation_pattern.match(line):
                current_split = 'val'
                if current_epoch is not None:
                    results[current_split].setdefault(current_epoch, {})
                continue

            # Check for the start of the testing block.
            if testing_pattern.match(line):
                current_split = 'test'
                if current_epoch is not None:
                    results[current_split].setdefault(current_epoch, {})
                continue

            # Check if the line contains one of the desired metrics.
            metric_match = metric_pattern.match(line)
            if metric_match and current_split is not None and current_epoch is not None:
                metric_name = metric_match.group(1)
                try:
                    metric_value = float(metric_match.group(2))
                except ValueError:
                    continue  # Skip the line if conversion fails.
                results[current_split][current_epoch][metric_name] = metric_value

    return results

def plot_metrics(experiments):
    """
    Create a figure with one row per metric (based on the METRICS list) and three
    columns (one for each split: training, validation, testing). Each subplot shows
    curves (one per experiment) plotted over epochs. The x-axis uses epochs 1 to 20
    (by adding 1 to the parsed epoch value).

    For each metric (row), the y-axis is automatically computed from the data across
    all experiments and splits and the same y-range is applied for the three subplots,
    facilitating horizontal comparison.

    The figure is saved as 'log_anal.png' in the './log' folder.
    """
    num_metrics = len(METRICS)
    num_splits = len(SPLITS)
    fig, axes = plt.subplots(num_metrics, num_splits, figsize=(num_splits * 5, num_metrics * 3), sharex=True)

    # Dictionary to accumulate all y-values for each metric across experiments and splits.
    metric_y_values = {metric: [] for metric in METRICS}

    # Loop over experiments and plot each curve.
    for exp_name, exp_data in experiments.items():
        for col, split in enumerate(SPLITS):
            # Get and sort the epoch values and map them: epoch 0 -> 1, 1 -> 2, etc.
            epochs = sorted(exp_data.get(split, {}).keys())
            for row, metric in enumerate(METRICS):
                x_values = []
                y_values = []
                for epoch in epochs:
                    value = exp_data[split].get(epoch, {}).get(metric)
                    if value is not None:
                        x_values.append(epoch + 1)
                        y_values.append(value)
                        metric_y_values[metric].append(value)
                ax = axes[row][col] if num_metrics > 1 else axes[col]
                if x_values and y_values:
                    ax.plot(x_values, y_values, marker='o', label=exp_name)
                if row == 0:
                    ax.set_title(split.capitalize())
                if col == 0:
                    ax.set_ylabel(metric)
                # Set x-axis limits and ticks.
                ax.set_xlim(1, 20)
                ax.set_xticks(range(1, 21))
                ax.grid(True)
                ax.legend(fontsize='small')
                if row == num_metrics - 1:
                    ax.set_xlabel("Epoch")

    # For each metric, adjust the y-axis limits across all three splits.
    for row, metric in enumerate(METRICS):
        all_y = metric_y_values[metric]
        if all_y:
            y_min = min(all_y)
            y_max = max(all_y)
            # If all y-values are equal, add a small padding.
            if y_max == y_min:
                delta = 0.1 if y_max == 0 else 0.1 * abs(y_max)
                y_min, y_max = y_min - delta, y_max + delta
            else:
                margin = 0.05 * (y_max - y_min)
                y_min -= margin
                y_max += margin
            for col in range(num_splits):
                ax = axes[row][col] if num_metrics > 1 else axes[col]
                ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    output_path = os.path.join("log", "log_anal.png")
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == '__main__':
    # Manually define the metrics you want to track. For example, to only plot "BCE Loss" and "Accuracy", set: METRICS = ['BCE Loss', 'Accuracy']
    # METRICS = ['BCE Loss', 'AU-ROC', 'AU-PRC', 'Accuracy', 'Precision', 'Recall']
    METRICS = ['BCE Loss',  'Accuracy', 'Precision', 'Recall']

    # Manually define the names of log files (located in ./log) to compare.
    log_files = [
        #"log_data1w-only72h_bz1_lr1e-5_pos_pred.txt",
        #"log_data1w-only72h_bz1_lr1e-5.txt", 
        "log_data1w-only72h_bz1_lr1e-6.txt", # 1e-5 overfit, reduce lr
        #"log_data1w-only72h_oversampling_bz1_lr1e-6.txt" # bad precision and recall, add data oversampling (in dataset.py, config option oversampling=true)
        # "log_data1w-only72h_oversampling_bz256_lr1e-5.txt", # data from gendata_balance.py config oversampling=false -- YES! but val bce raise -- try lr dicrease
        "log_data1w-only72h_oversampling_bz256_lr1e-6.txt", # test lr
        "log_data1w-only72h_oversampling_bz256_lr5e-7.txt", # test lr -> yes.
        "log_data1w-min72h_oversampling_bz256_lr5e-7.txt" # >= 72h
    ]

    # The three splits (columns) are fixed: training, validation, testing.
    SPLITS = ['train', 'val', 'test']

    experiments = {}
    log_directory = "log"
    for file_name in log_files:
        full_path = os.path.join(log_directory, file_name)
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} does not exist!")
            continue
        experiments[file_name] = parse_log_file(full_path)

    if not experiments:
        print("No valid experiments were found. Check your log file names and paths.")

    plot_metrics(experiments)
