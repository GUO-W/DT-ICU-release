# log plot with diff threshold

import os
import re
import matplotlib.pyplot as plt


SPLITS = ['train', 'val', 'test']
DEFAULT_THRESHOLD = 0.5

# Regex patterns for parsing
CLASS_THRESHOLD_PATTERN = re.compile(r"^class threshold\s*[:=]?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
EPOCH_PATTERN           = re.compile(r"=+\s*EPOCH\s+([0-9]+(?:\.[0-9]*)?)\s*=+", re.IGNORECASE)
VALIDATION_PATTERN      = re.compile(r"=+\s*VALIDATION\s*=+", re.IGNORECASE)
TESTING_PATTERN         = re.compile(r"=+\s*TESTING\s*=+", re.IGNORECASE)
METRIC_PATTERN          = re.compile(r"^(BCE Loss|AU-ROC|AU-PRC|Accuracy|Precision|Recall):\s*([0-9\.Ee+-]+)")


def parse_log_file(file_path):
    """
    Parse one log file, extracting metrics per epoch, split, and classification threshold.
    Returns: { threshold: { split: { epoch: {metric: value} } } }
    """
    data = {}
    current_threshold = DEFAULT_THRESHOLD
    current_split = None
    current_epoch = None

    # Initialize default threshold
    data[current_threshold] = {s: {} for s in SPLITS}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect class threshold lines
            m_thr = CLASS_THRESHOLD_PATTERN.match(line)
            if m_thr:
                current_threshold = float(m_thr.group(1))
                if current_threshold not in data:
                    data[current_threshold] = {s: {} for s in SPLITS}
                continue

            # Detect epoch blocks (start of training metrics)
            m_ep = EPOCH_PATTERN.match(line)
            if m_ep:
                current_epoch = int(float(m_ep.group(1)))
                current_split = 'train'
                data[current_threshold][current_split].setdefault(current_epoch, {})
                continue

            # Detect validation and testing sections
            if VALIDATION_PATTERN.match(line):
                current_split = 'val'
                continue
            if TESTING_PATTERN.match(line):
                current_split = 'test'
                continue

            # Detect metric lines
            m_met = METRIC_PATTERN.match(line)
            if m_met and current_split and current_epoch is not None:
                metric_name = m_met.group(1)
                metric_value = float(m_met.group(2))
                data[current_threshold][current_split].setdefault(current_epoch, {})
                data[current_threshold][current_split][current_epoch][metric_name] = metric_value

    print(f"Detected thresholds in {os.path.basename(file_path)}: {sorted(data.keys())}")
    return data


def plot_metrics(experiments, METRICS):
    """
    Plot metrics over epochs for each experiment and threshold.
    Rows = metrics; Columns = splits.
    Each curve gets a unique, vivid color from a shared palette,
    without maintaining a single color family.
    """
    n_metrics = len(METRICS)
    n_splits = len(SPLITS)
    fig, axes = plt.subplots(n_metrics, n_splits, figsize=(n_splits*5, n_metrics*3), sharex=True)

    # Collect all values to set uniform y-limits per metric
    all_vals = {m: [] for m in METRICS}
    # Use a qualitative palette with many distinct colors
    palette = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors

    # Flatten experiments+thresholds to assign unique colors
    curve_colors = []
    for exp_idx, (exp_name, cfg) in enumerate(experiments.items()):
        data = cfg['data']
        thr_spec = cfg['thresholds']
        thresholds = sorted(data.keys()) if thr_spec == 'all' else [t for t in thr_spec if t in data]
        for thr in thresholds:
            curve_colors.append((exp_name, thr))
    # Assign color per curve
    color_map = {ct: palette[i % len(palette)] for i, ct in enumerate(curve_colors)}

    for exp_name, cfg in experiments.items():
        data = cfg['data']
        thr_spec = cfg['thresholds']
        thresholds = sorted(data.keys()) if thr_spec == 'all' else [t for t in thr_spec if t in data]

        for thr in thresholds:
            color = color_map[(exp_name, thr)]
            label = f"{exp_name} thr={thr}"
            for col, split in enumerate(SPLITS):
                epochs = sorted(data[thr].get(split, {}).keys())
                for row, metric in enumerate(METRICS):
                    xs, ys = [], []
                    for e in epochs:
                        v = data[thr][split][e].get(metric)
                        if v is not None:
                            xs.append(e+1)
                            ys.append(v)
                            all_vals[metric].append(v)
                    ax = axes[row][col] if n_metrics > 1 else axes[col]
                    if xs:
                        ax.plot(xs, ys, marker='o', color=color, label=label)
                    if row == 0:
                        ax.set_title(split.capitalize())
                    if col == 0:
                        ax.set_ylabel(metric)
                    if row == n_metrics - 1:
                        ax.set_xlabel('Epoch')
                    ax.set_xlim(1, 20)
                    ax.set_xticks(range(1, 21))
                    ax.grid(True)

    # Uniform y-limits per metric
    for r, metric in enumerate(METRICS):
        vals = all_vals[metric]
        if not vals: continue
        lo, hi = min(vals), max(vals)
        if lo == hi:
            d = 0.1 if hi == 0 else 0.1 * abs(hi)
            lo, hi = lo - d, hi + d
        else:
            m = 0.05 * (hi - lo)
            lo, hi = lo - m, hi + m
        for c in range(n_splits):
            ax = axes[r][c] if n_metrics > 1 else axes[c]
            ax.set_ylim(lo, hi)

    # Consolidate legend
    ax0 = axes[0][0] if n_metrics > 1 else axes[0]
    handles, labels = ax0.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc='upper center', ncol=4, fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join('log', 'log_anal.png')
    plt.savefig(outpath)
    print(f"Saved plot to {outpath}")

if __name__ == '__main__':
    METRICS = ['BCE Loss', 'AU-ROC', 'AU-PRC', 'Accuracy', 'Precision', 'Recall']
    log_settings = [
        {'file': 'log_oversampling_bz16_lr1e-5.txt', 'thresholds': [0.5]}, #'all'}, # recall vs precision -- choose recall --thres=0.5
        #{'file': 'log_test.txt', 'thresholds': [0.8]}
    ]
    experiments = {}
    for cfg in log_settings:
        path = os.path.join('log', cfg['file'])
        if not os.path.exists(path): print(f"Missing {cfg['file']}"); continue
        experiments[cfg['file']] = {'data': parse_log_file(path), 'thresholds': cfg['thresholds']}
    if experiments:
        plot_metrics(experiments, METRICS)
