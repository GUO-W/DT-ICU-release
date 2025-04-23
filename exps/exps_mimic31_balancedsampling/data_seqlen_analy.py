# import matplotlib.pyplot as plt
# from collections import Counter
# import numpy as np
# import matplotlib.ticker as mtick

# def analyze_file(input_file: str, output_image: str) -> None:
#     """
#     calculates the distribution of stay len in hours, creates a PNG file containing four subplots to analyse the distributions of the train/test/val seperately.
#     Args:
#         input_file (str): Path to the input text file.
#         output_image (str): Path to save the output PNG image.
#     Expected file format (each line):
#         <sample_id> <number_of_hours>
#     """
#     # Read and parse the file.
#     hours_list = []
#     with open(input_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 try:
#                     hours = int(parts[1])
#                     hours_list.append(hours)
#                 except ValueError:
#                     continue

#     if not hours_list:
#         print("No valid data found.")
#         return

#     # Full frequency distribution.
#     distribution = Counter(hours_list)
#     sorted_hours = sorted(distribution.keys())
#     counts = [distribution[hour] for hour in sorted_hours]

#     # Determine cutoff for zooming in (90th percentile).
#     cutoff = np.percentile(hours_list, 90)
#     lower_hours = [hour for hour in sorted_hours if hour <= cutoff]
#     lower_counts = [distribution[hour] for hour in lower_hours]
#     upper_hours = [hour for hour in sorted_hours if hour > cutoff]
#     upper_counts = [distribution[hour] for hour in upper_hours]

#     # ---------------------------
#     # Define custom bins for the pie chart.
#     # Base important breakpoints.
#     base_bins = [0, 6, 12, 24, 48, 72]
#     max_val = max(hours_list)
#     if max_val > 72:
#         # Partition from 72 to max_val into 3 bins.
#         additional_bins = np.linspace(72, max_val, num=4).tolist()  # includes 72 and max_val
#         additional_bins = additional_bins[1:]  # remove duplicate 72
#         bins = base_bins + additional_bins + [float('inf')]
#     else:
#         bins = base_bins + [float('inf')]
#     # Use np.histogram to get counts for each bin.
#     counts_bins, _ = np.histogram(hours_list, bins=bins)
#     total_count = len(hours_list)
    
#     # Create legend labels and bin range labels.
#     legend_labels = []
#     bin_range_labels = []
#     for i in range(len(bins) - 1):
#         low = int(bins[i])
#         if bins[i+1] == float('inf'):
#             range_label = f"{low}+"
#         else:
#             # Subtract 1 from the upper bound for a closed interval representation.
#             high = int(bins[i+1]) - 1
#             range_label = f"{low}-{high}"
#         bin_range_labels.append(range_label)
#         count = counts_bins[i]
#         percentage = count / total_count * 100
#         legend_labels.append(f"{range_label}: {count} ({percentage:.1f}%)")
    
#     # ---------------------------
#     # Print some summary info.
#     print("Full distribution:")
#     for hour, count in zip(sorted_hours, counts):
#         print(f"{hour}: {count}")
#     print(f"\nCutoff (90th percentile): {cutoff:.1f}")
#     print("\nLower zoomed-in distribution (hours <= cutoff):")
#     for hour, count in zip(lower_hours, lower_counts):
#         print(f"{hour}: {count}")
#     print("\nUpper zoomed-in distribution (hours > cutoff):")
#     for hour, count in zip(upper_hours, upper_counts):
#         print(f"{hour}: {count}")
    
#     # ---------------------------
#     # Create figure with custom layout:
#     # Top row: three bar plots (full, zoomed-in lower, zoomed-in upper)
#     # Bottom row: one pie chart spanning all columns.
#     fig = plt.figure(constrained_layout=True, figsize=(18, 12))
#     gs = fig.add_gridspec(2, 3)
    
#     # (1) Full distribution bar plot (top left).
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.bar(sorted_hours, counts, color='skyblue')
#     ax1.set_title("Full Distribution of Number of Hours")
#     ax1.set_xlabel("Number of Hours")
#     ax1.set_ylabel("Frequency")
#     # Denser y-axis: set a smaller tick step.
#     step1 = max(counts) / 20 if max(counts) else 1
#     ax1.yaxis.set_major_locator(mtick.MultipleLocator(step1))
#     ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # (2) Zoomed-in bar plot for lower values (top middle).
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax2.bar(lower_hours, lower_counts, color='lightgreen')
#     ax2.set_title(f"Zoomed-In Distribution (<= {cutoff:.1f} Hours)")
#     ax2.set_xlabel("Number of Hours")
#     ax2.set_ylabel("Frequency")
#     step2 = max(lower_counts) / 20 if max(lower_counts) else 1
#     ax2.yaxis.set_major_locator(mtick.MultipleLocator(step2))
#     ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # (3) Zoomed-in bar plot for upper values (top right).
#     ax3 = fig.add_subplot(gs[0, 2])
#     ax3.bar(upper_hours, upper_counts, color='salmon')
#     ax3.set_title(f"Zoomed-In Distribution (> {cutoff:.1f} Hours)")
#     ax3.set_xlabel("Number of Hours")
#     ax3.set_ylabel("Frequency")
#     step3 = max(upper_counts) / 20 if max(upper_counts) else 1
#     ax3.yaxis.set_major_locator(mtick.MultipleLocator(step3))
#     ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # (4) Pie chart with custom bins.
#     ax4 = fig.add_subplot(gs[1, :])
#     wedges, texts = ax4.pie(counts_bins, startangle=90, counterclock=False, wedgeprops=dict(width=0.5))
#     ax4.set_title("Pie Chart of Binned Distribution\n(Custom Intervals)")
#     # Add legend to the right of the pie chart.
#     ax4.legend(wedges, legend_labels, title="Intervals", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
#     # Annotate each pie slice with the bin range (on the edge of the pie).
#     for i, wedge in enumerate(wedges):
#         angle = (wedge.theta2 + wedge.theta1) / 2.0
#         x = np.cos(np.deg2rad(angle))
#         y = np.sin(np.deg2rad(angle))
#         # Position the annotation slightly outside the wedge.
#         horizontal_alignment = "left" if x > 0 else "right"
#         ax4.annotate(bin_range_labels[i], xy=(x, y), xytext=(1.1 * x, 1.1 * y),
#                      horizontalalignment=horizontal_alignment, verticalalignment="center",
#                      fontsize=10, color='black')
    
#     # Save the entire figure to the specified output image.
#     plt.savefig(output_image, bbox_inches="tight")
#     print(f"\nAll plots saved to {output_image}")
#     plt.show()


# if __name__ == "__main__":
#     analyze_file("data/train_seqlen.txt", "data/train_seqlen_distribution.png")
#     analyze_file("data/val_seqlen.txt", "data/val_seqlen_distribution.png")
#     analyze_file("data/test_seqlen.txt", "data/test_seqlen_distribution.png")
    

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.cm as cm

def read_hours(filename: str) -> list:
    """Read a file and return a list of the second column (hours) as integers."""
    hours = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    hours.append(int(parts[1]))
                except ValueError:
                    continue
    return hours

def get_custom_bins(all_hours: list) -> list:
    """
    Define custom bins based on important breakpoints (6, 12, 24, 48, 72).
    If the global maximum is > 72, further subdivide the range above 72 into 3 bins.
    Always append an 'infinite' bin at the end.
    """
    base_bins = [0, 6, 12, 24, 48, 72]
    global_max = max(all_hours)
    if global_max > 72:
        # Partition from 72 to global_max into 3 bins (4 boundaries: 72, b1, b2, global_max)
        additional_bins = np.linspace(72, global_max, num=4).tolist()
        additional_bins = additional_bins[1:]  # remove the duplicate 72
        bins = base_bins + additional_bins
    else:
        bins = base_bins
    bins.append(float('inf'))
    return bins

def get_bin_labels(bins: list) -> list:
    """
    Create labels for each bin. For example, if bin edges are 0 and 6, the label will be "0-5".
    The final bin is shown as "X+".
    """
    labels = []
    for i in range(len(bins) - 1):
        low = int(bins[i])
        if bins[i+1] == float('inf'):
            labels.append(f"{low}+")
        else:
            high = int(bins[i+1]) - 1
            labels.append(f"{low}-{high}")
    return labels

def compute_histogram(data: list, bins: list) -> np.ndarray:
    """Compute histogram counts using the provided bins."""
    counts, _ = np.histogram(np.array(data), bins=bins)
    return counts

def plot_nested_donuts(train_counts, val_counts, test_counts, bin_labels):
    """
    Plot three nested donut charts for train, val, and test distributions.
    Outer ring: training data, Middle ring: validation data, Inner ring: test data.
    All rings use the same colors for corresponding bins.
    """
    n_bins = len(bin_labels)
    # Use a colormap to assign colors to each bin.
    colors = cm.tab20(np.linspace(0, 1, n_bins))

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Outer donut for training data (radius 1.0, width 0.3)
    wedges_train, _ = ax.pie(train_counts, radius=1.0, colors=colors,
                               startangle=90, counterclock=False,
                               wedgeprops=dict(width=0.3, edgecolor='white'))
    # Middle donut for validation data (radius 0.7, width 0.3)
    wedges_val, _ = ax.pie(val_counts, radius=0.7, colors=colors,
                           startangle=90, counterclock=False,
                           wedgeprops=dict(width=0.3, edgecolor='white'))
    # Inner donut for test data (radius 0.4, width 0.3)
    wedges_test, _ = ax.pie(test_counts, radius=0.4, colors=colors,
                            startangle=90, counterclock=False,
                            wedgeprops=dict(width=0.3, edgecolor='white'))
    
    # Set the title
    ax.set_title("Train/Val/Test Distribution Comparison", fontsize=16)
    
    # Create a legend that shows the bin intervals.
    legend_elements = [Patch(facecolor=colors[i], edgecolor='white', label=bin_labels[i]) 
                       for i in range(n_bins)]
    ax.legend(handles=legend_elements, title="Bins", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    return fig, ax

def main():
    # File paths
    train_file = "data/train_seqlen.txt"
    val_file   = "data/val_seqlen.txt"
    test_file  = "data/test_seqlen.txt"

    # Read data from files.
    train_hours = read_hours(train_file)
    val_hours   = read_hours(val_file)
    test_hours  = read_hours(test_file)
    
    # Get global bins based on all data.
    all_hours = train_hours + val_hours + test_hours
    bins = get_custom_bins(all_hours)
    bin_labels = get_bin_labels(bins)
    
    # Compute histograms for each dataset.
    train_counts = compute_histogram(train_hours, bins)
    val_counts   = compute_histogram(val_hours, bins)
    test_counts  = compute_histogram(test_hours, bins)
    
    # Create nested donut charts.
    fig, ax = plot_nested_donuts(train_counts, val_counts, test_counts, bin_labels)
    
    # Optionally, add text annotations for each ring.
    # For brevity, here we only rely on the legend.
    
    # Save and show the plot.
    output_image = "data/combined_seqlen_distribution.png"
    plt.savefig(output_image, bbox_inches="tight")
    print(f"Combined distribution plot saved to {output_image}")
    plt.show()

if __name__ == "__main__":
    main()
