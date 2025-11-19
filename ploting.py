import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Data
methods = ['Snorkel', '2nd-order', '3rd-order', '4th-order']
improvements = [32, 46, 41, 43]  # in percentage
baseline = 0  # baseline for semi-supervised method
accuracies = [baseline + imp for imp in improvements]

# Plot
plt.figure(figsize=(5.5, 4))
bar_width = 0.6  # narrower bar width
bar_positions = range(len(methods))
bar_colors = sns.color_palette("Blues_d", len(methods))
bars = plt.bar(bar_positions, accuracies, color=bar_colors, width=bar_width)

# Highlight baseline
# plt.axhline(y=baseline, color='red', linestyle='--', linewidth=1.5, label='Semi-supervised baseline (100%)')

# Annotate bars
for pos, acc in zip(bar_positions, accuracies):
    plt.text(pos, acc + 1, f'{acc}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Labels and title
plt.xlabel('Weak Supervised Methods', fontsize=12)
plt.ylabel('Accuracy Improvement(%)', fontsize=12)
# plt.title('Accuracy Improvement over Semi-supervised Method', fontsize=16, fontweight='bold')
plt.xticks(bar_positions, methods, fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(range(0, 51, 10), fontsize=12)
plt.legend()
plt.tight_layout()

plt.show()
