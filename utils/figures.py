import os
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import norm

# Register the font with Matplotlib
font_path = "./figure_font/times.ttf"
fm.fontManager.addfont(font_path)

# Get font family
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()

# Set globally
matplotlib.rcParams['font.family'] = font_name
matplotlib.rcParams.update({
    "axes.titlesize": 33,
    "axes.labelsize": 33,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 18,
})

def barplot(y, trait_name, save_dir, category=None):
    
    y = np.asarray(y)
    
    # Count values
    low_count = np.sum(y==0)
    high_count = np.sum(y==1)

    # Bar labels and heights
    labels = [f'Low {trait_name}', f'High {trait_name}']
    counts = [low_count, high_count]

    total = low_count + high_count

    plt.figure(figsize=(14, 12))
    bars = plt.bar(labels, counts, color=['#4C72B0', '#DD8452'], width=0.8, edgecolor='black', linewidth=1.2)

    # Title and labels
    plt.title(f'{trait_name}')
    plt.ylabel('Count')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,  height + height * 0.005, f'{int(height)}', ha='center', va='bottom', fontsize=18, fontweight='bold')
    plt.tight_layout()

    # Move ticks inside the plot
    plt.tick_params(
        axis='both',       # Apply to both x and y
        direction='in',    # Ticks go inside
        length=6,          # Tick length
        width=1.2,         # Tick width
        colors='black',    # Tick color
        top=False,          # Also draw top ticks
        right=False         # Also draw right ticks
    )
    if category:
        os.makedirs(os.path.join(save_dir, "distribution", f"{category}"), exist_ok=True)
        filename = f"distribution/{category}/{trait_name}_barplot.png"
    else:
        os.makedirs(os.path.join(save_dir, "distribution"), exist_ok=True)
        filename = f"distribution/{trait_name}_barplot.png"

    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close

def plot_loss(train_loss, validation_loss, test_loss, trait_name, save_dir):
    train_loss = np.asarray(train_loss)
    validation_loss = np.asarray(validation_loss)
    test_loss = np.asarray(test_loss)

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(16, 10))
    plt.plot(epochs, train_loss, label="Train Loss", marker='o')
    plt.plot(epochs, validation_loss, label="Validation Loss", marker='s')
    plt.plot(epochs, test_loss, label="Test Loss", marker='^')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    
    y_min = min(train_loss.min(), validation_loss.min(), test_loss.min())
    y_max = max(train_loss.max(), validation_loss.max(), test_loss.max())
    plt.ylim(y_min, y_max)
    plt.xlim(1, len(epochs))

    # legend
    leg = plt.legend(frameon=True, edgecolor="black", labelspacing=0.3)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.8)  # 0 = fully transparent, 1 = opaque
    leg.get_frame().set_edgecolor("gray")

    plt.tight_layout()

    # Move ticks inside the plot
    plt.tick_params(
        axis='both',       # Apply to both x and y
        direction='in',    # Ticks go inside
        length=6,          # Tick length
        width=1.2,         # Tick width
        colors='black',    # Tick color
        top=False,          # Also draw top ticks
        right=False         # Also draw right ticks
    )

    filename = f"{trait_name}_loss.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()

def plot_accuracy(train_accuracy, validation_accuracy, test_accuracy, trait_name, save_dir):
    train_accuracy = np.asarray(train_accuracy)
    validation_accuracy = np.asarray(validation_accuracy)
    test_accuracy = np.asarray(test_accuracy)
    
    epochs = np.arange(1, len(train_accuracy) + 1)
    plt.figure(figsize=(16, 10))
    plt.plot(epochs, train_accuracy, label="Train Accuracy", marker='o')
    plt.plot(epochs, validation_accuracy, label="Validation Accuracy", marker='s')
    plt.plot(epochs, test_accuracy, label="Test Accuracy", marker='^')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)

    y_min = min(train_accuracy.min(), validation_accuracy.min(), test_accuracy.min())
    y_max = max(train_accuracy.max(), validation_accuracy.max(), test_accuracy.max())
    plt.ylim(y_min, y_max)
    plt.xlim(1, len(epochs))

    # legend
    leg = plt.legend(frameon=True, edgecolor="black", labelspacing=0.3)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.8)  # 0 = fully transparent, 1 = opaque
    leg.get_frame().set_edgecolor("gray")

    plt.tight_layout()

    # Move ticks inside the plot
    plt.tick_params(
        axis='both',       # Apply to both x and y
        direction='in',    # Ticks go inside
        length=6,          # Tick length
        width=1.2,         # Tick width
        colors='black',    # Tick color
        top=False,          # Also draw top ticks
        right=False         # Also draw right ticks
    )

    filename = f"{trait_name}_accuracy.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()

def plot_roc(results, trait_name, save_dir):
    plt.figure(figsize=(16, 10))

    all_fprs = []
    all_tprs = []

    for label, probs, true_labels in results:
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.4f})")

        all_fprs.append(fpr)
        all_tprs.append(tpr)

    all_fprs = np.concatenate(all_fprs)
    all_tprs = np.concatenate(all_tprs)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # x_min, x_max = all_fprs.min(), all_fprs.max()
    # y_min, y_max = all_tprs.min(), all_tprs.max()

    # Slight zoom outward but keep ticks untouched
    plt.xlim(-0.001, 1.0)
    plt.ylim(-0.001, 1.0)

    # Force ticks at nice clean values
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1g'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1g'))

    # legend
    leg = plt.legend(frameon=True, edgecolor="black", labelspacing=0.3)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.8)  # 0 = fully transparent, 1 = opaque
    leg.get_frame().set_edgecolor("gray")
    plt.tight_layout()

    # Move ticks inside the plot
    plt.tick_params(
        axis='both',       # Apply to both x and y
        direction='in',    # Ticks go inside
        length=6,          # Tick length
        width=1.2,         # Tick width
        colors='black',    # Tick color
        top=False,          # Also draw top ticks
        right=False         # Also draw right ticks
    )

    filename = f"{trait_name}_roc.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()

def plot_pr(results, trait_name, save_dir):
    plt.figure(figsize=(16, 10))

    # Track global min/max for tight bounding box
    global_min_recall = 1.0
    global_max_recall = 0.0
    global_min_precision = 1.0
    global_max_precision = 0.0

    for label, probs, true_labels in results:
        precision, recall, _ = precision_recall_curve(true_labels, probs)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, label=f"{label} (AUC = {pr_auc:.4f})")

        # Track min/max for later axis cropping
        global_min_recall = min(global_min_recall, recall.min())
        global_max_recall = max(global_max_recall, recall.max())
        global_min_precision = min(global_min_precision, precision.min())
        global_max_precision = max(global_max_precision, precision.max())

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.xlim(global_min_recall, global_max_recall)
    plt.ylim(global_min_precision, global_max_precision)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1g'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1g'))
    
    # legend
    leg = plt.legend(frameon=True, edgecolor="black", labelspacing=0.3)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.8)  # 0 = fully transparent, 1 = opaque
    leg.get_frame().set_edgecolor("gray")
    plt.tight_layout()

    # Move ticks inside the plot
    plt.tick_params(
        axis='both',       # Apply to both x and y
        direction='in',    # Ticks go inside
        length=6,          # Tick length
        width=1.2,         # Tick width
        colors='black',    # Tick color
        top=False,          # Also draw top ticks
        right=False         # Also draw right ticks
    )    

    plt.tight_layout()
    
    filename = f"{trait_name}_pr.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()
