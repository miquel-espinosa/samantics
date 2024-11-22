import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_tensorboard_data(log_dirs):
    """Load and merge TensorBoard data from multiple event files."""
    if not isinstance(log_dirs, list):
        log_dirs = [log_dirs]
    
    merged_data = {}
    
    for log_dir in log_dirs:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        
        for tag in tags:
            events = ea.Scalars(tag)
            df = pd.DataFrame([(e.step, e.value) for e in events],
                            columns=['step', 'value'])
            
            if tag not in merged_data:
                merged_data[tag] = df
            else:
                merged_data[tag] = pd.concat([merged_data[tag], df])
                merged_data[tag] = merged_data[tag].sort_values('step').reset_index(drop=True)
    
    return merged_data

def exponential_moving_average(data, alpha=0.1):
    """Calculate EMA for smoothing."""
    return data.ewm(alpha=alpha, adjust=False).mean()

# Set the style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_style("whitegrid", {
    # 'axes.facecolor': '.95',
    'grid.color': '.8',
    'grid.linestyle': '-',
    'grid.linewidth': 1.5
})

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

experiment_ids = [
    ["./lightning_logs/clip/version_0"],
    ["./lightning_logs/dinov2/version_0"],
    ["./lightning_logs/sam/version_0", "./lightning_logs/sam/version_1"],
    ["./lightning_logs/sam2/version_0"],
]

# Create figure with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=300, sharex=True)
fig.patch.set_facecolor('white')

# Add small space between subplots
plt.subplots_adjust(hspace=0.1)

# Get color palette
colors = sns.color_palette("husl", len(experiment_ids))

# Plot training loss for all experiments
for idx, exp_paths in enumerate(experiment_ids):
    exp_name = os.path.basename(os.path.dirname(exp_paths[0]))
    tb_data = load_tensorboard_data(exp_paths)
    
    if 'train_loss' in tb_data:
        # Plot original loss with low alpha
        ax1.plot(tb_data['train_loss']['step'], tb_data['train_loss']['value'],
                color=colors[idx], linewidth=1, alpha=0.3)
        
        # Calculate and plot smoothed loss
        smoothed_data = tb_data['train_loss'].copy()
        smoothed_data['value'] = exponential_moving_average(smoothed_data['value'])
        ax1.plot(smoothed_data['step'], smoothed_data['value'],
                color=colors[idx], linewidth=2, alpha=0.9,
                label=exp_name)

ax1.set_ylabel('Training Loss')
ax1.grid(True, alpha=0.5)
# ax1.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left',
#           title='Models', title_fontsize=12, 
#           frameon=True, fancybox=True)

# Plot validation accuracy for all experiments
for idx, exp_paths in enumerate(experiment_ids):
    exp_name = os.path.basename(os.path.dirname(exp_paths[0]))
    tb_data = load_tensorboard_data(exp_paths)
    
    if 'val_acc' in tb_data:
        ax2.plot(tb_data['val_acc']['step'], tb_data['val_acc']['value'],
                color=colors[idx], linewidth=2, alpha=0.9,
                label=exp_name)

ax2.set_ylabel('Validation Accuracy')
ax2.set_xlabel('Training Steps')
ax2.grid(True, alpha=0.5)
ax2.legend(bbox_to_anchor=(0.02, 0.68), loc='upper left',
          title='Models', title_fontsize=12,
          frameon=True, fancybox=True)

# Adjust layout
plt.tight_layout()

# Save with high DPI for better quality
plt.savefig("train_loss_accuracy.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()