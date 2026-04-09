import pylab
from matplotlib import gridspec
import matplotlib as mpl
import numpy as np

# Detect if running in Jupyter notebook
def _is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        pass
    return False

# Setup matplotlib backend
if not _is_notebook():
    mpl.use('Agg')


def _show_plot(fig, filename='plot.png'):
    if _is_notebook():
        fig.show()
    else:
        fig.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {filename}")


# =========================
# Dataset Plot
# =========================
def plot_dataset(suptitle, features, labels, save_as='training_data.png'):
    fig, ax = pylab.subplots(1, 1)
    fig.suptitle(suptitle, fontsize=16)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    colors = ['r' if l > 0 else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], c=colors, s=100, alpha=0.6)

    ax.grid(True, alpha=0.3)

    _show_plot(fig, save_as)


# =========================
# Decision Boundary Plot
# =========================
def plot_boundary(suptitle, features, labels, weights, bias, save_as='boundary_plot.png'):
    pylab.figure()
    # Handle zero weights
    if np.isclose(weights[1], 0):
        if np.isclose(weights[0], 0):
            x = y = np.array([-6, 6], dtype = 'float32')
        else:
            y = np.array([-6, 6], dtype='float32')
            x = -(weights[1] * y + bias)/weights[0]
    else:
        x = np.array([-6, 6], dtype='float32')
        y = -(weights[0] * x + bias)/weights[1]

    pylab.xlim(-6, 6)
    pylab.ylim(-6, 6)                      
    pylab.plot(features[labels > 0, 0], features[labels > 0, 1], 'bo')
    pylab.plot(features[labels < 0, 0], features[labels < 0, 1], 'ro')
    pylab.plot(x, y, 'g', linewidth=2.0)
    pylab.title(suptitle)
    pylab.savefig(save_as, dpi=100, bbox_inches='tight')
    pylab.close()


# =========================
# Training Progress Plot
# =========================
def plot_training_progress(features, labels, snapshots, save_as='training_progress.png'):
    import math

    # 🔥 LIMIT number of plots
    max_plots = 8
    indices = np.linspace(0, len(snapshots) - 1, max_plots, dtype=int)
    snapshots = [snapshots[i] for i in indices]

    num_plots = len(snapshots)
    cols = 4
    rows = math.ceil(num_plots / cols)

    fig, axes = pylab.subplots(rows, cols, figsize=(14, 3 * rows))
    fig.suptitle('Training Progress', fontsize=16)

    axes = axes.flatten() if num_plots > 1 else [axes]

    for idx, sn in enumerate(snapshots):
        ax = axes[idx]
        
        # Train with this learning rate
        weights_lr,bias, epoch, accuracy, eta = sn
        
        # Plot decision boundary
        if np.isclose(weights_lr[1], 0):
            if np.isclose(weights_lr[0], 0):
                x = y = np.array([-6, 6], dtype='float32')
            else:
                y = np.array([-6, 6], dtype='float32')
                x = -(weights_lr[1] * y + bias)/weights_lr[0]
        else:
            x = np.array([-6, 6], dtype='float32')
            y = -(weights_lr[0] * x + bias)/weights_lr[1]
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.plot(features[labels > 0, 0], features[labels > 0, 1], 'bo', label='Positive', alpha=0.7)
        ax.plot(features[labels < 0, 0], features[labels < 0, 1], 'ro', label='Negative', alpha=0.7)
        ax.plot(x, y, 'g-', linewidth=2)
        ax.set_title(f'Learning Rate = {eta}\nEpoch = {epoch}\nAccuracy = {accuracy:.2f}', fontsize=10)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    pylab.tight_layout()
    pylab.savefig(save_as, dpi=100, bbox_inches='tight')
    pylab.close()
