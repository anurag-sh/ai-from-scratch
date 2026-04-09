
import numpy as np
import random

# Set seeds for reproducibility
RANDOM_SEED = 1
epoch=10000
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

from model import Perceptron
from utils import generate_and_split_data
from visualization.plot_boundary import plot_dataset, plot_boundary, plot_training_progress

train_x, train_y, test_x, test_y = generate_and_split_data()
plot_dataset("Training Dataset", train_x, train_y)

perceptron = Perceptron(train_x.shape[1], eta=.001, epoch=epoch)
snapshots = perceptron.train(train_x, train_y)
stochastic_snapshots = perceptron.stochastic_train(train_x, train_y)

# Visualize how boundary changed during training
plot_training_progress(train_x, train_y, snapshots, save_as='training_progress.png')
plot_training_progress(train_x, train_y, stochastic_snapshots, save_as='stochastic_training_progress.png')  

# Final boundary
plot_boundary("Final Decision Boundary", train_x, train_y, 
              perceptron.weights, perceptron.bias, save_as='final_boundary.png')


eta=[0.01, 0.1, 0.5,1.0]
output = []
for e in eta:
    model=Perceptron(train_x.shape[1], eta=e, epoch=epoch)
    model.stochastic_train(train_x, train_y)
    output.append([model.stochastic_weights, model.stochastic_bias, epoch, 0, e])
plot_training_progress(train_x, train_y, output, save_as='eta_diff_stochastic_boundary.png')

print("\nLearned weights:", perceptron.weights)
print("Learned bias:", perceptron.bias) 

plot_boundary("Trained Data", train_x, train_y, perceptron.weights, perceptron.bias, save_as='boundary_plot.png')


