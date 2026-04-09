import numpy as np
import random
# Set seeds for reproducibility
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)

class Perceptron:
    # Initialize the perceptron with the number of features and learning rate
    def __init__(self, n_features, eta=0.001, epoch=100):
        self.weights = np.zeros((n_features))
        self.stochastic_weights = np.zeros((n_features))
        self.stochastic_bias = 0
        self.bias = 0
        self.eta = eta
        self.epoch = epoch
    
    # Activation function (step function)
    # This function takes the weighted sum of inputs and applies a step function to determine the output class.
    def activation_function(self, x):
        return 1 if x >= 0 else -1
    
    # This function computes the weighted sum of the inputs and applies the activation function to determine the predicted class.
    def predict(self, inputs, use_stochastic=False):
        if use_stochastic:
            z = np.dot(self.stochastic_weights, inputs) + self.stochastic_bias
        else:
            z = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(z)
    
    # This function updates the weights and bias based on the error between the predicted output and the target output.
    def train(self, inputs, targets, report_frequency=10):
        snapshots = []  # Store snapshots of weights and accuracy
        
        for epoch in range(self.epoch):
            for x, y in zip(inputs, targets):
                output = self.predict(x)
                error = y - output
                # Vectorized update
                self.weights += self.eta * error * np.array(x)
                self.bias += self.eta * error
            
            # Periodically, print out the current accuracy on all examples 
            if epoch % report_frequency == 0:  
                pos_examples = np.array([ [t[0], t[1]] for i,t in enumerate(inputs) 
                          if targets[i]>0])
                neg_examples = np.array([ [t[0], t[1]] for i,t in enumerate(inputs) 
                          if targets[i]<0])     
                # Counts
                pos_count = len(pos_examples)
                neg_count = len(neg_examples)      
                pos_out = np.dot(pos_examples, self.weights)+ self.bias
                neg_out = np.dot(neg_examples, self.weights)+ self.bias   
                pos_correct = (pos_out >= 0).sum() / float(pos_count)
                neg_correct = (neg_out < 0).sum() / float(neg_count)
                accuracy = (pos_correct + neg_correct) / 2.0
                
                print("Epoch={}, pos correct={}, neg correct={}, avg accuracy={}".format(
                    epoch, pos_correct, neg_correct, accuracy))
                
                # Save snapshot of current state
                snapshots.append([np.copy(self.weights), self.bias, epoch, accuracy, self.eta])
        
        return snapshots
    
    # This function updates the weights and bias based on the error between the predicted output and the target output.
    def stochastic_train(self, inputs, targets, report_frequency=10):
        snapshots = []  # Store snapshots of weights and accuracy
        pos_examples = np.array([ [t[0], t[1]] for i,t in enumerate(inputs) 
                          if targets[i]>0])
        neg_examples = np.array([ [t[0], t[1]] for i,t in enumerate(inputs) 
                          if targets[i]<0])
        
        pos_count = pos_examples.shape[0]
        neg_count = neg_examples.shape[0]

        for epoch in range(self.epoch):
            # Pick one positive and one negative example
            pos = random.choice(pos_examples)
            neg = random.choice(neg_examples)
            output = self.predict(pos, use_stochastic=True)
            # Positive
            if output < 1:
                self.stochastic_weights += self.eta * pos
                self.stochastic_bias += self.eta
            

            output = self.predict(neg, use_stochastic=True)
            # Negative
            if output >= 0:
                self.stochastic_weights -= self.eta * neg
                self.stochastic_bias -= self.eta
            
            # Periodically, print out the current accuracy on all examples 
            if epoch % report_frequency == 0:       
                     
                pos_out = np.dot(pos_examples, self.stochastic_weights)+ self.stochastic_bias
                neg_out = np.dot(neg_examples, self.stochastic_weights)+ self.stochastic_bias   
                pos_correct = (pos_out >= 0).sum() / float(pos_count)
                neg_correct = (neg_out < 0).sum() / float(neg_count)
                accuracy = (pos_correct + neg_correct) / 2.0
                
                print("Epoch={}, pos correct={}, neg correct={}, avg accuracy={}".format(
                    epoch, pos_correct, neg_correct, accuracy))
                
                # Save snapshot of current state
                snapshots.append([np.copy(self.stochastic_weights), self.stochastic_bias, epoch, accuracy, self.eta])
        
        return snapshots
            
        
        