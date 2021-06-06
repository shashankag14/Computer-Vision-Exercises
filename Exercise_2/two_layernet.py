from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

np.random.seed(10)

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = 0.
        #############################################################################
        # TODO: Perform the forward pass, computing the class probabilities for the #
        # input. Store the result in the scores variable, which should be an array  #
        # of shape (N, C).                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # HIDDEN LAYER 1
        '''
        W1:4x10, X:5x4, b1(1D Array):1x10 
        hidden_layer1_out : 10x5
        '''
        hidden_layer1_out = np.dot(np.transpose(W1), np.transpose(X)) + b1[:, None]

        # RELU LAYER
        relu_out = hidden_layer1_out
        relu_out[relu_out<0]=0 

        # HIDDEN LAYER 2
        ''' 
        W2:10x3, relu_out:10x5, b2(1D Array):1x3 
        hidden_layer2_out : 3x5
        '''
        hidden_layer2_out = np.dot(np.transpose(W2), relu_out) + b2[:, None]
        
        # SOFTMAX LAYER
        '''
        Calculate maximum over all possible target classes for each input 
        '''
        max_target_val = np.max(hidden_layer2_out, axis = 0)
        e_x = np.exp(hidden_layer2_out - max_target_val)
        softmax_out = e_x / np.sum(e_x, axis = 0)
        scores = softmax_out.T
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = 0.
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Implement the loss for softmax output layer
        
        '''
        Converts 5x3 scores array to 1x5 predicted labels array using one hot encoded 
        truth labels
        '''
        pred_label = scores[range(N), y]
        softmax_loss = 1/N * np.sum(-np.log(pred_label))

        W1_l2norm = np.sum(np.square(W1))
        W2_l2norm = np.sum(np.square(W2))
        l2_penalty = reg * (W1_l2norm + W2_l2norm)
        
        loss += softmax_loss + l2_penalty
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        del_matrix = np.zeros((softmax_out.T).shape)
        for i in range(softmax_out.shape[1]):
            for j in range(softmax_out.shape[0]):
                if y[i] == j:
                    del_matrix[i,j]=1
        dl_dz3 = (1/N)*(softmax_out-del_matrix.T)   #3x5
        
        '''
        Derivative of ReLU
        '''
        der_rel = np.zeros(hidden_layer1_out.shape)
        der_rel[hidden_layer1_out > 0] = 1
        der_rel[hidden_layer1_out <= 0] = 0

        grads['W2'] = np.matmul(dl_dz3,relu_out.T).T + (2 * reg * W2)   #10x3
        grads['b2'] = np.sum(dl_dz3, axis = 1)  #3x1
  
        grads['W1'] = np.matmul(np.multiply(np.matmul(W2,dl_dz3),der_rel),X).T + (2 * reg * W1) #10x4
        grads['b1'] = np.sum(np.multiply(np.matmul(W2,dl_dz3),der_rel), axis = 1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False, early_stop_key=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        # For early stopping
        prev_loss = -100
        loss_increase_counter = 0
        early_stop = True
        early_stop_threshold = 100
        for it in range(num_iters):
            X_batch = X
            y_batch = y
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            temp=np.random.randint(num_train,size=batch_size)
            X_batch = X_batch[temp]  # slicing training data as per batches
            y_batch = y_batch[temp]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.params["b1"] -= learning_rate * grads["b1"]
            self.params["b2"] -= learning_rate * grads["b2"]
            self.params["W1"] -= learning_rate * grads["W1"]
            self.params["W2"] -= learning_rate * grads["W2"]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            '''
            Implemented Early Stopping for experimental purpose
            '''
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                if early_stop_key == True :
                    if loss > prev_loss:
                      loss_increase_counter += 1
                    else:
                      loss_increase_counter = 0
                      prev_loss = loss
                    if loss_increase_counter > early_stop_threshold:
                      print("Early Stopping..")
                      break

            # Every epoch, check train and val accuracy and decay learning rate.
            # Converting iterations_per_epoch to 'int' as it might be a decimal val too
            if it % int(iterations_per_epoch) == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
            if it % iterations_per_epoch == 0:    
                # Decay learning rate
                learning_rate *= learning_rate_decay

                
        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        temp_1 = self.loss(X)
        y_pred = np.argmax(temp_1, axis=1) 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
