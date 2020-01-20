from builtins import range
import numpy as np
import math as m
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    S = X.dot(W)
    exp = np.exp(S - np.max(S))
    row_sum = np.sum(exp, axis=1)
    
    # calculate softmax
    softmax = exp / row_sum[:, np.newaxis]
    
    # calculate loss
    log_liklyhood = -np.log(softmax[np.arange(softmax.shape[0]), y])
    loss = np.sum(log_liklyhood) / softmax.shape[0] + 0.5*reg*np.sum(W**2)
    
    # calculate gradient for that loss
    
    # y       (N,  )
    # softmax (N, C)
    # W       (D, C)
    # dW      (D, C)
    
    for i in range(X.shape[0]):
        for j in range(W.shape[1]):
            dW[:, j] += (softmax[i, j] - (y[i] == j)) * X[i]

    dW /= X.shape[0]
    dW += 2*reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_test = y.shape[0]
    
    S_raw = X.dot(W)
    S = S_raw - np.max(S_raw)
    
    exp = np.exp(S)
    row_sum = np.sum(exp, axis=1, keepdims=True)
    
    # calculate softmax (N, C)
    softmax = exp / row_sum
    
    softmax[np.arange(num_test), y] -= 1
    
    # W   (D, C)
    # dW  (D, C)
    # X   (N, D)
    # X.T (D, N)
    # softmax (N, C)
    dW = X.T.dot(softmax)
    dW /= num_test
    dW += 2*reg*W
    
    loss_i = -S[np.arange(num_test), y] + np.log(np.sum(exp, axis=1))
    loss = np.sum(loss_i) / num_test + reg*np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
