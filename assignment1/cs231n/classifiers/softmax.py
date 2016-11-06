import numpy as np
from random import shuffle

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

  N, D = X.shape
  _, K = W.shape


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(N):
      # compute and stabilize f by substracting log-max value of the f
      f = X[i, :].dot(W)
      f -= np.max(f)
      f_exp = np.exp(f)

      # compute loss
      l = -f[y[i]] + np.log(np.sum(f_exp))
      loss += l

      # compute gradient
      for k in range(K):
          p = np.exp(f[k]) / np.sum(f_exp)
          dW[:, k] += (p - (k == y[i])) * X[i, :]

  # compute average 
  loss /= N
  dW /= N

  # regulaization
  loss += 0.5 * reg * np.sum(W ** 2)
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N, D = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # compute and stabilize f by substracting log-max value of the f
  F = X.dot(W)
  F = F - np.max(F)
  F_exp = np.exp(F)

  # compute loss
  L = -np.log(F_exp[np.arange(N), :] / np.sum(F_exp, axis=1).reshape(-1, 1))
  loss += np.sum(L[np.arange(N), y])

  # compute gradient
  ys = np.zeros(L.shape)
  ys[np.arange(ys.shape[0]), y] = 1
  P = F_exp / np.sum(F_exp, axis=1).reshape(-1, 1)
  dW = X.T.dot(P - ys)

  # compute average
  loss /= N
  dW /= N

  # regulaization
  loss += 0.5 * reg * np.sum(W ** 2)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

