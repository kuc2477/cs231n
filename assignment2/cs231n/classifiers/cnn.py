import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               conv_param=None, pool_param=None,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.conv_param = conv_param or {'stride': 1, 'pad': (filter_size - 1) / 2}
    self.pool_param = pool_param or {'pool_height': 2, 'pool_width': 2, 
                                     'stride': 2}

    # set spatial parameters of conv layer
    C, H, W = input_dim
    F = num_filters
    HH = WW = filter_size
    CP = self.conv_param['pad']
    CS = self.conv_param['stride']
    
    # set spatial parameters of pool layer
    PH = self.pool_param['pool_height']
    PW = self.pool_param['pool_width']
    PS = self.pool_param['stride']

    # set spatial parameters of hidden fc layer
    FH = ((H + 2 * CP - HH) / CS + 1 - PH) / PS + 1
    FW = ((W + 2 * CP - WW) / CS + 1 - PW) / PS + 1

    # set derived spatial parameters on cache
    self.spatial_param = (C, H, W, F, HH, CP, CS, PH, PW, PS, FH, FW)


    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(F, C, HH, WW))
    self.params['b1'] = np.zeros(F)
    self.params['W2'] = np.random.normal(
        scale=weight_scale,
        size=(F * FH * FW, hidden_dim)
    )
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(
        scale=weight_scale,
        size=(hidden_dim, num_classes)
    )
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # get derived spatial params
    C, H, W, F, HH, CP, CS, PH, PW, PS, FH, FW = self.spatial_param

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    crp_a, crp_c = conv_relu_pool_forward(
        X, W1, b1, 
        self.conv_param, 
        self.pool_param
    )
    ar_a, ar_c = affine_relu_forward(crp_a, W2, b2)
    a_a, a_c = affine_forward(ar_a, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return a_a
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # calculate data loss and reg loss
    data_loss, da_a = softmax_loss(a_a, y)
    reg_loss = (self.reg / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
    loss = data_loss + reg_loss

    dar_a, dW3, db3 = affine_backward(da_a, a_c)
    dcrp_a, dW2, db2 = affine_relu_backward(dar_a, ar_c)
    dx, dW1, db1 = conv_relu_pool_backward(dcrp_a, crp_c)

    grads['W1'] = dW1 + self.reg * np.sum(W1)
    grads['W2'] = dW2 + self.reg * np.sum(W2)
    grads['W3'] = dW3 + self.reg * np.sum(W3)
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
