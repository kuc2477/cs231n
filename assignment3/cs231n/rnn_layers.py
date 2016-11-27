import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  xh = x.dot(Wx)
  hh = prev_h.dot(Wh)
  score = xh + hh + b
  next_h = np.tanh(score)

  cache = (x, prev_h, next_h, Wx, Wh, b)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None

  x, prev_h, next_h, Wx, Wh, b = cache
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  # N * H
  dscore = (1 - next_h ** 2) * dnext_h
  dscore_dxh = 1
  dscore_dhh = 1
  dscore_db = 1

  # D * H
  dxh_dx = Wx

  # N * D
  dxh_dWx = x

  # H * H
  dhh_dprev_h = Wh

  # N * H
  dhh_dWh = prev_h

  # (N * H) * (D * H)^T
  dx = (dscore * dscore_dxh).dot(dxh_dx.T)

  # (N * H) * (H * H)
  dprev_h = (dscore * dscore_dhh).dot(dhh_dprev_h.T)

  # (N * D)^T * (N * H)
  dWx = dxh_dWx.T.dot(dscore * dscore_dxh)

  # (N * H)^T * (N * H)
  dWh = dhh_dWh.T.dot(dscore * dscore_dhh)

  # H
  db = np.sum(dscore * dscore_db, axis=0)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  N, T, D = x.shape
  _, H = h0.shape
  D, _ = Wx.shape

  h, cache = None, None
  h = np.zeros((N, T, H))
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################

  # Run RNN steps through T time with N mini-batch 
  h_t = h0
  for t in xrange(T):
      h_t, cache = rnn_step_forward(x[:, t, :], h_t, Wx, Wh, b)
      h[:, t, :] = h_t
  cache = x, h0, Wx, Wh, b, h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None

  x, h0, Wx, Wh, b, h = cache

  N, T, H = dh.shape
  N, T, D = x.shape

  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros(H)

  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  dh_t_prev = np.zeros((N, H))
  for t in reversed(xrange(T)):
      # Prepare dh, x, h, h_prev and cache of step t. Note that h's actual 
      # gradient dh is sum of given upstream gradient and gradient from 
      # next step.
      dh_t = dh[:, t, :] + dh_t_prev
      x_t = x[:, t, :]
      h_t = h[:, t, :]
      h_t_prev = h[:, t - 1, :] if t - 1 >= 0 else h0
      c_t = x_t, h_t_prev, h_t, Wx, Wh, b

      # Run backpropagation for a single step
      dx_t, dh_t_prev, dWx_t, dWh_t, db_t = rnn_step_backward(dh_t, c_t)

      # Set gradient of x_t
      dx[:, t, :] = dx_t

      # Accumulate gradients on Wx, Wh and b along with t
      dWx += dWx_t
      dWh += dWh_t
      db += db_t

  # Gradient of h0
  dh0 = dh_t_prev
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  out = W[x, :]
  cache = x, W
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  x, W = cache
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  dW = np.zeros_like(W)
  np.add.at(dW, x, dout)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None

  N, D = x.shape
  _, H = prev_h.shape

  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  xh = x.dot(Wx)
  hh = prev_h.dot(Wh)
  score = xh + hh + b
  score_i, score_f, score_o, score_g = \
      score[:, :H], score[:, H:2*H], \
      score[:, 2*H:3*H], score[:, 3*H:4*H]

  i = sigmoid(score_i)
  f = sigmoid(score_f)
  o = sigmoid(score_o)
  g = np.tanh(score_g)

  next_c = f * prev_c + i * g
  next_h = o * np.tanh(next_c)
  cache = (
      x, prev_c, prev_h, Wx, Wh, b, xh, hh, 
      score, score_i, score_f, score_o, score_g,
      i, f, o, g,
      next_c, next_h,
  )
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None

  # unpack the cache
  (x, prev_c, prev_h, Wx, Wh, b, xh, hh, 
   score, score_i, score_f, score_o, score_g, 
   i, f, o, g, next_c, next_h) = cache

  # set spatial parameters
  N, D = x.shape
  _, H = prev_c.shape

  # set gradients into appropriate dimensions of zeros to backprop in
  # accumulative way.
  dx = np.zeros_like(x)
  dprev_h = np.zeros_like(prev_h)
  dprev_c = np.zeros_like(prev_c)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)

  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################

  # dnext_h
  dnext_h_do = np.tanh(next_c)
  dnext_h_dnext_c = o * (1 - np.tanh(next_c) ** 2)

  # dnext_c
  dnext_c_df = prev_c
  dnext_c_dprev_c = f
  dnext_c_di = g
  dnext_c_dg = i

  # dscore
  dscore_dx = dxh_dx
  dscore_dWx = x.T
  dscore_dWh = prev_h.T
  dscore_dprev_h = Wh
  dscore_db = 1

  # di, df, do, dg
  di_dscore_i = i * (1 - i)
  df_dscore_f = f * (1 - f)
  do_dscore_o = o * (1 - o)
  dg_dscore_g = 1 - g ** 2

  # di, df, do, dg
  di_dscore = np.zeros_like(score)
  df_dscore = np.zeros_like(score)
  do_dscore = np.zeros_like(score)
  dg_dscore = np.zeros_like(score)
  di_dscore[:, :H] = di_dscore_i
  df_dscore[:, H:2*H] = df_dscore_f
  do_dscore[:, 2*H:3*H] = do_dscore_o
  dg_dscore[:, 3*H:4*H] = dg_dscore_g

  # di
  di_dx = di_dscore * dscore_dx
  di_dWx = di_dscore * dscore_dWx
  di_dWh = di_dscore * dscore_dWh
  di_dprev_h = di_dscore * dscore_dprev_h
  di_db = di_dscore * db

  # df
  df_dx = df_dscore * dscore_dx
  df_dWx = df_dscore * dscore_dWx
  df_dWh = df_dscore * dscore_dWh
  df_dprev_h = df_dscore * dscore_dprev_h
  df_db = df_dscore * db

  # do
  do_dx = do_dscore * dscore_dx
  do_dWx = do_dscore * dscore_dWx
  do_dWh = do_dscore * dscore_dWh
  do_dprev_h = do_dscore * dscore_dprev_h
  do_db = do_dscore * db

  # dg
  dg_dx = dg_dscore * dscore_dx
  dg_dWx = dg_dscore * dscore_dWx
  dg_dWh = dg_dscore * dscore_dWh
  dg_dprev_h = dg_dscore * dscore_dprev_h
  dg_db = dg_dscore * db

  # compute gradients over next_c
  dnext_c_dx = dnext_c_df * df_dx + dnext_c_di * di_dx + dnext_c_dg * dg_dx
  dnext_c_dprev_h = (
      dnext_c_df * df_dprev_h + 
      dnext_c_di * di_dprev_h + 
      dnext_c_dg * dg_dprev_h
  )
  dnext_c_dWx = (
      dnext_c_df * df_dWx + 
      dnext_c_di * di_dWx + 
      dnext_c_dg * dg_dWx
  )
  dnext_c_dWh = (
      dnext_c_df * df_dWh +
      dnext_c_di * di_dWh +
      dnext_c_dg * dg_dWh
  )
  dnext_c_db = (
      dnext_c_df * df_db +
      dnext_c_di * di_db +
      dnext_c_dg * dg_db
  )

  # compute gradients over next_h
  dnext_h_dx = dnext_h_do * do_dx
  dnext_h_dprev_h = dnext_h_do * do_dprev_h
  dnext_h_dprev_c = dnext_h_dnext_c * dnext_c_dprev_c
  dnext_h_dWx = dnext_h_do * do_dWx
  dnext_h_dWh = dnext_h_do * do_dWh
  dnext_h_db = dnext_h_do * do_db
  dnext_h_dx += dnext_h_dnext_c * dnext_c_dx
  dnext_h_dprev_h += dnext_h_dnext_c * dnext_c_dprev_h
  dnext_h_dWx += dnext_h_dnext_c * dnext_c_dWx
  dnext_h_dWh += dnext_h_dnext_c * dnext_c_dWh
  dnext_h_db = dnext_h_dnext_c * dnext_c_db

  # compute final gradients
  dx = dnext_c * dnext_c_dx + dnext_h * dnext_h_dx
  dprev_c = dnext_c * dnext_c_dprev_c + dnext_h * dnext_h_dprev_c
  dprev_h = dnext_c * dnext_c_dprev_h + dnext_h * dnext_h_dprev_h
  dWx = dnext_c * dnext_c_dWx + dnext_h * dnext_h_dWx
  dWh = dnext_c * dnext_c_dWh + dnext_h * dnext_h_dWh
  db = dnext_c * dnext_c_db + dnext_h * dnext_h_db


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

