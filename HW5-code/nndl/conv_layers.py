import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape    
  F, C, HH, WW = w.shape  

  x_afterpad = (np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))  

  height = np.int(((H + 2 * pad - HH) / stride) + 1)  

  width = np.int(((W + 2 * pad - WW) / stride) + 1)  

  out = np.zeros([N, F, height, width]) 

  for img in range(N):  
    for kernel in range(F):  
      for row in range(height): 
        for col in range(width): 
          out[img, kernel, row, col] = np.sum(w[kernel, ...] * x_afterpad[img, :, row*stride:row*stride+HH, col*stride:col*stride+WW]) + b[kernel]


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape

  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #

  N_t, C_t, H, W = x.shape  

  temp_x = np.zeros_like(xpad) 

  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  
  for kernel in range(F):

    db[kernel] += np.sum(dout[:, kernel, :, :]) 

  for img in range(N):  
    for kernel in range(F):  
      for row in range(out_height):  
        for col in range(out_width):  
          dw[kernel, ...] += dout[img, kernel, row, col] * xpad[img, :, row*stride:row*stride+f_height, col*stride:col*stride+f_width]
  
  
  for img in range(N):  
    for kernel in range(F):  
      for row in range(out_height):  
        for col in range(out_width):  
          temp_x[img, :, row*stride:row*stride+f_height, col*stride:col*stride+f_width] += dout[img, kernel, row,col] * w[kernel, ...]
  
  dx = temp_x[:, :, pad:H+pad, pad:W+pad]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  height_p = pool_param.get('pool_height')
  width_p = pool_param.get('pool_width')
  stride = pool_param.get('stride')

  N, C, H, W = x.shape  

  out_height = np.int(((H - height_p) / stride) + 1)  

  out_width = np.int(((W - width_p) / stride) + 1)    
  out = np.zeros([N, C, out_height, out_width])

  for img in range(N): 
    for channel in range(C): 

      for row in range(out_height): 
        for col in range(out_width): 
          out[img, channel, row, col] = np.max(x[img, channel, row*stride:row*stride+height_p, col*stride:col*stride+width_p])


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  
  N, C, H, W = x.shape 
  N_t, C_t, height_temp, width_temp = dout.shape

  dx = np.zeros_like(x)

  for img in range(N): 
    for channel in range(C): 

      for row in range(height_temp): 
        for col in range(width_temp): 

          id_idx = np.argmax(x[img, channel, row*stride:row*stride+pool_height, col*stride:col*stride+pool_width])
          dx[img, channel, row*stride:row*stride+pool_height, col*stride:col*stride+pool_width][np.unravel_index(id_idx, [pool_height, pool_width])] = dout[img, channel, row, col]



  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape  

  x_t = x.transpose(0, 2, 3, 1)
  x_r = np.reshape(x_t, (N*H*W, C)) 
  out_dim2, cache = batchnorm_forward(x_r, gamma, beta, bn_param) 

  out = out_dim2.reshape((N, H, W, C)).transpose(0, 3, 1, 2) 

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
	 
  dx = np.zeros_like(dout)

  N, C, H, W = dout.shape
  dout_t = dout.transpose(( 0, 2, 3, 1))
  dout_r = np.reshape(dout_t, (N*H* W, C))

  dx_dim2, dgamma, dbeta = batchnorm_backward(dout_r, cache)

  dx = dx_dim2.reshape((N, H, W, C)).transpose(0, 3, 1, 2) 


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta