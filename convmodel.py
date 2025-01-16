import numpy as np
import h5py
import matplotlib.pyplot as plt


#padding horizontally and vertically 
def zero_pad(X, pad):
 
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))

    return X_pad


# applying a filter to a region of the current activation
def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + float(b)

    return Z 


# filtering through the whole activation (one conv layer application)
def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev)
    (f,f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev - f + 2*pad)/stride) + 1
    n_W = int((n_W_prev - f + 2*pad)/stride) + 1
    
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):

        # shifting vertically
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            
            #shifting horizontally 
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f 
                
                # applying each filter 
                for c in range(n_C):
                    a_prev_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    # saving for backprop 
    cache = (A_prev, W, b, hparameters)
    return Z, cache 


# pooling layer application
def pool_forward(A_prev, hparameters, mode="max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C)) 

    for i in range(m):
       
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f 

                for c in range(n_C):

                    # applied to each channel separately 
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    cache = (A_prev, hparameters)

    return A, cache


def conv_backward(dZ, cache):
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape 
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((A_prev.shape))
    dW = np.zeros((W.shape))
    db = np.zeros((b.shape))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f 
                    
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += dZ[i, h, w, c] * a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db


def create_mask_from_window(x):
    mask = np.where(x == np.max(x), 1, 0)
    return mask 

def distribute_value(dz, shape):
    (n_H, n_W) = shape 
    average = n_H * n_W
    a = (np.ones(shape)*dz)/average

    return a 


def pool_backward(dA, cache, mode="max"):
    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = np.zeros((A_prev.shape))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    if mode == "max":
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape) 
    
    return dA_prev