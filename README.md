# stdw_pytorch
Implementation of soft dynamic time warping in pytorch

Here is my implementation of the Soft Dynamic Time Warping loss function described in https://arxiv.org/abs/1703.01541.
    
Currently I have only a 'naive' implementation without extending the fast cython implementation in 
https://github.com/mblondel/soft-dtw to incorporate a batch dimension. If I continue to use this in my line of
research I may implement a cython / CUDA version to increase speed.
