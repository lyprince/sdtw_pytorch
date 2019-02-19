'''
Author : Luke Y. Prince
email  : luke.prince@utoronto.ca
github : lyprince
date   : 19 Feb 2019
'''

class SoftDTWLoss(torch.nn.Module):
    '''
    Soft-DTW (Dynamic Time Warping) Loss function as defined in Cuturi and Blondel (2017) Soft-DTW:
    a Differentiable Loss Function for Time-Series. In: Proc. of ICML 2017. 
    https://arxiv.org/abs/1703.01541.
    '''
    
    def __init__(self, gamma=1.0, spatial_independent=False):
        '''
        __init__(self, gamma=1.0, spatial_independent=False):
        
        Arguments:
            gamma (float) : smoothing parameter (default=1.0)
            spatial_independent (bool) : argument to treat spatial dimensions as independent (default=False)
                                         When false, each time point x_t is treated as a vector in multi-dimensional
                                         space. When true, each time point x_t is treated as a set of independent scalars
                                         x_i,t. This is a short-cut for creating a 'false' singular spatial dimension such
                                         that data can continue to be treated as a 3-tensor of size (batch x space x time).
                                         TODO: implement for arbitrary spatial dimensions.
        '''
        
        super(SoftDTWLoss, self).__init__()
        
        self.gamma = gamma
        self.spatial_independent = spatial_independent
        
    def forward(self, x, y):
        '''
        forward(self, x, y):
        
        Arguments:
            x (torch.Tensor): Time series data of size (batch_dim x space_dim x x_time_dim)
            y (torch.Tensor): Time series data of size (batch_dim x space_dim x y_time_dim)
            
        Returns:
            loss (torch.Tensor): Loss for each data point in batch. Size = batch_dim
        '''
        
        return SoftDTWLossFunction.apply(x, y, self.gamma, self.spatial_independent)

class SoftDTWLossFunction(torch.autograd.Function):
    '''
    Custom autograd function for Soft DTW.
    
    See https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-defining-new-autograd-functions
    for details on defining new autograd functions
    '''    
    
    @staticmethod
    def forward(ctx, x, y, gamma=1.0, spatial_independent=False):
        
        '''
        @staticmethod
        forward(ctx, x, y, gamma=1.0, spatial_independent=False):
        
        Compute the forward pass for Soft-DTW, storing intermediate alignment costs R, and Squared Euclidean
        Distance costs D for use in the backward pass. See algorithm 1 in https://arxiv.org/abs/1703.01541
        
        Arguments:
            ctx : context
            x (torch.Tensor) : tensor of dimensions (batch_dim x space_dim x x_time_dim)
            y (torch.Tensor) : tensor of dimensions (batch_dim x space_dim x y_time_dim)
            gamma (float) : smoothing parameter (TODO: find a way to remove this as an argument)
            spatial_independent (bool): treat spatial dimension as independent (TODO: find a way to remove)
            
        Returns:
            SoftDTW_Loss (torch.Tensor)
        '''
        
        # Store parameters in context variable
        ctx.gamma = gamma
        ctx.spatial_independent = spatial_independent
        
        # Determine device and store in context variable
        ctx.device = 'cuda' if x.is_cuda else 'cpu'
        
        # Determine dimensions
        x_batch_dim, x_space_dim, x_time_dim = x.shape
        y_batch_dim, y_space_dim, y_time_dim = y.shape
        
        # Store dimensions in context variable
        ctx.x_time_dim = x_time_dim
        ctx.y_time_dim = y_time_dim
        
        # Check batch dimensions are equal
        if x_batch_dim == y_batch_dim:
            batch_dim = x_batch_dim
            ctx.batch_dim = batch_dim
            del x_batch_dim, y_batch_dim
        else:
            raise RuntimeError('Unequal batch dimensions')
            
        # Check space dimensions are equal
        if x_space_dim == y_space_dim:
            space_dim = x_space_dim
            ctx.space_dim = space_dim
            del x_space_dim, y_space_dim
        else:
            raise RuntimeError('Unequal space dimensions')
        
        # Determine dimensions for Squared Euclidean Distance Gram Matrix
        # +1 because padding needed at the end for backward function
        D_dims = (batch_dim, x_time_dim + 1, y_time_dim + 1, space_dim) \
        if spatial_independent else (batch_dim, x_time_dim + 1, y_time_dim + 1)
        
        # Determine dimensions for Soft-DTW Distance Gram Matrix. 
        # +2 because padding needed either side for forward and backward function
        R_dims = (batch_dim, x_time_dim + 2, y_time_dim + 2, space_dim) \
        if spatial_independent else (batch_dim, x_time_dim + 2, y_time_dim + 2)
        
        # Create Gram Matrices
        D = torch.zeros(D_dims).to(ctx.device)
        R = torch.zeros(R_dims).to(ctx.device)
        
        from math import inf
        
        # Initialize edges of Soft-DTW Gram Matrix
        R[:, 0, 1:] = inf
        R[:, 1:, 0] = inf
        
        niters = x_time_dim + y_time_dim + 2
        
        # Sweep diagonally through Gram Matrices to compute alignment costs. 
        # See https://towardsdatascience.com/gpu-optimized-dynamic-programming-8d5ba3d7064f for inspiration
        for (i,j),(ip1,jp1) in zip(MatrixDiagonalIndexIterator(m = x_time_dim, n = y_time_dim),
                                   MatrixDiagonalIndexIterator(m = x_time_dim + 1, n= y_time_dim + 1, k_start=1)):
            
            # Compute Squared Euclidean Distance
            if spatial_independent:
                D[:, i, j] = (x[:, :, i] - y[:, :, j]).permute(0, 2, 1).pow(2)
            else:
                D[:, i, j] = (x[:, :, i] - y[:, :, j]).permute(0, 2, 1).pow(2).sum(dim=-1)
            
            # Add soft minimum alignment costs
            R[:, ip1, jp1] = D[:, i, j] + softmin([R[:, i, j],
                                                   R[:, ip1, j],
                                                   R[:, i, jp1]],
                                                   gamma=1.0)
        ctx.save_for_backward(x, y)
        ctx.R = R
        ctx.D = D
        return R[:, -2, -2].sum(dim=-1) if spatial_independent else R[:, -2, -2]
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        @staticmethod
        backward(ctx, grad_output):
        
        Compute SoftDTW gradient wrt x. See algorithm 2 in https://arxiv.org/abs/1703.01541
        '''
        # Get saved tensors
        x, y = ctx.saved_tensors
        
        # Determine size of alignment gradient matrix
        E_dims = (ctx.batch_dim, ctx.x_time_dim + 2, ctx.y_time_dim + 2,  ctx.space_dim) \
        if ctx.spatial_independent else (ctx.batch_dim, ctx.x_time_dim + 2, ctx.y_time_dim + 2)
        
        # Create alignment gradient matrix
        E = torch.zeros(E_dims).to(ctx.device)
        E[:, -1, -1] = 1
        
        from math import inf
        ctx.R[:, :-1,  -1] = -inf
        ctx.R[:,  -1, :-1] = -inf
        ctx.R[:,  -1,  -1] = ctx.R[:, -2, -2]
        
        rev_idxs   = reversed(list(MatrixDiagonalIndexIterator(ctx.x_time_dim,     ctx.y_time_dim)))
        rev_idxsp1 = reversed(list(MatrixDiagonalIndexIterator(ctx.x_time_dim + 1, ctx.y_time_dim + 1, k_start = 1)))
        rev_idxsp2 = reversed(list(MatrixDiagonalIndexIterator(ctx.x_time_dim + 2, ctx.y_time_dim + 2, k_start = 2)))
        
        # Sweep diagonally through alignment gradient matrix
        for (i,j),(ip1,jp1),(ip2,jp2) in zip(rev_idxs, rev_idxsp1, rev_idxsp2):
            a = torch.exp((ctx.R[:, ip2, jp1] - ctx.R[:, ip1, jp1] - ctx.D[:, ip1, j  ])/ctx.gamma)
            b = torch.exp((ctx.R[:, ip1, jp2] - ctx.R[:, ip1, jp1] - ctx.D[:, i,   jp1])/ctx.gamma)
            c = torch.exp((ctx.R[:, ip2, jp2] - ctx.R[:, ip1, jp1] - ctx.D[:, ip1, jp1])/ctx.gamma)
            
            E[:, ip1, jp1] = E[:, ip2, jp1]*a + E[:, ip1, jp2]*b +  E[:, ip2, jp2]*c
        
        # Compute Jacobean product to compute gradient wrt x
        if ctx.spatial_independent:
            G = jacobean_product_squared_euclidean(x.unsqueeze(2), y.unsqueeze(2), E[:, 1:-1, 1:-1].permute(0, 3, 2, 1)).squeeze(2)
        else:
            G = jacobean_product_squared_euclidean(x, y, E[:, 1:-1, 1:-1].permute(0, 2, 1))
        
        # Must return as many outputs as inputs to forward function
        return G, None, None, None, None
    
def softmin(x, gamma):
    '''
    softmin(x, gamma):
    
    Soft minimum function used to smooth DTW and make it differentiable
    
    Arguments
        x     : list of tensors [x1, ..., xN] to compute soft-minimum over
        gamma : smoothing parameter
    
    Return
        smin_x : softmin of x
    '''
    # Obtain dimensions of inputs
    dims = tuple([len(x), *x[0].shape])
    
    # Concatenate inputs
    x = -torch.cat(x).reshape(dims)/gamma
    
    # Compute and return soft minimum
    return -gamma * torch.logsumexp(x, dim=0)

def jacobean_product_squared_euclidean(X, Y, Bt):
    '''
    jacobean_product_squared_euclidean(X, Y, Bt):
    
    Jacobean product of squared Euclidean distance matrix and alignment matrix.
    See equations 2 and 2.5 of https://arxiv.org/abs/1703.01541
    '''
    ones = torch.ones(Y.shape).to('cuda' if Bt.is_cuda else 'cpu')
    return 2 * (ones.matmul(Bt) * X - Y.matmul(Bt))

class MatrixDiagonalIndexIterator:
    '''
    Custom iterator class to return successive diagonal indices of a matrix
    '''
    
    def __init__(self, m, n, k_start=0):
        '''
        __init__(self, m, n, k_start=0):
        
        Arguments:
            m (int)       : number of rows in matrix
            n (int)       : number of columns in matrix
            k_start (int) : (k_start, k_start) index to begin from
        '''
        self.m = m
        self.n = n
        self.k = k_start
        self.k_max = self.m + self.n - k_start
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if hasattr(self, 'i') and hasattr(self, 'j'):
            
            if self.k == self.k_max-1:
                raise StopIteration
            
            elif self.k < self.m and self.k < self.n:
                self.i = self.i + [self.k]
                self.j = [self.k] + self.j
                self.k+=1
            
            elif self.k >= self.m and self.k < self.n:
                self.j.pop(-1)
                self.j = [self.k] + self.j
                self.k+=1
            
            elif self.k < self.m and self.k >= self.n:
                self.i.pop(0)
                self.i = self.i + [self.k]
                self.k+=1
            
            elif self.k >= self.m and self.k >= self.n:
                self.i.pop(0)
                self.j.pop(-1)
                self.k+=1

        else:
            self.i = [self.k]
            self.j = [self.k]
            self.k+=1
        
        return self.i.copy(), self.j.copy()