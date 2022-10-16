'''
implement 2
https://github.com/danielgreenfeld3/XIC
'''
def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)
    
def HSIC2(x, y, s_x=1, s_y=1):
    x = torch.tensor(x)
    y = torch.tensor(y)
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.double()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC