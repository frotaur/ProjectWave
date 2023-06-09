import WaveModel as wm
import torch,torch.nn.functional as F


def test_chamfersingle():
    x1 = torch.randn((5,1,2))
    y1 = torch.randn((5,1,2))

    l2_loss = F.mse_loss(x1,y1,reduction='mean')

    ch_loss = wm.chamfer_dist(x1,y1)
    

    assert (l2_loss-ch_loss)<1e-6

def test_chamfermultiple():
    x = torch.randn(5,10,3)
    y = torch.randn(5,10,3)

    B,N,D = x.shape

    ch_loss = wm.chamfer_dist(x,y)
    
    pairsumsquared = torch.zeros(5,10,10)
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            pairsumsquared[:,i,j]=((x[:,i,:]-y[:,j,:])**2).mean(dim=-1)

    sumxy = torch.min(pairsumsquared,dim=1)[0].mean()
    sumyx = torch.min(pairsumsquared,dim=2)[0].mean()

    ch_loss2 = .5*(sumxy+sumyx)

    assert ch_loss-ch_loss2<1e-6
