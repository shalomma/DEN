from torch.nn import Module
from torch.nn import functional as F


class DBELoss(Module):
    def __init__(self):
        super(DBELoss, self).__init__()

    def g(self, d, a1, a2):
            return a1*d + 0.5*a2*(d**2)
        
    def forward(self, d_hat, d, a1=1.5, a2=-0.1):
        
        g_d_hat = self.g(d_hat, a1, a2)    
        g_d = self.g(d, a1, a2)
        dbe = 0.5 * F.mse_loss(g_d_hat, g_d, reduction='sum')
        
        return dbe
