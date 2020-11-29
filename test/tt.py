import torch
from torch.distributions.dirichlet import Dirichlet

if __name__ == '__main__':
    x = Dirichlet(torch.tensor([0.5, 0.5]))
    print(x.sample((5,)))
