import time
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X

class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")


    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold, hidden_size):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.norm = nn.LayerNorm(hidden_size)

        self.act = nn.SiLU()
    def forward(self, x):
        b, c, d = x.shape[0], x.shape[1], x.shape[2]
        x = x.transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous()
        
        x = self.act(self.norm(x))

        return x

if __name__ == "__main__":
    hyper = HyperComputeModule(257, 257, 9, 384).cuda()
    import time
    start = time.time()
    x = torch.randn(4, 257, 384).cuda()
    end = time.time()
    print(end-start)
    print(hyper(x).shape)














