import torch
from torch import nn
import math

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping by github.com/ndahlquist
    (https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py)

    Modified to make gaussians learnable

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10,learnable = False):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        _B = torch.randn((num_input_channels, mapping_size)) * scale
        if learnable:
            self._B = nn.Parameter(_B, requires_grad=True)
        else:
            self.register_buffer('_B',_B)

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class GFFT1D(GaussianFourierFeatureTransform):
    """
    An implementation of Gaussian Fourier feature mapping by github.com/ndahlquist
    (https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py)

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Modified to map 1D vectors
    Given an input of size [batches, input_size]
     returns a tensor of size [batches, mapping_size,input_size]
    """

    def __init__(self, mapping_size=256, scale=10,learnable = False):
        assert mapping_size % 2 == 0
        super().__init__(1,mapping_size//2,scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,S] -> [B,C=1,W=S,H=1]
        x = x.unsqueeze(1).unsqueeze(-1)
        x = super().forward(x)
        # [B,M,W=S,H=1] -> [B,M,S]
        x = x.squeeze(-1)
        return x
class PositionalEncoding(GFFT1D):
    """
    For each input dimmension compute sin/cos positional encoding along need dimension
    [B,S] -> [B,M,S]
    """
    def __init__(self, mapping_size=256, scale=10,learnable = False):
        assert mapping_size % 2 == 0
        super().__init__(mapping_size,1,learnable)
        _B = (1 / (scale ** torch.arange(mapping_size//2))).unsqueeze(0)
        if learnable:
            _B = nn.Parameter(_B, requires_grad=True)
        self._B = _B


def plt_scatter(point, ax, c='b', alpha=0.4,sample=None):
    with torch.no_grad():
        if len(point.shape) == 1:
            point = point.unsqueeze(0)
        if sample is not None:
            indices = torch.randperm(len(point))[:min(sample,len(point))]
            point = point[indices]
        point = point.clone().detach().cpu().numpy()
        ax.scatter(point[:, 0], point[:, 1], point[:, 2], c=c, alpha=alpha)


def stats(x):
    print(f'MIN:{x.min()},MEAN:{x.mean()},MAX:{x.max()},STD{x.std()}')