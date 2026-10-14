"""Complex-valued linear projections into single or multi-head Hilbert spaces."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..functional import ensure_complex


class ComplexLinear(nn.Module):
    """Complex linear projection layer mapping R^D or C^D -> C^N.

    Stores real and imaginary weight matrices as explicit `nn.Parameter` tensors
    for full compatibility with all PyTorch optimizers and gradient clippers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_r = nn.Parameter(torch.randn(in_features, out_features) * init_std)
        self.weight_i = nn.Parameter(torch.randn(in_features, out_features) * init_std)

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

    @property
    def weight(self) -> torch.Tensor:
        """Return complex weight tensor of shape `(in_features, out_features)`."""
        return torch.complex(self.weight_r, self.weight_i)

    @property
    def bias(self) -> torch.Tensor | None:
        """Return complex bias tensor of shape `(out_features,)` if enabled."""
        if self.bias_r is None or self.bias_i is None:
            return None
        return torch.complex(self.bias_r, self.bias_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = ensure_complex(x)
        out = torch.matmul(x_c, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias_r is not None}"
        )


class MultiHeadComplexProjection(nn.Module):
    """Multi-head complex projection mapping `(..., D)` -> `(..., H, N)` or `(B, H, L, N)`.

    Used for projecting classical or complex features into H Hilbert space heads of dimension N.
    """

    def __init__(
        self,
        in_features: int,
        hilbert_dim: int,
        num_heads: int = 4,
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hilbert_dim = hilbert_dim
        self.num_heads = num_heads

        self.W_r = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_i = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)

    @property
    def weight(self) -> torch.Tensor:
        """Return complex projection tensor of shape `(H, in_features, hilbert_dim)`."""
        return torch.complex(self.W_r, self.W_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input tensor into multi-head complex Hilbert space.

        Args:
            x: Tensor of shape `(B, D)` or `(B, L, D)`.

        Returns:
            Complex tensor of shape `(B, H, N)` if input is 2D, or `(B, H, L, N)` if input is 3D.
        """
        x_c = ensure_complex(x)
        if x_c.dim() == 2:
            return torch.einsum("bd,hdn->bhn", x_c, self.weight)
        elif x_c.dim() == 3:
            return torch.einsum("bld,hdn->bhln", x_c, self.weight)
        else:
            raise ValueError(f"Expected 2D (B, D) or 3D (B, L, D) input, got shape {tuple(x.shape)}")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hilbert_dim={self.hilbert_dim}, "
            f"num_heads={self.num_heads}"
        )
