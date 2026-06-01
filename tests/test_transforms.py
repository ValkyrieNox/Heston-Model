import numpy as np
import torch

from finflow.transforms import (
    inverse_lambert_w_transform,
    inverse_lambert_w_transform_torch,
    lambert_w_transform,
    lambert_w_transform_torch,
)


def test_lambert_w_torch_matches_numpy_round_trip() -> None:
    x_np = np.linspace(-2.0, 2.0, 17, dtype=np.float32)
    x = torch.tensor(x_np, requires_grad=True)
    y = lambert_w_transform_torch(x, delta=0.1)
    restored = inverse_lambert_w_transform_torch(y, delta=0.1)
    np.testing.assert_allclose(
        y.detach().numpy(),
        lambert_w_transform(x_np, delta=0.1),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        restored.detach().numpy(),
        inverse_lambert_w_transform(lambert_w_transform(x_np, delta=0.1), delta=0.1),
        rtol=1e-5,
        atol=1e-5,
    )
    restored.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
