import torch

from finflow.models import (
    SinusoidalTimeEmbedding,
    TransitionFM,
    conditional_flow_matching_loss,
    euler_sample,
    sample_conditional_flow_batch,
)


def test_sinusoidal_time_embedding_shape() -> None:
    embedding = SinusoidalTimeEmbedding(embedding_dim=7)
    tau = torch.tensor([0.0, 0.5, 1.0])

    out = embedding(tau)

    assert out.shape == (3, 7)
    assert torch.isfinite(out).all()


def test_transition_fm_forward_shape_and_gradients() -> None:
    torch.manual_seed(0)
    model = TransitionFM(hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    condition = torch.randn(5, 2)
    target = torch.randn(5, 2)
    x_tau, tau, velocity, _ = sample_conditional_flow_batch(target)

    prediction = model(x_tau=x_tau, tau=tau, condition=condition)
    loss = (prediction - velocity).pow(2).mean()
    loss.backward()

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    assert any(param.grad is not None for param in model.parameters())


def test_conditional_flow_matching_loss_returns_scalar() -> None:
    torch.manual_seed(1)
    model = TransitionFM(hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    condition = torch.randn(8, 2)
    target = torch.randn(8, 2)

    loss = conditional_flow_matching_loss(model, condition=condition, target=target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_conditional_flow_matching_loss_accepts_target_weights() -> None:
    torch.manual_seed(1)
    model = TransitionFM(hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    condition = torch.randn(8, 2)
    target = torch.randn(8, 2)

    loss = conditional_flow_matching_loss(
        model, condition=condition, target=target,
        target_weights=torch.tensor([1.0, 2.0]),
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_euler_sample_shape() -> None:
    torch.manual_seed(2)
    model = TransitionFM(hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    condition = torch.randn(6, 2)

    sample = euler_sample(model, condition=condition, n_steps=3)

    assert sample.shape == (6, 2)
    assert torch.isfinite(sample).all()

