import torch

from finflow.inference import (
    ConsistencySampler,
    FMTeacherSampler,
    MeanFlowSampler,
)
from finflow.models import ConsistencyStudent, MeanFlowStudent, TransitionFM


def _cond(B, dim):
    return torch.randn(B, dim)


def test_fm_teacher_sampler_shape():
    torch.manual_seed(0)
    model = TransitionFM(state_dim=1, condition_dim=3, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    sampler = FMTeacherSampler(model, n_steps=4)
    out = sampler.sample(_cond(6, 3))
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()


def test_fm_teacher_sampler_heun_shape():
    torch.manual_seed(0)
    model = TransitionFM(state_dim=1, condition_dim=3, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    sampler = FMTeacherSampler(model, n_steps=4, solver="heun")
    out = sampler.sample(_cond(6, 3))
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()


def test_mean_flow_sampler_shape():
    torch.manual_seed(0)
    model = MeanFlowStudent(state_dim=1, condition_dim=3, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    sampler = MeanFlowSampler(model)
    out = sampler.sample(_cond(5, 3))
    assert out.shape == (5, 1)
    assert torch.isfinite(out).all()


def test_mean_flow_sampler_subtracts_average_velocity():
    class ConstantMeanFlow(torch.nn.Module):
        state_dim = 1
        condition_dim = 2

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, r, t, condition):
            return torch.ones_like(x) * (0.25 + self.anchor)

    sampler = MeanFlowSampler(ConstantMeanFlow())
    cond = torch.zeros(3, 2)
    noise = torch.full((3, 1), 2.0)
    out = sampler.sample(cond, noise=noise)
    assert torch.allclose(out, torch.full((3, 1), 1.75))


def test_consistency_sampler_shape():
    torch.manual_seed(0)
    model = ConsistencyStudent(state_dim=1, condition_dim=3, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    sampler = ConsistencySampler(model, time_eps=1e-3)
    out = sampler.sample(_cond(7, 3))
    assert out.shape == (7, 1)
    assert torch.isfinite(out).all()


def test_sampler_accepts_provided_noise():
    model = MeanFlowStudent(state_dim=1, condition_dim=2, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    sampler = MeanFlowSampler(model)
    cond = _cond(4, 2)
    z = torch.zeros(4, 1)
    out_a = sampler.sample(cond, noise=z)
    out_b = sampler.sample(cond, noise=z)
    assert torch.allclose(out_a, out_b)


def test_mean_flow_sampler_cfg_zeroes_action_slice_for_uncond_branch():
    class ActionMeanFlow(torch.nn.Module):
        state_dim = 1
        condition_dim = 3

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, r, t, condition):
            action_strength = condition[:, -2:].sum(dim=1, keepdim=True)
            return action_strength + self.anchor

    sampler = MeanFlowSampler(ActionMeanFlow(), num_actions=2)
    cond = torch.tensor([[0.5, 1.0, 0.0]])
    noise = torch.full((1, 1), 2.0)
    unguided = sampler.sample(cond, noise=noise)
    guided = sampler.sample(cond, noise=noise, cfg_w=2.0)
    assert torch.allclose(unguided, torch.tensor([[1.0]]))
    assert torch.allclose(guided, torch.tensor([[-1.0]]))
