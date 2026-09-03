import pytest
import torch

from lejepa_sae.models import relu_forward_leaky_backward
from lejepa_sae.train import activation_transition_metrics, aggregate_metrics


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("leaky_backward", [False, True])
def test_gate_transitions_match_preactivations_and_exact_gap(dtype, leaky_backward):
    global_a = torch.tensor([[0, 2, -1, 3], [-2, 0, 4, -3]], dtype=dtype)
    local_a = torch.tensor(
        [
            [[1, 2, 0, -1], [-2, 1, 4, 2]],
            [[0, -1, 1, 3], [-2, 0, -1, -3]],
            [[0, 2, -1, 3], [-2, 0, 4, -3]],
            [[1e-7, 0, -1, 3], [1, 0, 0, -3]],
        ], dtype=dtype,
    )

    def activate(a):
        a.requires_grad_()
        return relu_forward_leaky_backward(a, 0.1) if leaky_backward else a.relu()

    metrics = activation_transition_metrics(activate(global_a), activate(local_a))
    assert metrics["off_to_on"].item() == 6 / 32
    assert metrics["on_to_off"].item() == 5 / 32
    assert metrics["global_active_fraction"].item() == 3 / 8
    assert metrics["local_active_fraction"].item() == 13 / 32
    assert metrics["local_global_active_fraction_gap"].item() == 1 / 32
    assert metrics["transition_rate_gap"].item() == 1 / 32
    for value in metrics.values():
        assert value.dtype == torch.float32
        assert not value.requires_grad
    torch.testing.assert_close(
        metrics["off_to_on"], ((global_a <= 0) & (local_a > 0)).float().mean()
    )
    torch.testing.assert_close(
        metrics["on_to_off"], ((global_a > 0) & (local_a <= 0)).float().mean()
    )


def test_transition_identity_survives_optional_metric_aggregation():
    torch.manual_seed(21)
    records = [{"loss": torch.tensor(1.0)}]
    for batch_size in (3, 7):
        records.append(activation_transition_metrics(
            torch.randn(batch_size, 16).relu(), torch.randn(4, batch_size, 16).relu()
        ))
    combined = aggregate_metrics(records)
    assert combined["off_to_on"] - combined["on_to_off"] == pytest.approx(
        combined["local_active_fraction"] - combined["global_active_fraction"], abs=1e-7
    )
    assert combined["transition_rate_gap"] == pytest.approx(
        combined["local_global_active_fraction_gap"], abs=1e-7
    )


def test_identical_views_have_no_transitions():
    global_features = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    metrics = activation_transition_metrics(global_features, global_features.expand(4, -1, -1))
    assert metrics["off_to_on"] == metrics["on_to_off"] == 0
    assert metrics["transition_rate_gap"] == metrics["local_global_active_fraction_gap"] == 0
