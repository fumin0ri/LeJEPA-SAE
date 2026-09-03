import pytest
import torch

from lejepa_sae.diagnostics import thresholded_active_counts


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_threshold_counts_are_strict_detached_and_do_not_modify_features_or_rng(dtype):
    features = torch.tensor(
        [[-1, 0, 1e-6, 1e-4, 1e-3], [1e-2, 5e-2, 1e-1, 1, 2]],
        dtype=dtype, requires_grad=True,
    )
    before = features.detach().clone()
    rng = torch.get_rng_state().clone()
    counts = thresholded_active_counts(features)
    for label, threshold in (
        ("0", 0), ("1e-4", 1e-4), ("1e-3", 1e-3), ("1e-2", 1e-2),
        ("5e-2", 5e-2), ("1e-1", 1e-1),
    ):
        count = counts[f"active_fraction_gt_{label}"]
        expected = features.detach().float().gt(threshold).sum()
        assert count == expected and not count.requires_grad
    if dtype == torch.float32:
        assert list(map(int, counts.values())) == [8, 6, 5, 4, 3, 2]
    torch.testing.assert_close(features, before, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert features.grad is None


def test_zero_activations_have_no_thresholded_activity():
    assert all(count == 0 for count in thresholded_active_counts(torch.zeros(2, 8)).values())
