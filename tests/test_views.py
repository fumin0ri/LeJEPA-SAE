import torch

from lejepa_sae.views import sample_dimension_masks


def test_dimension_masks_keep_exactly_half_and_are_independent():
    values = torch.zeros(32, 8)
    masks = sample_dimension_masks(
        values,
        num_views=4,
        keep_fraction=0.5,
        generator=torch.Generator().manual_seed(9),
    )
    assert all(torch.equal(mask.sum(dim=1), torch.full((32,), 4)) for mask in masks)
    assert torch.unique(masks[0], dim=0).shape[0] > 1
    assert not torch.equal(masks[0], masks[1])


def test_inverted_dimension_mask_is_unbiased_in_expectation():
    values = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]])
    masks = sample_dimension_masks(
        values,
        num_views=4000,
        keep_fraction=0.5,
        generator=torch.Generator().manual_seed(17),
    )
    average = torch.stack([mask * values / 0.5 for mask in masks]).mean(dim=0)
    torch.testing.assert_close(average, values, atol=0.0, rtol=0.05)
