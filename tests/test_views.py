import torch

from lejepa_sae.views import sample_dimension_masks, sample_local_views


def test_local_views_remove_tokens_and_keep_original_positions():
    residuals = torch.arange(2 * 10 * 3).reshape(2, 10, 3).float()
    views = sample_local_views(
        residuals,
        num_views=4,
        retained_tokens=3,
        generator=torch.Generator().manual_seed(7),
    )

    assert len(views) == 4
    for view in views:
        assert view.residuals.shape == (2, 3, 3)
        assert view.positions.shape == (2, 3)
        assert torch.all(view.positions[:, 1:] > view.positions[:, :-1])
        expected = residuals.gather(1, view.positions.unsqueeze(-1).expand(-1, -1, 3))
        torch.testing.assert_close(view.residuals, expected)


def test_local_subsets_are_independent_per_batch_item():
    residuals = torch.zeros(32, 10, 2)
    view = sample_local_views(
        residuals, 1, 3, generator=torch.Generator().manual_seed(11)
    )[0]
    assert torch.unique(view.positions, dim=0).shape[0] > 1


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
