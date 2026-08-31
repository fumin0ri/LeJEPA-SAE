import torch

from lejepa_sae.views import sample_local_views


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
