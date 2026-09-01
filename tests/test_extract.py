import sys

import torch

from lejepa_sae.extract import (
    _document_identifier,
    _truncate_to_source_token_budget,
    parse_args,
)


def test_content_identity_is_stable_when_source_order_changes():
    first = _document_identifier({}, "same text", 1, None)
    second = _document_identifier({}, "same text", 999, None)
    assert first == second


def test_explicit_source_id_takes_precedence():
    assert _document_identifier({"Index": 123}, "text", 0, "Index") == "123"


def test_source_token_budget_truncates_only_the_final_source_unit():
    token_ids = torch.arange(10)
    assert torch.equal(
        _truncate_to_source_token_budget(token_ids, 6, 10), token_ids[:4]
    )
    assert _truncate_to_source_token_budget(token_ids, 10, 10).numel() == 0
    assert torch.equal(
        _truncate_to_source_token_budget(token_ids, 0, None), token_ids
    )


def test_extractor_defaults_to_1024_token_contexts(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["lejepa-extract", "--dataset", "json", "--output-dir", "output"],
    )
    assert parse_args().context_length == 1024
