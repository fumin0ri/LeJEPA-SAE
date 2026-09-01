import sys

import pytest
import torch

from lejepa_sae.extract import (
    _document_identifier,
    _require_nonempty_dataset,
    _resolve_data_files,
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


def test_local_data_file_globs_are_resolved_before_loading(tmp_path):
    first = tmp_path / "00.jsonl.zst"
    second = tmp_path / "01.jsonl.zst"
    first.touch()
    second.touch()
    assert _resolve_data_files([str(tmp_path / "*.jsonl.zst")]) == [
        str(first.resolve()),
        str(second.resolve()),
    ]


def test_missing_local_data_file_glob_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No data files matched"):
        _resolve_data_files([str(tmp_path / "*.jsonl.zst")])


def test_remote_data_file_pattern_is_left_for_datasets_to_resolve():
    url = "https://example.test/train-*.jsonl.zst"
    assert _resolve_data_files([url]) == [url]


def test_nonempty_dataset_peek_preserves_the_first_record():
    assert list(_require_nonempty_dataset([{"text": "first"}, {"text": "second"}])) == [
        {"text": "first"},
        {"text": "second"},
    ]


def test_empty_dataset_fails_before_model_loading():
    with pytest.raises(RuntimeError, match="source dataset is empty"):
        _require_nonempty_dataset([])
