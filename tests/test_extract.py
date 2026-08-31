from lejepa_sae.extract import _document_identifier


def test_content_identity_is_stable_when_source_order_changes():
    first = _document_identifier({}, "same text", 1, None)
    second = _document_identifier({}, "same text", 999, None)
    assert first == second


def test_explicit_source_id_takes_precedence():
    assert _document_identifier({"Index": 123}, "text", 0, "Index") == "123"
