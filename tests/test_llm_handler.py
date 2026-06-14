import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import json

# Mock LlmHandler pour les tests
@pytest.fixture
def handler():
    from src.llm_handler import LlmHandler
    h = LlmHandler.model_construct(
        llm=MagicMock(),
        id_to_token={},
        token_to_id={},
        fn_names=[],
        fn_descriptions=[],
        vocab_path=None,
    )
    return h


# ─── validate_string ──────────────────────────────────────────────────────────

def make_logits(size: int, val: float = 0.0) -> list[float]:
    return [val] * size

VOCAB_SIZE = 10

def test_validate_string_first_token_forces_quote(handler):
    """Premier token : seul " doit passer."""
    quote_id = 3
    handler.llm.decode = lambda ids: '"' if ids == [quote_id] else "a"
    logits = make_logits(VOCAB_SIZE)
    result = handler.validate_string([], logits)
    assert result[quote_id] == 0.0
    for i, v in enumerate(result):
        if i != quote_id:
            assert v == float('-inf')


def test_validate_string_after_second_token_no_constraint(handler):
    """Après le deuxième token, aucune contrainte supplémentaire."""
    handler.id_to_token = {3: '"'}
    logits = make_logits(VOCAB_SIZE)
    result = handler.validate_string([3, 5], logits)
    assert result == make_logits(VOCAB_SIZE)


# ─── get_valid_string ─────────────────────────────────────────────────────────

def test_get_valid_string_returns_quoted_string(handler):
    """get_valid_string doit retourner une string entre guillemets."""
    # Simule : " h e l l o "
    tokens = ['"', 'h', 'e', 'l', 'l', 'o', '"']
    token_ids = list(range(len(tokens)))
    id_to_token = {i: t for i, t in enumerate(tokens)}
    handler.id_to_token = id_to_token

    call_count = [0]
    def fake_encode(text):
        m = MagicMock()
        m.__getitem__ = lambda self, i: MagicMock(tolist=lambda: [])
        return m
    handler.llm.encode = fake_encode

    def fake_logits(ids):
        idx = len(ids)
        logits = [float('-inf')] * len(tokens)
        if idx < len(tokens):
            logits[idx] = 1.0
        return logits

    handler.llm.get_logits_from_input_ids = fake_logits
    handler.llm.decode = lambda ids: ''.join(tokens[i] for i in ids)

    result = handler.get_valid_string("")
    assert result.startswith('"')
    assert result.endswith('"')


# ─── validate_number (via get_valid_number) ───────────────────────────────────

def test_get_valid_number_integer(handler):
    """get_valid_number doit retourner un entier valide après strip."""
    tokens = ['"', '4', '2', '"']
    token_ids = list(range(len(tokens)))
    handler.id_to_token = {i: t for i, t in enumerate(tokens)}
    handler.llm.encode = lambda t: [MagicMock(tolist=lambda: [])]

    def fake_logits(ids):
        idx = len(ids)
        logits = [float('-inf')] * len(tokens)
        if idx < len(tokens):
            logits[idx] = 1.0
        return logits

    handler.llm.get_logits_from_input_ids = fake_logits
    handler.llm.decode = lambda ids: ''.join(tokens[i] for i in ids)

    result = handler.get_valid_number("")
    assert float(result) == 42.0


def test_get_valid_number_float(handler):
    """get_valid_number doit retourner un float valide."""
    tokens = ['"', '3', '.', '1', '4', '"']
    handler.id_to_token = {i: t for i, t in enumerate(tokens)}
    handler.llm.encode = lambda t: [MagicMock(tolist=lambda: [])]

    def fake_logits(ids):
        idx = len(ids)
        logits = [float('-inf')] * len(tokens)
        if idx < len(tokens):
            logits[idx] = 1.0
        return logits

    handler.llm.get_logits_from_input_ids = fake_logits
    handler.llm.decode = lambda ids: ''.join(tokens[i] for i in ids)

    result = handler.get_valid_number("")
    assert float(result) == pytest.approx(3.14)


# ─── validate_boolean ─────────────────────────────────────────────────────────

def test_validate_boolean_allows_true_prefix(handler):
    """'tr' est un préfixe valide de 'true'."""
    handler.llm.decode = lambda ids: 'tr'
    logits = make_logits(VOCAB_SIZE)
    logits[0] = 1.0
    result = handler.validate_boolean([], logits)
    assert result[0] == 1.0


def test_validate_boolean_blocks_invalid(handler):
    """'xyz' n'est pas un préfixe valide."""
    handler.llm.decode = lambda ids: 'xyz'
    logits = make_logits(VOCAB_SIZE)
    result = handler.validate_boolean([], logits)
    assert all(v == float('-inf') for v in result)


def test_validate_boolean_allows_false_prefix(handler):
    """'fal' est un préfixe valide de 'false'."""
    handler.llm.decode = lambda ids: 'fal'
    logits = make_logits(VOCAB_SIZE)
    logits[0] = 1.0
    result = handler.validate_boolean([], logits)
    assert result[0] == 1.0


def test_get_valid_boolean_true(handler):
    """get_valid_boolean doit retourner 'true'."""
    tokens = ['t', 'r', 'u', 'e']
    handler.id_to_token = {i: t for i, t in enumerate(tokens)}
    handler.llm.encode = lambda t: [MagicMock(tolist=lambda: [])]

    def fake_logits(ids):
        idx = len(ids)
        logits = [float('-inf')] * len(tokens)
        if idx < len(tokens):
            logits[idx] = 1.0
        return logits

    handler.llm.get_logits_from_input_ids = fake_logits
    handler.llm.decode = lambda ids: ''.join(tokens[i] for i in ids)

    result = handler.get_valid_boolean("")
    assert result == "true"


def test_get_valid_boolean_false(handler):
    """get_valid_boolean doit retourner 'false'."""
    tokens = ['f', 'a', 'l', 's', 'e']
    handler.id_to_token = {i: t for i, t in enumerate(tokens)}
    handler.llm.encode = lambda t: [MagicMock(tolist=lambda: [])]

    def fake_logits(ids):
        idx = len(ids)
        logits = [float('-inf')] * len(tokens)
        if idx < len(tokens):
            logits[idx] = 1.0
        return logits

    handler.llm.get_logits_from_input_ids = fake_logits
    handler.llm.decode = lambda ids: ''.join(tokens[i] for i in ids)

    result = handler.get_valid_boolean("")
    assert result == "false"


# ─── validate_array ───────────────────────────────────────────────────────────

def test_validate_array_first_token_forces_bracket(handler):
    """Premier token doit être [."""
    handler.llm.decode = lambda ids: 'a'
    logits = make_logits(VOCAB_SIZE)
    result = handler.validate_array([], logits)
    assert all(v == float('-inf') for v in result)


def test_validate_array_allows_valid_prefix(handler):
    """[1, 2, est un préfixe valide."""
    handler.llm.decode = lambda ids: '[1, 2,'
    logits = make_logits(VOCAB_SIZE)
    logits[0] = 1.0
    result = handler.validate_array([0], logits)
    assert result[0] == 1.0


def test_validate_array_allows_string_prefix(handler):
    """["a est un préfixe valide."""
    handler.llm.decode = lambda ids: '["a'
    logits = make_logits(VOCAB_SIZE)
    logits[0] = 1.0
    result = handler.validate_array([0], logits)
    assert result[0] == 1.0


def test_validate_array_allows_closing_bracket(handler):
    """[1, 2, 3] est valide et doit passer."""
    handler.llm.decode = lambda ids: '[1, 2, 3]'
    logits = make_logits(VOCAB_SIZE)
    logits[0] = 1.0
    result = handler.validate_array([0], logits)
    assert result[0] == 1.0


# ─── validate_object ──────────────────────────────────────────────────────────

def test_validate_object_first_token_forces_brace(handler):
    """Premier token doit être {."""
    handler.llm.decode = lambda ids: 'a'
    logits = make_logits(VOCAB_SIZE)
    result = handler.validate_object([], logits)
    assert all(v == float('-inf') for v in result)


def test_validate_object_allows_valid_prefix(handler):
    """{"key": est un préfixe valide."""
    handler.llm.decode = lambda ids: '{"key":'
    logits = make_logits(VOCAB_SIZE)
    logits[0] = 1.0
    result = handler.validate_object([0], logits)
    assert result[0] == 1.0


def test_validate_object_allows_closing_brace(handler):
    """{"key": "value"} est valide et doit passer."""
    handler.llm.decode = lambda ids: '{"key": "value"}'
    logits = make_logits(VOCAB_SIZE)
    logits[0] = 1.0
    result = handler.validate_object([0], logits)
    assert result[0] == 1.0


def test_validate_object_blocks_invalid(handler):
    """'xyz' n'est pas un préfixe valide d'objet."""
    handler.llm.decode = lambda ids: 'xyz'
    logits = make_logits(VOCAB_SIZE)
    result = handler.validate_object([0], logits)
    assert all(v == float('-inf') for v in result)