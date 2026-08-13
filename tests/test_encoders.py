"""Tests for stage3_model/encoders.py.

Only `resolve_encoder`'s fallback logic and `load_encoder`'s dispatch are
exercised here. No test calls a real per-backend loader: every one of
phikon/resnet50/uni/musk needs either a network download or gated HF access,
which tests must never do.
"""
from __future__ import annotations

import pytest

from canvas_brca.stage3_model.encoders import ENCODERS, load_encoder, resolve_encoder


def test_ungated_encoder_resolves_to_itself():
    spec = resolve_encoder("phikon")
    assert spec.name == "phikon"


def test_gated_encoder_without_token_falls_back_to_phikon(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    spec = resolve_encoder("uni")
    assert spec.name == "phikon"


def test_gated_encoder_with_token_resolves_to_itself(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    spec = resolve_encoder("uni")
    assert spec.name == "uni"


def test_musk_without_token_falls_back_to_phikon(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    spec = resolve_encoder("musk")
    assert spec.name == "phikon"


def test_unknown_encoder_raises():
    with pytest.raises(ValueError):
        resolve_encoder("not-a-real-encoder")


def test_load_encoder_rejects_unwired_name():
    with pytest.raises(NotImplementedError):
        load_encoder(ENCODERS["ctranspath"])


def test_load_encoder_rejects_gated_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        load_encoder(ENCODERS["uni"])
