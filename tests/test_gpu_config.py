import pytest

from gpu import worker


def test_resolve_gpu_count_auto_and_explicit(monkeypatch):
    monkeypatch.setattr(worker, "_detect_gpu_count", lambda: 4)
    assert worker._resolve_gpu_count("auto") == 4
    assert worker._resolve_gpu_count("all") == 4
    assert worker._resolve_gpu_count("2") == 2
    assert worker._resolve_gpu_count("8") == 4


def test_resolve_gpu_count_rejects_invalid_values(monkeypatch):
    monkeypatch.setattr(worker, "_detect_gpu_count", lambda: 2)
    with pytest.raises(ValueError):
        worker._resolve_gpu_count("0")
    with pytest.raises(ValueError):
        worker._resolve_gpu_count("nonsense")
