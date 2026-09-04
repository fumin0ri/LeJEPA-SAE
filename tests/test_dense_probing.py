import json
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lejepa_sae import dense_probing
from lejepa_sae.config import ExperimentConfig


@pytest.fixture
def split_stub(monkeypatch):
    pytest.importorskip("sklearn")
    package = ModuleType("sae_probes")
    training = ModuleType("sae_probes.utils_training")
    training.get_cv = lambda features: None
    training.get_splits = lambda cv, features, labels: [
        (list(range(len(features) - 20)), list(range(len(features) - 20, len(features))))
    ]
    monkeypatch.setitem(sys.modules, "sae_probes", package)
    monkeypatch.setitem(sys.modules, "sae_probes.utils_training", training)


def test_dense_solver_uses_the_last_feature_and_is_deterministic(split_stub):
    labels = torch.tensor([index % 2 for index in range(80)])
    features = torch.zeros(80, 7)
    features[:, -1] = labels.float() * 4 - 2
    test_labels = torch.tensor([index % 2 for index in range(20)])
    test_features = torch.zeros(20, 7)
    test_features[:, -1] = test_labels.float() * 4 - 2

    kwargs = dict(device="cpu", cs=[1.0], max_iter=100)
    first = dense_probing.fit_dense_logistic_probe(
        features, labels, test_features, test_labels, **kwargs
    )
    second = dense_probing.fit_dense_logistic_probe(
        features, labels, test_features, test_labels, **kwargs
    )
    assert first == second
    assert first["feature_dim"] == 7
    assert first["best_c"] == 1.0
    assert first["test_auc"] == pytest.approx(1.0)
    assert first["test_f1"] == pytest.approx(1.0)


def test_l2_does_not_penalize_intercept():
    features = torch.zeros(4, 3)
    labels = torch.tensor([0, 1, 1, 1])
    fit = dense_probing._fit_logistic(features, labels, c_value=1e-5, device="cpu", max_iter=100)
    torch.testing.assert_close(fit.weight, torch.zeros_like(fit.weight))
    assert float(fit.bias) > 0
    assert fit.loss < 0.7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dense_solver_parameters_and_features_are_on_cuda():
    features = torch.randn(32, 8)
    labels = torch.tensor([0, 1] * 16)
    fit = dense_probing._fit_logistic(features, labels, c_value=1.0, device="cuda", max_iter=20)
    assert fit.weight.device.type == "cuda"
    assert fit.bias.device.type == "cuda"
    assert math.isfinite(fit.loss)


@pytest.mark.parametrize(
    "corruption",
    [
        lambda train, test: (train.fill_(float("nan")), test),
        lambda train, test: (train, test.fill_(float("inf"))),
    ],
)
def test_dense_solver_rejects_nonfinite_features(split_stub, corruption):
    train = torch.randn(40, 4)
    test = torch.randn(20, 4)
    train, test = corruption(train, test)
    with pytest.raises(ValueError, match="non-finite"):
        dense_probing.fit_dense_logistic_probe(
            train,
            torch.tensor([0, 1] * 20),
            test,
            torch.tensor([0, 1] * 10),
            device="cpu",
            cs=[1.0],
            max_iter=10,
        )


def _dense_fit_result(feature_dim):
    return {
        "feature_dim": feature_dim,
        "best_c": 1.0,
        "val_auc": 0.6,
        "test_f1": 0.7,
        "test_acc": 0.8,
        "test_auc": 0.9,
        "validation": [],
        "optimizer": {"converged": True},
    }


def test_dense_runner_uses_official_normal_split_and_resumes(tmp_path, monkeypatch):
    package = ModuleType("sae_probes")
    package.DATASETS = ["task-b", "task-a"]
    generation = ModuleType("sae_probes.generate_sae_activations")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            X_train=torch.randn(8, 4),
            y_train=torch.tensor([0, 1] * 4),
            X_test=torch.randn(4, 4),
            y_test=torch.tensor([0, 1] * 2),
        )

    generation.generate_sae_activations = generate
    monkeypatch.setitem(sys.modules, "sae_probes", package)
    monkeypatch.setitem(sys.modules, generation.__name__, generation)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dense_probing, "load_model", lambda *args: torch.nn.Linear(4, 4))

    class FakeAdapter(torch.nn.Linear):
        def to(self, *args, **kwargs):
            return self

    monkeypatch.setattr(dense_probing, "ProbeSAEAdapter", lambda *args: FakeAdapter(4, 4))
    monkeypatch.setattr(dense_probing, "_cache_spec", lambda *args: {"cache": "test"})
    monkeypatch.setattr(dense_probing, "prepare_activation_cache", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(dense_probing, "_release_memory", lambda *args: None)
    fits = []
    monkeypatch.setattr(
        dense_probing,
        "fit_dense_logistic_probe",
        lambda *args, **kwargs: fits.append((args, kwargs)) or _dense_fit_result(4),
    )

    config = ExperimentConfig()
    config.model.d_llm = 4
    config.model.feature_dim = 4
    config.loss.axis_projections = 2
    config.train.device = "cuda:0"
    config.train.precision = "float32"
    config.baseline.k = 2
    config.validate()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "dense-results"
    kwargs = dict(datasets=["task-a"], parity=False, cs=[1.0], max_iter=10)
    dense_probing.run_dense_z_probes(config, checkpoint, output, tmp_path / "cache", **kwargs)
    assert len(calls) == 1 and len(fits) == 1
    assert calls[0]["setting"] == "normal"
    assert calls[0]["seed"] == 42
    assert calls[0]["dataset"] == "task-a"
    assert (output / "dense_probe_manifest.json").is_file()
    summary = json.loads((output / "dense_z_probe_summary.json").read_text())
    assert summary["complete"] and summary["tasks"] == ["task-a"]
    assert summary["macro_metrics"] == {"f1": 0.7, "auroc": 0.9, "accuracy": 0.8}

    dense_probing.run_dense_z_probes(config, checkpoint, output, tmp_path / "cache", **kwargs)
    assert len(calls) == 1 and len(fits) == 1
    with pytest.raises(ValueError, match="new results-path"):
        dense_probing.run_dense_z_probes(
            config,
            checkpoint,
            output,
            tmp_path / "cache",
            datasets=["task-a"],
            parity=False,
            cs=[0.5],
            max_iter=10,
        )


def test_dense_runner_requires_cuda_before_loading_checkpoint(tmp_path, monkeypatch):
    package = ModuleType("sae_probes")
    package.DATASETS = ["task-a"]
    generation = ModuleType("sae_probes.generate_sae_activations")
    generation.generate_sae_activations = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "sae_probes", package)
    monkeypatch.setitem(sys.modules, generation.__name__, generation)
    config = ExperimentConfig()
    config.train.device = "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        dense_probing.run_dense_z_probes(config, "missing.pt", tmp_path, tmp_path)
