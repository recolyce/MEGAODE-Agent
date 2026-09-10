"""Model registry. build(name) returns a bundle-native estimator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.models.graph_ode import GraphOmicsODEModel
from src.models.multimodal import (
    CrossAttnFusionModel,
    DualLSTMModel,
    GatedFusionModel,
    KoopmanAEModel,
    MMVAEForecastModel,
    MogonetFusionModel,
)
from src.models.neural import FeatureChunkLSTMModel, MLPModel, NeuralODEModel
from src.models.prior_fusion import PriorFusionMLPModel

Factory = Callable[..., Any]

# Library is ML / multimodal only. Non-ML baselines (mean, last_value, ridge) are not registered.
MODELS: dict[str, tuple[Factory, bool]] = {
    "mlp": (MLPModel, False),
    "neural_ode": (NeuralODEModel, False),
    "feature_chunk_lstm": (FeatureChunkLSTMModel, False),
    "cross_attn_fusion": (CrossAttnFusionModel, False),
    "gated_fusion": (GatedFusionModel, False),
    "koopman_ae": (KoopmanAEModel, False),
    "dual_lstm": (DualLSTMModel, False),
    "mmvae_forecast": (MMVAEForecastModel, False),
    "mogonet_fusion": (MogonetFusionModel, False),
    "graph_omics_ode": (GraphOmicsODEModel, True),
    "prior_fusion_mlp": (PriorFusionMLPModel, True),
}

GENERATED: dict[str, tuple[Factory, bool]] = {}
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_MODELS = list(MODELS)


def _load_generated() -> None:
    import importlib.util
    import inspect

    if not GENERATED_DIR.exists():
        return
    for path in sorted(GENERATED_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"src.models.generated.{path.stem}", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, "name", None) and hasattr(obj, "fit") and hasattr(obj, "predict"):
                GENERATED[str(obj.name)] = (obj, bool(getattr(obj, "requires_prior", True)))


_load_generated()


def available_models() -> list[str]:
    return list(MODELS) + [name for name in GENERATED if name not in MODELS]


def model_role(name: str) -> str:
    if name in MODELS:
        return "library_learned"
    return "generated"


def coerce_eval_models(picked: list[str] | None = None, extra: list[str] | None = None) -> list[str]:
    """Keep library ML models (or an explicit learned subset) plus this-run extras."""
    catalog = set(_catalog())
    library = [name for name in MODELS if name in catalog]
    picked_ok = [name for name in (picked or []) if name in catalog]
    out = [name for name in picked_ok if name in MODELS] or list(library)
    for name in extra or []:
        token = str(name or "")
        if token and token in catalog and token not in out:
            out.append(token)
    return out


def register(name: str, factory: Factory, requires_prior: bool = True) -> None:
    GENERATED[name] = (factory, requires_prior)


def _catalog() -> dict[str, tuple[Factory, bool]]:
    return {**MODELS, **GENERATED}


class ModelRegistry:
    def list(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "requires_prior": needs_prior, "generated": name in GENERATED}
            for name, (_factory, needs_prior) in _catalog().items()
        ]

    def build(self, name: str, **hparams: Any) -> Any:
        catalog = _catalog()
        if name not in catalog:
            raise KeyError(f"Unknown model {name!r}. Available: {sorted(catalog)}")
        factory, _needs = catalog[name]
        return factory(**hparams)
