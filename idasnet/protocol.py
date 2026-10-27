"""Shared training defaults and access to module-owned experiment settings."""
from copy import deepcopy
import json
from pathlib import Path
from idasnet.statistics import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED

STUDY_TITLE = (
    'IDASNet: An Interpretable Dual-Attention Spectral Network for Hyperspectral Prediction '
    'of Grape Total Soluble Solids'
)
PROTOCOL_VERSION = '1.0'
TRAINING = {
    'outer_splits': 5,
    'outer_repeats': 5,
    'outer_base_seed': 2026,
    'inner_splits': 4,
    'n_trials': 30,
    'max_epochs': 500,
    'patience': 50,
    'batch_size': 8,
    'train_seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'validation_fraction': 0.2,
    'bootstrap_iterations': BOOTSTRAP_ITERATIONS,
    'bootstrap_seed': BOOTSTRAP_SEED,
}


def load_run_config(path):
    """Read saved settings while accepting retired non-training metadata."""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if settings.pop("smoke_test", False):
        raise ValueError("This run used a discontinued quick-check mode; start a new experiment run")
    settings.pop("save_plots", None)
    return settings


def suggest(trial, model_type="IDASNet"):
    if model_type == "IDASNet":
        from idasnet.model import SEARCH_SPACE
    else:
        from baselines.neural_models import SEARCH_SPACE
    values = {
        name: (trial.suggest_float(name, **options) if isinstance(options, dict)
               else trial.suggest_categorical(name, options))
        for name, options in SEARCH_SPACE.items()
    }
    if model_type != "IDASNet":
        values["model_type"] = model_type
    return values


def protocol_settings():
    """Collect a detached settings snapshot for saved runs and resume checks."""
    from data import NORMALIZATION
    from data.build import PREPROCESSING
    from idasnet.model import SEARCH_SPACE as idasnet_space
    from baselines.neural_models import SEARCH_SPACE as neural_space
    from baselines.plsr import DEFAULT_SEARCH_SPACE as plsr_space
    from baselines.rf import DEFAULT_SEARCH_SPACE as rf_space
    from baselines.svr import DEFAULT_SEARCH_SPACE as svr_space
    from experiments.joint_inputs import DEFAULT_METHOD_TOPKS, DEFAULT_WAVELENGTH_TOPKS

    return deepcopy({
        "title": STUDY_TITLE,
        "protocol_version": PROTOCOL_VERSION,
        "training": TRAINING,
        "input_combinations": {
            "method_counts": DEFAULT_METHOD_TOPKS,
            "wavelength_counts": DEFAULT_WAVELENGTH_TOPKS,
        },
        "normalization": NORMALIZATION,
        "preprocessing": PREPROCESSING,
        "IDASNet": idasnet_space,
        "neural_baselines": neural_space,
        "PLSR": plsr_space,
        "SVR": svr_space,
        "RF": rf_space,
    })
