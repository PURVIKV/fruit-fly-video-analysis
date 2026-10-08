"""
Saving trained models.

Each model file has exactly one trainer that may write it. This guards
against the earlier bug where train/train_fly_count.py (a copy of the sex
trainer) silently overwrote models/sex_model.pkl.
"""

from pathlib import Path

import joblib


MODEL_DIR = Path("models")

MODEL_OWNERS = {
    "fly_count_model.pkl": "train_fly_count.py",
    "sex_model.pkl": "train_sex.py",
    "orientation_model.pkl": "train_orientation.py",
    "wing_angle_model.pkl": "train_wing_angle.py"
}


def save_model(model, model_path, trainer_file):
    """
    Save model to model_path, but only if trainer_file (pass __file__)
    is the script registered as the owner of that model file.
    """

    model_path = Path(model_path)

    trainer_name = Path(trainer_file).name

    owner = MODEL_OWNERS.get(model_path.name)

    if owner is None or owner != trainer_name or model_path.parent != MODEL_DIR:
        raise PermissionError(
            f"{trainer_name} is not allowed to write {model_path} "
            f"(owner: {owner})"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    return model_path
