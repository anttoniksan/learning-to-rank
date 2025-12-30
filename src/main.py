import argparse
import tomllib
from argparse import BooleanOptionalAction
from pathlib import Path

import torch

from src.dataloader import MovieLensDataset
from src.train import train_dcnv2, train_deepfm, train_lgbm

DEEP_LEARNING_MODELS = ["deepfm", "dcnv2"]
INCLUDES_METADATA = ["dcnv2", "lgbm"]
SHOULD_NORMALIZE = ["deepfm", "dcnv2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train different models on MovieLens dataset"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to the directory containing the MovieLens data files",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["deepfm", "dcnv2", "lgbm"],
        default="deepfm",
        help="Model architecture for training.",
    )

    parser.add_argument(
        "--save",
        type=bool,
        action=BooleanOptionalAction,
        default=False,
        help="Whether to save the trained model to disk.",
    )

    return parser.parse_args()


def load_config(model_type: str) -> dict:
    configs_path = Path("configs") / f"{model_type}.toml"

    try:
        with open(configs_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        return None

    return config


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    model_type = args.model
    should_save = args.save

    data_file = data_path / "u.data"
    user_file = data_path / "u.user"
    movie_file = data_path / "u.item"

    include_metadata = model_type in INCLUDES_METADATA
    should_normalize = model_type in SHOULD_NORMALIZE

    dataset = MovieLensDataset(
        data_file=data_file,
        user_data=user_file,
        movie_data=movie_file,
        include_metadata=include_metadata,
        drop_ids=False,
        normalize=should_normalize,
    )

    data = dataset.data
    labels = dataset.labels

    config = load_config(model_type)

    match model_type:
        case "lgbm":
            model = train_lgbm(data, labels)
        case "dcnv2":
            model = train_dcnv2(data, labels, config)
        case "deepfm":
            model = train_deepfm(data, labels, config)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    if should_save and model_type in DEEP_LEARNING_MODELS:
        save_path = data_path / f"{model_type}_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
