from collections import defaultdict
from typing import Any

import numpy as np
import torch
from lightgbm import LGBMRanker
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.early_stopper import EarlyStopper
from src.models.dcnv2 import DCNV2
from src.models.deepfm import DeepFM


def _get_groups(data: list[tuple[int, int, int]]) -> dict[int, int]:
    groups = defaultdict(int)
    for user_id, _, _ in data:
        groups[user_id] += 1
    return groups


DEFAULT_TEST_SIZE = 0.1
DEFAULT_RANDOM_STATE = None


def train_lgbm(data: list[tuple[int, int, int]], labels: list[int]):
    print(f"Training with {len(data)} samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )

    X_train, y_train = zip(*sorted(zip(X_train, y_train), key=lambda x: x[0][0]))
    X_test, y_test = zip(*sorted(zip(X_test, y_test), key=lambda x: x[0][0]))

    train_groups = list(_get_groups(X_train).values())
    test_groups = list(_get_groups(X_test).values())

    ranker_params = {"objective": "lambdarank", "n_estimators": 20}

    ranker = LGBMRanker(**ranker_params)
    ranker = ranker.fit(
        X=np.array(X_train),
        y=np.array(y_train),
        group=train_groups,
        eval_set=[(np.array(X_test), np.array(y_test))],
        eval_group=[test_groups],
        eval_metric=["map"],
    )

    print("Evaluation on test set:")
    print(ranker.predict(np.array(X_test[:5])))
    print("Ground truth:")
    print(np.array(y_test[:5]))

    return ranker


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def _evaluate_mae(model, val_loader):
    total_labels = []
    total_predictions = []

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            predictions = outputs.squeeze(-1)
            labels = batch_y.view(-1)
            total_predictions.extend(predictions.tolist())
            total_labels.extend(labels.tolist())

    # Compute MAE and multiply by 5 to scale back to original rating range
    return 5.0 * mean_absolute_error(total_labels, total_predictions)


def train_deepfm(data: list[Any], labels: list[int], config: Any):
    assert config is not None, "Config for DeepFM training is required."

    device = _get_device()
    print(f"Using device: {device}")

    X = torch.tensor(data, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)

    train_split = config.get("data", {}).get("train_split", 0.8)
    val_split = config.get("data", {}).get("val_split", 0.2)

    dataset = TensorDataset(X, y)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_split, val_split]
    )

    print(f"Training with {len(data)} samples.")
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    n_epochs = config.get("training", {}).get("n_epochs", 100)
    batch_size = config.get("data", {}).get("batch_size", 64)
    shuffle = config.get("data", {}).get("shuffle", True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    n_features = X.shape[1]
    # Calculate the number of unique values for each feature (field)
    # This sets the offsets for the embedding layer
    field_dims = [torch.max(X[:, i]).item() + 1 for i in range(n_features)]

    model = DeepFM(
        n_features=n_features,
        field_dims=field_dims,
        embedding_dim=config.get("model", {}).get("embedding_dim", 64),
    ).to(device=device)
    early_stopper = EarlyStopper(patience=5)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("optimizer", {}).get("learning_rate", 0.001),
        weight_decay=config.get("optimizer", {}).get("weight_decay", 1e-5),
    )

    iterator = tqdm(range(n_epochs), desc="Training DeepFM")
    evaluation_interval = config.get("training", {}).get("evaluation_interval", 10)

    for epoch in iterator:
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(-1), batch_y.view(-1))
            loss.backward()
            optimizer.step()
        iterator.set_postfix(loss=loss.item())

        if (epoch + 1) % evaluation_interval == 0:
            print("Evaluating on validation set...")
            mae = _evaluate_mae(model, val_loader)
            print(f"Validation MAE after epoch {epoch + 1}: {mae:.4f}")

            if early_stopper.step(mae):
                print("Early stopping triggered. Stopping training.")
                break

    return model


def train_dcnv2(data: list[Any], labels: list[int], config: Any):
    assert config is not None, "Config for DCNv2 training is required."

    device = _get_device()
    print(f"Using device: {device}")

    X = torch.tensor(data, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)

    train_split = config.get("data", {}).get("train_split", 0.8)
    val_split = config.get("data", {}).get("val_split", 0.2)

    dataset = TensorDataset(X, y)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_split, val_split]
    )

    print(f"Training with {len(data)} samples.")
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    n_epochs = config.get("training", {}).get("n_epochs", 100)
    batch_size = config.get("data", {}).get("batch_size", 64)
    shuffle = config.get("data", {}).get("shuffle", True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    n_features = X.shape[1]
    # Calculate the number of unique values for each feature (field)
    # This sets the offsets for the embedding layer
    field_dims = [torch.max(X[:, i]).item() + 1 for i in range(n_features)]

    embedding_dim = config.get("model", {}).get("embedding_dim", 64)
    dropout = config.get("model", {}).get("dropout", 0.1)
    structure = config.get("model", {}).get("structure", "stacked")
    n_layers = config.get("model", {}).get("n_layers", 2)

    model = DCNV2(
        n_features=n_features,
        field_dims=field_dims,
        embedding_dim=embedding_dim,
        dropout=dropout,
        structure=structure,
        n_layers=n_layers,
    ).to(device=device)
    early_stopper = EarlyStopper(patience=5)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("optimizer", {}).get("learning_rate", 0.001),
        weight_decay=config.get("optimizer", {}).get("weight_decay", 1e-5),
    )

    iterator = tqdm(range(n_epochs), desc="Training DCNv2")
    evaluation_interval = config.get("training", {}).get("evaluation_interval", 10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in iterator:
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(-1), batch_y.view(-1))
            loss.backward()

            # From paper: https://arxiv.org/pdf/2008.13535 part 7.1 Optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

        iterator.set_postfix(loss=loss.item())
        scheduler.step()

        if (epoch + 1) % evaluation_interval == 0:
            print("Evaluating on validation set...")
            mae = _evaluate_mae(model, val_loader)
            print(f"Validation MAE after epoch {epoch + 1}: {mae:.4f}")

            if early_stopper.step(mae):
                print("Early stopping triggered. Stopping training.")
                break

    return model
