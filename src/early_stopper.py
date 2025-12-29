class EarlyStopper:
    """Utility class for early stopping during training.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
    """

    def __init__(self, patience: int = 1, min_delta=0.01):
        self.best_loss = float("inf")
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            if current_loss - self.best_loss < self.min_delta:
                return False

            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
