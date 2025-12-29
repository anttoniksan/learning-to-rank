class EarlyStopper:
    """Utility class for early stopping during training.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
    """

    def __init__(self, patience: int = 1):
        self.best_loss = float("inf")
        self.counter = 0
        self.patience = patience

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
