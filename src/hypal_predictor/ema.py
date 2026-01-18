class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema = None

    def update(self, loss: float) -> float:
        if self.ema is None:
            self.ema = loss
        else:
            self.ema = self.alpha * loss + (1 - self.alpha) * self.ema
        return self.ema
