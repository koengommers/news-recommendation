from torch.optim.lr_scheduler import LambdaLR


class WarmupLinearDecayLR(LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup: float,
        steps_per_epoch: int,
        epochs: int,
        last_epoch: int = -1,
    ):
        self.total_steps = steps_per_epoch * epochs
        self.warmup_steps = warmup * self.total_steps
        last_step = ((last_epoch + 1) * steps_per_epoch) - 1
        super(WarmupLinearDecayLR, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_step
        )

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        remaining_steps = self.total_steps - step
        return remaining_steps / (self.total_steps - self.warmup_steps)
