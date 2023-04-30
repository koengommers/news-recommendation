import torch.nn as nn


class PredictionLoss(nn.Module):
    def __init__(self, weight=1):
        super(PredictionLoss, self).__init__()

        self.weight = weight
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        values,
    ):
        click_probability = values["click_probability"]
        labels = values["labels"]
        return self.weight * self.loss_fn(click_probability, labels)
