import torch.nn as nn


class StereoRegression(nn.Module):

    def __init__(self, unary, cost_volume, regression, classification):
        super(StereoRegression, self).__init__()
        self.unary = unary
        self.cost_volume = cost_volume
        self.regression = regression
        self.classification = classification

    def forward(self, *input):
        out = self.unary(*input)
        out = self.cost_volume(*out)
        out = self.regression(out)
        return self.classification(out)