import torch.nn as nn
import torch.nn.functional as F


class StereoRegression(nn.Module):
    def __init__(self, unary, cost_volume, regression, classification):
        super(StereoRegression, self).__init__()
        self.unary = unary
        self.cost_volume = cost_volume
        self.regression = regression
        self.classification = classification
        self.criterion = None

    def forward_all(self, *input, reference='L'):
        assert reference in ['L', 'R']
        direction = 1 if reference == 'L' else -1
        unary = self.unary(*input)
        volume = self.cost_volume(*unary, direction)
        regression = self.regression(volume)
        classification = self.classification(regression)
        return unary, volume, regression, classification

    def forward(self, *input, reference='L'):
        unary, volume, regression, classification = self.forward_all(*input, reference=reference)
        return classification

    def set_criterion(self, criterion):
        self.criterion = criterion

    def classify(self, volume, size):
        return self.classification(F.upsample(volume, size, mode='trilinear'))

    def loss(self, target, *input, reference='L', unary_scale=0.0):
        unary, volume, regression, classification = self.forward_all(*input, reference=reference)
        loss = self.criterion(classification, target)
        if unary_scale > 0:
            disp_wta = self.classify(volume, regression.size()[2:])
            loss += unary_scale * self.criterion(disp_wta, target)
        return loss


class SeLuConsistencyStereoRegression(nn.Module):
    def __init__(self, unary, cost_volume, regression, classification):
        super(SeLuConsistencyStereoRegression, self).__init__()
        self.unary = unary
        self.cost_volume = cost_volume
        self.regression = regression
        self.classification = classification
        self.criterion = None

    def set_criterion(self, criterion):
        self.criterion = criterion

    def loss(self, target, left, right):
        dispL, dispLrec, dispR, explain = self.forward(left, right)
        return self.criterion(target, dispL, dispLrec, dispR, explain)

    def forward(self, left, right):
        unaryL, unaryR = self.unary(left, right)
        cost_volumeL = self.cost_volume(unaryL, unaryR, direction=1)
        cost_volumeR = self.cost_volume(unaryR, unaryL, direction=-1)
        regressionL = self.regression(cost_volumeL)
        regressionR = self.regression(cost_volumeR)
        dispL, dispLrec, dispR, explain = self.classification(regressionL, regressionR)
        return dispL, dispLrec, dispR, explain
