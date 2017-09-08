from . import classification, regression, stereo, unary, volume

unaries = {
    'resnet': unary.ResnetUnary,
}

volumes = {
    'concat': volume.CostVolumeConcat,
}

regressions = {
    'resnet': regression.ResnetRegression,
}

classifications = {
    'softargmin': classification.SoftArgmin
}


def create_unary(config):
    return unaries[config.unary](**config.config)


def create_cost_volume(config):
    return volumes[config.cost_volume](**config.config)


def create_regression(config):
    return regressions[config.regression](**config.config)


def create_classification(config):
    return classifications[config.classification](**config.config)


def create_model(config):
    return stereo.StereoRegression(
        create_unary(config),
        create_cost_volume(config),
        create_regression(config),
        create_classification(config)
    )
