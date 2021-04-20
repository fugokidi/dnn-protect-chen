from .resnet import *


def get_model(config):
    return globals()[config.architecture]()
