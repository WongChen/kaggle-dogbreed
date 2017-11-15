import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import resnet

class DogBreedNet(nn.Module):
    def __init__(self):
        super(DogBreedNet).__init__(self)

