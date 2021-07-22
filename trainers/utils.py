import math
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple

import torch
from torch import nn


def unnormalize_img(img, mean, std):
    std = img.new_tensor(std)[:, None, None]
    mean = img.new_tensor(mean)[:, None, None]

    return img * std + mean


def cosine_schedule(iter: int, max_iter: int = None, decay: float = None):
    slope = math.acos(decay)

    return math.cos(slope * iter / max_iter)


class MovingAverage(nn.Module):
    def __init__(
        self, size: Tuple[int, ...], buffer_size: int = 128, init_value: float = 0
    ):
        super().__init__()

        self.register_buffer(
            "buffer", torch.full((buffer_size,) + size, fill_value=init_value)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.buffer = torch.cat([self.buffer[1:], x[None]])

        return self.buffer.mean(dim=0)


class ExponentialMovingAverage(nn.Module):
    def __init__(
        self, size: Tuple[int, ...], momentum: float = 0.999, init_value: float = 0
    ):
        super().__init__()

        self.momentum = momentum
        self.register_buffer("avg", torch.full(size, fill_value=init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.avg += (self.avg - x) * (self.momentum - 1)

        return self.avg


class ExponentialMovingAverageModule(nn.Module):
    """Taken from: https://www.zijianhu.com/post/pytorch/ema/"""

    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1.0 - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.model(inputs)
        else:
            return self.shadow(inputs)
