from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


class SimpleClassifier(nn.Module):
    c_hid: int
    num_classes: int
    act_fn: str
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x, train=True, **kwargs):
        act_fn = getattr(nn.activation, self.act_fn)
        while x.shape[1] > 4:
            x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 32x32 => 16x16
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = act_fn(x)
            x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = act_fn(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.num_classes)(x)
        x = nn.log_softmax(x)
        return x
