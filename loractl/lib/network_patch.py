from modules import shared
import numpy as np

from loractl.lib.lora_ctl_network import network, lora_weights
from loractl.lib.utils import calculate_weight, is_hires

# Patch network.Network so it reapplies properly for dynamic weights
# By default, network application is cached, with (name, te, unet, dim) as a key
# By replacing the bare properties with getters, we can ensure that we cause SD
# to reapply the network each time we change its weights, while still taking advantage
# of caching when weights are not updated.


def get_weight(m):
    return calculate_weight(m, shared.state.sampling_step, shared.state.sampling_steps, step_offset=2)


def get_dynamic_te(self):
    if self.name in lora_weights:
        key = "te" if not is_hires() else "hrte"
        w = lora_weights[self.name]
        return get_weight(w.get(key, self._te_multiplier))

    return get_weight(self._te_multiplier)


def get_dynamic_unet(self):
    if self.name in lora_weights:
        key = "unet" if not is_hires() else "hrunet"
        w = lora_weights[self.name]
        return get_weight(w.get(key, self._unet_multiplier))

    return get_weight(self._unet_multiplier)


def set_dynamic_te(self, value):
    self._te_multiplier = value


def set_dynamic_unet(self, value):
    self._unet_multiplier = value


def apply():
    if getattr(network.Network, "te_multiplier", None) == None:
        network.Network.te_multiplier = property(get_dynamic_te, set_dynamic_te)
        network.Network.unet_multiplier = property(
            get_dynamic_unet, set_dynamic_unet)
