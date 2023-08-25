from modules import extra_networks, script_callbacks, shared
from loractl.lib import utils

import sys, importlib
from pathlib import Path

# extensions-builtin isn't normally referencable due to the dash; this hacks around that
lora_path = str(Path(__file__).parent.parent.parent.parent.parent / "extensions-builtin" / "Lora")
sys.path.insert(0, lora_path)
import network, networks, network_lora, extra_networks_lora
sys.path.remove(lora_path)

lora_weights = {}


def reset_weights():
    lora_weights.clear()


class LoraCtlNetwork(extra_networks_lora.ExtraNetworkLora):
    # Hijack the params parser and feed it dummy weights instead so it doesn't choke trying to
    # parse our extended syntax
    def activate(self, p, params_list):
        if not utils.is_active():
            return super().activate(p, params_list)

        for params in params_list:
            assert params.items
            name = params.positional[0]
            if lora_weights.get(name, None) == None:
                lora_weights[name] = utils.params_to_weights(params)
            # The hardcoded 1 weight is fine here, since our actual patch looks up the weights from
            # our lora_weights dict
            params.positional = [name, 1]
            params.named = {}
        return super().activate(p, params_list)
