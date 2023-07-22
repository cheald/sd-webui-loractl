from modules import extra_networks, shared, script_callbacks
import utils

import sys, os
# extensions-builtin isn't normally referencable due to the dash; this hacks around that
sys.path.append(os.path.join("extensions-builtin", "Lora"))
import network, networks, network_lora, extra_networks_lora
sys.path.pop()

lora_weights = {}

def reset_weights():
    lora_weights.clear()

class LoraCtlNetwork(extra_networks_lora.ExtraNetworkLora):
    """
    Replace the standard LoraNetwork with one that understands additional syntax for weights. Both
    positional arguments and named arguments are honored.

    The standard form is:

        <lora:network_name[:te_multiplier[:unet_multiplier[:dyn_dim]]]>

    te_multiplier and unet_multiplier may be any of:

    * A single number, which will be used for all steps
    * A comma-separated list of weight-at-step pairs, e.g. "0@0,1@0.5,0@1" to start at 0, go to
      1 at step 0.5, and return to 0 at step 1

    The step value (after the @) may be a float in the 0.0-1.0 domain, in which case it is
    interpreted as a percentage. If it is greater than 1, then it is interpreted as an
    absolute step number.

    You may also use named arguments, such as:

        <lora:network_name:te=0@0,1@0.5,0@1:unet=0@0,1@0.5,0@1:dyn=256>

    This is functionally identical, but may be more readable.

    If only a single argument (or just te) is given, then it applies to both the text encoder
    and the unet.

    The default weight for the network at step 0 is the earliest weight given. That is:

        0.25@0.5,1@1

    will begin at 0.25 weight, stay there until until half the steps are run, then interpolate
    up to 1.0 for the final step.

    """
    def activate(self, p, params_list):
        additional = shared.opts.sd_lora

        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        for params in params_list:
            assert params.items
            name = params.positional[0]
            names.append(name)

            te = 1.0
            unet = 1.0

            if not name in lora_weights:
                lora_weights[name] = {"unet": None, "te": 1.0}
            if len(params.positional) > 1:
                lora_weights[name]["te"] = utils.sorted_positions(params.positional[1])
            if len(params.positional) > 2:
                lora_weights[name]["unet"] = utils.sorted_positions(params.positional[2])

            if params.named.get("te"):
                lora_weights[name]["te"] = utils.sorted_positions(params.named.get("te"))
            if params.named.get("unet"):
                lora_weights[name]["unet"] = utils.sorted_positions(params.named.get("unet"))
            if params.named.get("hr"):
                lora_weights[name]["hrunet"] = utils.sorted_positions(params.named.get("hr"))
                lora_weights[name]["hrte"] = utils.sorted_positions(params.named.get("hr"))
            if params.named.get("hrunet"):
                lora_weights[name]["hrunet"] = utils.sorted_positions(params.named.get("hrunet"))
            if params.named.get("hrte"):
                lora_weights[name]["hrte"] = utils.sorted_positions(params.named.get("hrte"))

            if lora_weights[name]["unet"] == None:
                lora_weights[name]["unet"] = lora_weights[name]["te"]

            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim

            te_multipliers.append(te)
            unet_multipliers.append(unet)
            dyn_dims.append(dyn_dim)

        networks.load_networks(names,
            te_multipliers=te_multipliers,
            unet_multipliers=unet_multipliers,
            dyn_dims=dyn_dims
        )

        if shared.opts.lora_add_hashes_to_infotext:
            network_hashes = []
            for item in networks.loaded_networks:
                shorthash = item.network_on_disk.shorthash
                if not shorthash:
                    continue

                alias = item.mentioned_name
                if not alias:
                    continue

                alias = alias.replace(":", "").replace(",", "")

                network_hashes.append(f"{alias}: {shorthash}")

            if network_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(network_hashes)

def before_ui():
    extra_networks.register_extra_network(LoraCtlNetwork())
script_callbacks.on_before_ui(before_ui)
