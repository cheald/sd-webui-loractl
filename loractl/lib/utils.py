import numpy as np
import re

# Given a string like x@y,z@a, returns [[x, z], [y, a]] sorted for consumption by np.interp


def sorted_positions(raw_steps):
    steps = [[float(s.strip()) for s in re.split("[@~]", x)]
             for x in re.split("[,;]", str(raw_steps))]
    # If we just got a single number, just return it
    if len(steps[0]) == 1:
        return steps[0][0]

    # Add implicit 1s to any steps which don't have a weight
    steps = [[s[0], s[1] if len(s) == 2 else 1] for s in steps]

    # Sort by index
    steps.sort(key=lambda k: k[1])

    steps = [list(v) for v in zip(*steps)]
    return steps


def calculate_weight(m, step, max_steps, step_offset=2):
    if isinstance(m, list):
        if m[1][-1] <= 1.0:
            if max_steps > 0:
                step = (step) / (max_steps - step_offset)
            else:
                step = 1.0
        else:
            step = step
        v = np.interp(step, m[1], m[0])
        return v
    else:
        return m


def params_to_weights(params):
    weights = {"unet": None, "te": 1.0, "hrunet": None, "hrte": None}

    if len(params.positional) > 1:
        weights["te"] = sorted_positions(params.positional[1])

    if len(params.positional) > 2:
        weights["unet"] = sorted_positions(params.positional[2])

    if params.named.get("te"):
        weights["te"] = sorted_positions(params.named.get("te"))

    if params.named.get("unet"):
        weights["unet"] = sorted_positions(params.named.get("unet"))

    if params.named.get("hr"):
        weights["hrunet"] = sorted_positions(params.named.get("hr"))
        weights["hrte"] = sorted_positions(params.named.get("hr"))

    if params.named.get("hrunet"):
        weights["hrunet"] = sorted_positions(params.named.get("hrunet"))

    if params.named.get("hrte"):
        weights["hrte"] = sorted_positions(params.named.get("hrte"))

    # If unet ended up unset, then use the te value
    weights["unet"] = weights["unet"] if weights["unet"] is not None else weights["te"]
    # If hrunet ended up unset, use unet value
    weights["hrunet"] = weights["hrunet"] if weights["hrunet"] is not None else weights["unet"]
    # If hrte ended up unset, use te value
    weights["hrte"] = weights["hrte"] if weights["hrte"] is not None else weights["te"]

    return weights


hires = False
loractl_active = True

def is_hires():
    return hires


def set_hires(value):
    global hires
    hires = value


def set_active(value):
    global loractl_active
    loractl_active = value

def is_active():
    global loractl_active
    return loractl_active
