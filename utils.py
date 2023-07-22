import numpy as np
import re

# Given a string like x@y,z@a, returns [[x, z], [y, a]] sorted for consumption by np.interp
def sorted_positions(raw_steps):
    steps = [[float(s.strip()) for s in x.split("@")] for x in re.split("[,;]", raw_steps)]
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

hires = False
def is_hires():
    return hires

def set_hires(value):
    global hires
    hires = value
