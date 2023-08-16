import io
from PIL import Image
from modules import script_callbacks
import matplotlib
import pandas as pd
from loractl.lib.lora_ctl_network import networks

log_weights = []
log_names = []
last_plotted_step = -1


# Copied from composable_lora
def plot_lora_weight(lora_weights, lora_names):
    data = pd.DataFrame(lora_weights, columns=lora_names)
    ax = data.plot()
    ax.set_xlabel("Steps")
    ax.set_ylabel("LoRA weight")
    ax.set_title("LoRA weight in all steps")
    ax.legend(loc=0)
    result_image = fig2img(ax)
    matplotlib.pyplot.close(ax.figure)
    del ax
    return result_image


# Copied from composable_lora
def fig2img(fig):
    buf = io.BytesIO()
    fig.figure.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def reset_plot():
    global last_plotted_step
    log_weights.clear()
    log_names.clear()


def make_plot():
    return plot_lora_weight(log_weights, log_names)


# On each step, capture our lora weights for plotting
def on_step(params):
    global last_plotted_step
    if last_plotted_step == params.sampling_step and len(log_weights) > 0:
        log_weights.pop()
    last_plotted_step = params.sampling_step
    if len(log_names) == 0:
        for net in networks.loaded_networks:
            log_names.append(net.name + "_te")
            log_names.append(net.name + "_unet")
    frame = []
    for net in networks.loaded_networks:
        frame.append(net.te_multiplier)
        frame.append(net.unet_multiplier)
    log_weights.append(frame)


script_callbacks.on_cfg_after_cfg(on_step)
