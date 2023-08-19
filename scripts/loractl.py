import modules.scripts as scripts
from modules import extra_networks
from modules.processing import StableDiffusionProcessing
import gradio as gr
from loractl.lib import utils, plot, lora_ctl_network, network_patch


class LoraCtlScript(scripts.Script):
    def title(self):
        return "Dynamic Lora Weights"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Dynamic Lora Weights", open=False):
                opt_enable = gr.Checkbox(
                    value=True, label="Enable Dynamic Lora Weights")
                opt_plot_lora_weight = gr.Checkbox(
                    value=False, label="Plot the LoRA weight in all steps")
        return [opt_enable, opt_plot_lora_weight]

    def process(self, p: StableDiffusionProcessing, opt_enable=True, opt_plot_lora_weight=False, **kwargs):
        network_patch.apply()
        utils.set_hires(False)
        utils.set_active(opt_enable)
        lora_ctl_network.reset_weights()
        plot.reset_plot()

    def before_hr(self, p, *args):
        utils.set_hires(True)

    def postprocess(self, p, processed, opt_enable=True, opt_plot_lora_weight=False, **kwargs):
        if opt_plot_lora_weight and opt_enable:
            processed.images.extend([plot.make_plot()])
