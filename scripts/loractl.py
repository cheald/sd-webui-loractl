import modules.scripts as scripts
from modules import extra_networks
from modules.processing import StableDiffusionProcessing
import gradio as gr
import utils, plot, lora_ctl_network, network_patch


class LoraCtlScript(scripts.Script):
    def title(self):
        return "Dynamic Lora Weights"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Dynamic Lora Weights", open=False):
                opt_plot_lora_weight = gr.Checkbox(value=False, label="Plot the LoRA weight in all steps")
        return [opt_plot_lora_weight]

    def process(self, p: StableDiffusionProcessing, opt_plot_lora_weight: bool):
        utils.set_hires(False)
        lora_ctl_network.reset_weights()
        plot.reset_plot()

    def before_hr(self, p, *args):
        utils.set_hires(True)

    def postprocess(self, p, processed, opt_plot_lora_weight: bool):
        if opt_plot_lora_weight:
            processed.images.extend([plot.make_plot()])


network_patch.apply()
