import sys, unittest
from pathlib import Path
from loractl.lib.utils import sorted_positions, calculate_weight, params_to_weights

path = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, path)
from modules.extra_networks import ExtraNetworkParams
sys.path.remove(path)


class LoraCtlTests(unittest.TestCase):
    def test_sorted_positions(self):
        self.assertEqual(sorted_positions("1"), 1.0)
        self.assertEqual(sorted_positions("1@0,0.5@3,1@6"),
                         [[1.0, 0.5, 1.0], [0.0, 3.0, 6.0]])
        self.assertEqual(sorted_positions("0.5@3,1@6,1@0"),
                         [[1.0, 0.5, 1.0], [0.0, 3.0, 6.0]])
        self.assertEqual(sorted_positions("0.5@0,0.5@0.5,0@1"),
                         [[0.5, 0.5, 0.0], [0.0, 0.5, 1.0]])

    def test_sorted_position_semicolons(self):
        self.assertEqual(sorted_positions("1@0;0.5@3;1@6"),
                         [[1.0, 0.5, 1.0], [0.0, 3.0, 6.0]])

    def test_weight_interpolation(self):
        # Bare weights are never interpolated
        steps = sorted_positions("1.0")
        self.assertEqual(calculate_weight(steps, 0, 30), 1.0)
        self.assertEqual(calculate_weight(steps, 15, 30), 1.0)
        self.assertEqual(calculate_weight(steps, 30, 30), 1.0)

        # Weights are interpolated correctly
        steps = sorted_positions("0.75@0;0.5@3;1@6")
        self.assertEqual(calculate_weight(steps, 0, 30), 0.75)
        self.assertEqual(calculate_weight(steps, 3, 30), 0.5)
        self.assertEqual(calculate_weight(steps, 6, 30), 1.0)
        self.assertEqual(calculate_weight(steps, 9, 30), 1.0)

        # An implicit 0-step is added
        steps = sorted_positions("0.5@5,1.0@10")
        self.assertEqual(calculate_weight(steps, 0, 30), 0.5)
        self.assertEqual(calculate_weight(steps, 5, 30), 0.5)
        self.assertEqual(calculate_weight(steps, 8, 30), 0.8)
        self.assertEqual(calculate_weight(steps, 15, 30), 1.0)


class LoraCtlNetworkTests(unittest.TestCase):
    def assert_params(self, str, expected):
        params = ExtraNetworkParams(str.split(":"))
        self.assertEqual(params_to_weights(params), expected)

    def test_params_to_weights(self):
        # TE cascades to all
        self.assert_params("loraname:1.0", {
            'hrte': 1.0,
            'hrunet': 1.0,
            'te': 1.0,
            'unet': 1.0
        })

        # HR can be specified separately
        self.assert_params("loraname:0.5@0,1@1:hr=0.6", {
            'hrte': 0.6,
            'hrunet': 0.6,
            'te': [[0.5, 1.0], [0.0, 1.0]],
            'unet': [[0.5, 1.0], [0.0, 1.0]]
        })

        # Explicit TE cascades
        self.assert_params("loraname:te=0.5@0,1@1", {
            'te': [[0.5, 1.0], [0.0, 1.0]],
            'unet': [[0.5, 1.0], [0.0, 1.0]],
            'hrte': [[0.5, 1.0], [0.0, 1.0]],
            'hrunet': [[0.5, 1.0], [0.0, 1.0]],
        })

        # Implicit TE cascades, explicit unet cascades
        self.assert_params("loraname:unet=0.5@0,1@1", {
            'te': 1.0,
            'unet': [[0.5, 1.0], [0.0, 1.0]],
            'hrte': 1.0,
            'hrunet': [[0.5, 1.0], [0.0, 1.0]],
        })

        # Explicit HR TE overrides lowres TE
        self.assert_params("loraname:unet=0.5@0,1@1:hrte=0.5", {
            'te': 1.0,
            'unet': [[0.5, 1.0], [0.0, 1.0]],
            'hrte': 0.5,
            'hrunet': [[0.5, 1.0], [0.0, 1.0]],
        })

        # Explicit HR TE overrides HR
        self.assert_params("loraname:hr=0.6:hrte=0.5", {
            'te': 1.0,
            'unet': 1.0,
            'hrte': 0.5,
            'hrunet': 0.6,
        })

        self.assert_params("loraname:0.8@0.15,0@0.3:hr=0", {
            'hrte': 0.0,
            'hrunet': 0.0,
            'te': [[0.8, 0.0], [0.15, 0.3]],
            'unet': [[0.8, 0.0], [0.15, 0.3]]
        })
