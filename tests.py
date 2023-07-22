import unittest

from utils import sorted_positions, calculate_weight

class LoraCtlTests(unittest.TestCase):
    def test_sorted_positions(self):
        self.assertEqual(sorted_positions("1"), 1.0)
        self.assertEqual(sorted_positions("1@0,0.5@3,1@6"), [[1.0, 0.5, 1.0], [0.0, 3.0, 6.0]])
        self.assertEqual(sorted_positions("0.5@3,1@6,1@0"), [[1.0, 0.5, 1.0], [0.0, 3.0, 6.0]])
        self.assertEqual(sorted_positions("0.5@0,0.5@0.5,0@1"), [[0.5, 0.5, 0.0], [0.0, 0.5, 1.0]])


    def test_sorted_position_semicolons(self):
        self.assertEqual(sorted_positions("1@0;0.5@3;1@6"), [[1.0, 0.5, 1.0], [0.0, 3.0, 6.0]])


    def test_weight_interpolation(self):
        # Bare weights are never interpolated
        steps = sorted_positions("1.0")
        self.assertEqual( calculate_weight(steps, 0, 30), 1.0 )
        self.assertEqual( calculate_weight(steps, 15, 30), 1.0 )
        self.assertEqual( calculate_weight(steps, 30, 30), 1.0 )

        # Weights are interpolated correctly
        steps = sorted_positions("0.75@0;0.5@3;1@6")
        self.assertEqual( calculate_weight(steps, 0, 30), 0.75 )
        self.assertEqual( calculate_weight(steps, 3, 30), 0.5 )
        self.assertEqual( calculate_weight(steps, 6, 30), 1.0 )
        self.assertEqual( calculate_weight(steps, 9, 30), 1.0 )

        # An implicit 0-step is added
        steps = sorted_positions("0.5@5,1.0@10")
        self.assertEqual( calculate_weight(steps, 0, 30), 0.5 )
        self.assertEqual( calculate_weight(steps, 5, 30), 0.5 )
        self.assertEqual( calculate_weight(steps, 8, 30), 0.8 )
        self.assertEqual( calculate_weight(steps, 15, 30), 1.0 )
