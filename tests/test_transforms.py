import unittest

import numpy as np
from PIL import Image

from src.data.transforms import BirdImageTransform, _rgb_moments


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "seed": 12,
            "preprocessing": {
                "input_size": 224,
                "resize_short_side": 256,
                "augmentation": {
                    "light": {
                        "crop_scale": [0.85, 1.0],
                        "rotation_degrees": 5,
                        "brightness": 0.08,
                        "contrast": 0.08,
                        "horizontal_flip_probability": 0.5,
                    },
                    "medium": {
                        "crop_scale": [0.7, 1.0],
                        "rotation_degrees": 10,
                        "brightness": 0.15,
                        "contrast": 0.15,
                        "horizontal_flip_probability": 0.5,
                    },
                },
            },
            "normalization": {"rgb_mean": [0.5, 0.5, 0.5], "rgb_std": [0.2, 0.2, 0.2]},
        }

    def test_grayscale_evaluation_becomes_three_channel_224_square(self):
        source = Image.new("L", (100, 200), color=128)
        output = BirdImageTransform(self.config, mode="evaluation").apply_image(source)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (224, 224))

    def test_train_augmentation_is_seeded_and_has_expected_size(self):
        source = Image.new("RGB", (300, 200), color=(80, 120, 160))
        first = BirdImageTransform(self.config, "train", "light", seed=12).apply_image(source)
        second = BirdImageTransform(self.config, "train", "light", seed=12).apply_image(source)
        self.assertEqual(first.size, (224, 224))
        self.assertEqual(list(first.getdata()), list(second.getdata()))

    def test_rgb_moments_include_grayscale_as_repeated_rgb(self):
        gray = Image.fromarray(np.full((2, 3), 128, dtype=np.uint8))
        mean, std = _rgb_moments([gray])
        self.assertEqual(len(mean), 3)
        self.assertAlmostEqual(mean[0], 128 / 255)
        self.assertEqual(mean[0], mean[1])
        self.assertEqual(std, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
