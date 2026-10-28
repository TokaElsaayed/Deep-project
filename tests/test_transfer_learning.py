import unittest

import torch

from src.models.transfer_learning import configure_trainable_layers, create_transfer_model


class TransferModelTests(unittest.TestCase):
    def test_model_factories_return_200_class_logits(self):
        for model_name in ("resnet50", "efficientnet_b3"):
            with self.subTest(model=model_name):
                model, _ = create_transfer_model(model_name, num_classes=200, pretrained=False)
                model.eval()
                input_size = 300 if model_name == "efficientnet_b3" else 224
                with torch.inference_mode():
                    output = model(torch.zeros(2, 3, input_size, input_size))
                self.assertEqual(tuple(output.shape), (2, 200))

    def test_feature_extraction_freezes_backbone(self):
        model, _ = create_transfer_model("resnet50", pretrained=False)
        configure_trainable_layers(model, "resnet50", "feature_extraction")
        self.assertTrue(all(p.requires_grad for p in model.fc.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.layer4.parameters()))


if __name__ == "__main__":
    unittest.main()
