import unittest

import torch

from TBMD.core.modal_processor.modes import (
    BatchModalProcessor,
    ModalProcessorConfig,
    ModalTensorStacker,
    ProcessingStrategy,
)


class TestTensorTimeInsensitiveModes(unittest.TestCase):
    def setUp(self):
        self.cores = {
            "subject1": torch.randn(5, 5, 3, 5),
            "subject2": torch.randn(5, 5, 3, 5),
        }
        self.factors = {
            "subject1": [
                torch.randn(10, 5),
                torch.randn(10, 5),
                torch.randn(5, 3),
                torch.randn(10, 5),
            ],
            "subject2": [
                torch.randn(10, 5),
                torch.randn(10, 5),
                torch.randn(5, 3),
                torch.randn(10, 5),
            ],
        }
        self.config = ModalProcessorConfig(
            processing_strategy=ProcessingStrategy.BATCH,
        )

    def test_BatchModalProcessor(self):
        processor = BatchModalProcessor(self.config)
        modal_tensors = processor.process_multiple_subjects(self.cores, self.factors)
        self.assertIn("subject1", modal_tensors)
        self.assertEqual(modal_tensors["subject1"].shape, (10, 10, 5, 5))

    def test_ModalTensorStacker(self):
        processor = BatchModalProcessor(self.config)
        modal_tensors = processor.process_multiple_subjects(self.cores, self.factors)
        stacker = ModalTensorStacker(self.config)
        A_tensor = stacker.stack_modal_tensors(modal_tensors)
        self.assertEqual(A_tensor.shape, (10, 10, 5, 10))


if __name__ == "__main__":
    unittest.main()
