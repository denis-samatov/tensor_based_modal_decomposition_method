import unittest
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import tempfile
import sys
import os

# Ensure algorithm directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../algorithm')))

from algorithm.TBMD.utils.DataLoader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.num_images = 5
        self.image_size = (10, 10)
        self.subject_id = "test_subject"

        subject_dir = Path(self.test_dir) / self.subject_id
        subject_dir.mkdir()

        self.generated_images = []
        for i in range(self.num_images):
            # Create a random image
            img_array = np.random.randint(0, 256, self.image_size + (3,), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = subject_dir / f"PRESSURE_STEP_{i}.png"
            img.save(img_path)
            # Store normalized float array for comparison
            self.generated_images.append(img_array.astype(np.float32) / 255.0)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_images_tensor(self):
        loader = DataLoader()
        data, subjects = loader.load_images_tensor(self.test_dir)

        self.assertIn(self.subject_id, subjects)
        self.assertIn(self.subject_id, data)

        loaded_tensor = data[self.subject_id]

        # Expected shape: (H, W, 3, T) or (H, W, T) depending on implementation
        # Current implementation:
        # img_array = np.array(img, dtype=np.float32) / 255.0  -> shape (H, W, 3)
        # image_list.append(img_array)
        # np.stack(image_list, axis=-1) -> shape (H, W, 3, T)

        expected_shape = self.image_size + (3, self.num_images)
        self.assertEqual(loaded_tensor.shape, expected_shape)

        # Check content
        # loaded_tensor is (H, W, 3, T)
        # self.generated_images[i] is (H, W, 3)
        for i in range(self.num_images):
            original = self.generated_images[i]
            loaded = loaded_tensor[..., i]
            np.testing.assert_allclose(loaded, original, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
