"""Script to download images in the Pinterest Multimodal Dataset.

See py_utils.PinDataset for advanced usage.
Code belongs to the project on
    https://github.com/mjhucla/P-Multimodal-Dataset-Toolbox
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from py_utils import PinDataset

if __name__ == '__main__':
  pin_dataset = PinDataset()
  pin_dataset.download_images()
