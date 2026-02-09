# image_array_and_histogram

[![CI](https://github.com/rishi-chauhan/my-packages/actions/workflows/ci.yml/badge.svg)](https://github.com/rishi-chauhan/my-packages/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/image-array-and-histogram.svg)](https://pypi.org/project/image-array-and-histogram/)
[![Python versions](https://img.shields.io/pypi/pyversions/image-array-and-histogram.svg)](https://pypi.org/project/image-array-and-histogram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python utilities for converting images to NumPy arrays, computing grayscale histograms, and reconstructing images from arrays.

## Installation

```sh
pip install image-array-and-histogram
```

## Usage

```python
from PIL import Image
import numpy as np
from image_array_and_histogram import get_image_array, get_hist, array_to_image

# Convert image to array
img = Image.open('photo.jpg')
arr = get_image_array(img)  # shape: (height, width)

# Compute histogram
hist = get_hist(arr)  # list of 256 counts

# Get normalized histogram
hist_density = get_hist(arr, as_density=True)

# Create image from array
gradient = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
gradient_img = array_to_image(gradient)
gradient_img.save('gradient.png')
```

## API Reference

### `get_image_array(image, ensure_grayscale=True)`

Returns a 2D NumPy array of shape `(height, width)` with dtype `uint8`.

**Parameters:**
- `image` (PIL.Image.Image): Input PIL image
- `ensure_grayscale` (bool): If True, converts to grayscale mode 'L'. Default: True

### `get_hist(image_or_array, as_density=False)`

Returns a 256-element list representing the pixel intensity histogram.

**Parameters:**
- `image_or_array` (PIL.Image.Image | array-like): PIL image or 2D array of pixels
- `as_density` (bool): If True, returns normalized probabilities summing to 1.0. Default: False

### `array_to_image(arr, width=None, height=None)`

Creates a grayscale PIL image from a 1D or 2D array.

**Parameters:**
- `arr` (array-like): 1D or 2D array of pixel values (0-255)
- `width` (int, optional): Required if `arr` is 1D
- `height` (int, optional): Required if `arr` is 1D

**Note:** Values outside [0, 255] are automatically clipped with a warning.

## Migration from 1.0.x

Version 1.1+ fixes a historical bug where arrays were shaped `(width, height)`. Arrays are now correctly shaped `(height, width)` following NumPy conventions. If your code depends on the old ordering, you may need to transpose your arrays or swap width/height arguments.

The deprecated camelCase function names (`getImageArray`, `getHist`, `getImageFromArray`) still work but emit deprecation warnings.

## Development

Clone the repository and install with development dependencies:

```sh
git clone https://github.com/rishi-chauhan/my-packages.git
cd my-packages
pip install -e .[dev]
```

Run tests:

```sh
pytest
```

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.
