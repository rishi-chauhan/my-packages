# image_array_and_histogram

Utilities to convert images to NumPy arrays, compute grayscale histograms, and
reconstruct images from arrays.

Version 1.1.0 introduces PEP-8 function names and fixes the historical axis
ordering bug. Arrays are now always shaped `(height, width)`. The old camelCase
names are still available but deprecated.

## Installation

```sh
$ pip install image-array-and-histogram
```

## Functions (current API)

- `get_image_array(image, ensure_grayscale=True)` – Return a 2D uint8 NumPy array (height, width) from a PIL image. Converts to grayscale by default.
- `get_hist(image_or_array, as_density=False)` – Return a 256-length list of counts (or probabilities if `as_density=True`). Accepts either a PIL image or a NumPy/list array.
- `array_to_image(arr, width=None, height=None)` – Build a grayscale PIL image from a 1D or 2D array.

Deprecated aliases (will emit `DeprecationWarning`): `getImageArray`, `getHist`, `getImageFromArray`.

## Quick Start

```python
from PIL import Image
import numpy as np
from image_array_and_histogram import get_image_array, get_hist, array_to_image

# Load image and get array
img = Image.open('photo.jpg')
arr = get_image_array(img)  # shape (H, W)

# Compute histogram
hist = get_hist(arr)  # list of 256 counts

# Normalize histogram
hist_density = get_hist(arr, as_density=True)

# Create an image from a NumPy array
gradient = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
gradient_img = array_to_image(gradient)
gradient_img.save('gradient.png')
```

## Notes

- If you pass a color image to `get_image_array` or `get_hist`, it will be converted to grayscale (mode 'L').
- Histogram computation is vectorized with NumPy (`numpy.bincount`) for speed.
- For legacy behavior (<=1.0.x) the array shape used `(width, height)`. Adjust any downstream code if it relied on that ordering.

## Testing

After cloning the repository:

```sh
pip install -e .[dev]
pytest -q
```

## License

MIT
