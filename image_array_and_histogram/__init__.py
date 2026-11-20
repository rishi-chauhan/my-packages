"""image_array_and_histogram public API.

This version modernizes the API with PEP-8 function names while keeping
backward compatible deprecated camelCase aliases.
"""

from .main import (
	array_to_image,
	get_hist,
	get_image_array,
	getHist,
	# Deprecated aliases
	getImageArray,
	getImageFromArray,
)

__all__ = [
	"array_to_image",
	"getHist",
	"getImageArray",
	"getImageFromArray",
	"get_hist",
	"get_image_array",
]

__author__ = 'Rishi Raj Singh Chauhan'
__version__ = '1.1.9'