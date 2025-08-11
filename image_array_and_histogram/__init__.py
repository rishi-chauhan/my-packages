"""image_array_and_histogram public API.

This version modernizes the API with PEP-8 function names while keeping
backward compatible deprecated camelCase aliases.
"""

from .main import (
	get_image_array,
	get_hist,
	array_to_image,
	# Deprecated aliases
	getImageArray,
	getHist,
	getImageFromArray,
)

__all__ = [
	"get_image_array",
	"get_hist",
	"array_to_image",
	"getImageArray",
	"getHist",
	"getImageFromArray",
]

__author__ = 'Rishi Raj Singh Chauhan'
__version__ = '1.1.1'