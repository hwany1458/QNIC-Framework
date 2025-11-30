"""
Q-DIC: Quantum-Enhanced DNA Image Compression

A comprehensive framework integrating quantum optimization with DNA-based 
molecular storage for enhanced image compression.

Authors: Yong-Hwan Lee, Wan-Bum Lee
Institution: Wonkwang University
Year: 2025
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Yong-Hwan Lee, Wan-Bum Lee"
__email__ = "hwany1458@empal.com"

from .compression.encoder import QDICEncoder
from .compression.decoder import QDICDecoder
from .compression.evaluation import (
    compute_ssim,
    compute_psnr,
    compute_compression_ratio
)

__all__ = [
    'QDICEncoder',
    'QDICDecoder',
    'compute_ssim',
    'compute_psnr',
    'compute_compression_ratio',
]
