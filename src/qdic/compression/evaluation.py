"""
Evaluation metrics: SSIM, PSNR, compression ratio
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM)
    
    Parameters
    ----------
    original : np.ndarray
        Original image
    reconstructed : np.ndarray
        Reconstructed image
        
    Returns
    -------
    float
        SSIM value in [0, 1], higher is better
    """
    return ssim(original, reconstructed, data_range=255)

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    
    Parameters
    ----------
    original : np.ndarray
        Original image
    reconstructed : np.ndarray
        Reconstructed image
        
    Returns
    -------
    float
        PSNR value in dB, higher is better
    """
    return psnr(original, reconstructed, data_range=255)

def compute_compression_ratio(original: np.ndarray, compressed_dna: str) -> float:
    """
    Compute compression ratio
    
    Parameters
    ----------
    original : np.ndarray
        Original image
    compressed_dna : str
        DNA sequence
        
    Returns
    -------
    float
        Compression ratio (original_size / compressed_size)
    """
    original_bits = original.size * 8  # 8 bits per pixel
    dna_bits = len(compressed_dna) * 2  # 2 bits per DNA base
    return original_bits / dna_bits

def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Mean Squared Error"""
    return np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
