"""
Q-DIC Decoder: Decompression and reconstruction
"""

import numpy as np
from typing import Dict
from .encoder import CompressedData

class QDICDecoder:
    """
    Q-DIC Decoder for image reconstruction
    
    Reverses the compression process:
    1. Error correction decoding
    2. DNA to pixel mapping
    3. Image reconstruction
    """
    
    def __init__(self):
        pass
        
    def decompress(self, compressed: CompressedData) -> np.ndarray:
        """
        Decompress DNA sequence back to image
        
        Parameters
        ----------
        compressed : CompressedData
            Compressed data from Q-DIC encoder
            
        Returns
        -------
        np.ndarray
            Reconstructed grayscale image
        """
        print("[Q-DIC Decoder] Starting decompression...")
        
        # Step 1: Remove error correction
        print("[1/3] Removing error correction...")
        dna_sequence = self._remove_error_correction(compressed.dna_sequence)
        
        # Step 2: DNA to pixel mapping
        print("[2/3] Decoding DNA to pixels...")
        pixels = self._decode_from_dna(
            dna_sequence, 
            compressed.codon_dictionary,
            compressed.metadata['original_pixels']
        )
        
        # Step 3: Reconstruct image
        print("[3/3] Reconstructing image...")
        shape = compressed.metadata['shape']
        image = pixels.reshape(shape)
        
        print("✓ Decompression complete!")
        
        return image
        
    def _remove_error_correction(self, protected_sequence: str) -> str:
        """Remove error correction parity bases"""
        # Remove parity bases (every 9th base in simplified scheme)
        original = []
        for i in range(0, len(protected_sequence), 9):
            block = protected_sequence[i:i+8]  # Get 8-base block
            original.append(block)
        return ''.join(original)
        
    def _decode_from_dna(self, dna_sequence: str, codon_dict: Dict[int, str], 
                         n_pixels: int) -> np.ndarray:
        """Decode DNA sequence to pixel values using codon dictionary"""
        # Create reverse mapping: codon -> pixel value
        reverse_dict = {codon: pixel for pixel, codon in codon_dict.items()}
        
        # Decode each codon
        pixels = []
        for i in range(0, len(dna_sequence), 8):
            codon = dna_sequence[i:i+8]
            if codon in reverse_dict:
                pixels.append(reverse_dict[codon])
            else:
                # Find nearest codon if exact match not found
                nearest = min(reverse_dict.keys(), 
                             key=lambda x: self._hamming_distance(x, codon))
                pixels.append(reverse_dict[nearest])
                
        return np.array(pixels[:n_pixels], dtype=np.uint8)
        
    def _hamming_distance(self, s1: str, s2: str) -> int:
        """Calculate Hamming distance between two strings"""
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
