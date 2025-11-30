"""
Unit tests for Q-DIC compression pipeline
"""

import pytest
import numpy as np
from qdic import QDICEncoder, QDICDecoder
from qdic.compression.evaluation import compute_ssim, compute_psnr

class TestQDICPipeline:
    """Test complete compression-decompression pipeline"""
    
    def test_basic_compression(self):
        """Test basic compression on small image"""
        # Create simple 32x32 test image
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        # Compress
        encoder = QDICEncoder(variant="nisq", n_clusters=10)
        compressed = encoder.compress(img)
        
        # Verify compressed data structure
        assert isinstance(compressed.dna_sequence, str)
        assert len(compressed.dna_sequence) > 0
        assert isinstance(compressed.codon_dictionary, dict)
        assert compressed.ratio > 0
        
    def test_decompression(self):
        """Test decompression produces correct shape"""
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        encoder = QDICEncoder(variant="nisq", n_clusters=10)
        compressed = encoder.compress(img)
        
        decoder = QDICDecoder()
        reconstructed = decoder.decompress(compressed)
        
        assert reconstructed.shape == img.shape
        assert reconstructed.dtype == np.uint8
        
    def test_quality_metrics(self):
        """Test quality metrics are computed correctly"""
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        encoder = QDICEncoder(variant="nisq", n_clusters=10)
        compressed = encoder.compress(img)
        
        decoder = QDICDecoder()
        reconstructed = decoder.decompress(compressed)
        
        ssim = compute_ssim(img, reconstructed)
        psnr = compute_psnr(img, reconstructed)
        
        assert 0 <= ssim <= 1
        assert psnr > 0
        
    def test_gradient_image(self):
        """Test on gradient image (simple pattern)"""
        img = np.zeros((64, 64), dtype=np.uint8)
        for i in range(64):
            img[i, :] = i * 4
        
        encoder = QDICEncoder(variant="nisq", n_clusters=20)
        compressed = encoder.compress(img)
        
        decoder = QDICDecoder()
        reconstructed = decoder.decompress(compressed)
        
        # Should achieve high quality on gradient
        ssim = compute_ssim(img, reconstructed)
        assert ssim > 0.8
        
    def test_compression_ratio(self):
        """Test compression ratio is reasonable"""
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        
        encoder = QDICEncoder(variant="nisq", n_clusters=50)
        compressed = encoder.compress(img)
        
        # Should achieve some compression
        assert compressed.ratio > 1.0
        
    def test_nisq_variant(self):
        """Test NISQ variant"""
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        encoder = QDICEncoder(variant="nisq")
        compressed = encoder.compress(img)
        
        assert compressed.metadata['variant'] == 'nisq'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
