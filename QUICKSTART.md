# Q-DIC Quickstart Guide

Get started quickly with Q-DIC.

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/qdic-compression.git
cd qdic-compression

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Q-DIC package
pip install -e .
```

## Basic Usage

### 1. Compress an Image

```python
import numpy as np
from PIL import Image
from qdic import QDICEncoder

# Load image
img = np.array(Image.open("data/test_images/lena.png").convert("L"))

# Compress (NISQ variant - works on current quantum hardware)
encoder = QDICEncoder(variant="nisq", n_clusters=200)
compressed = encoder.compress(img)

print(f"Compression ratio: {compressed.ratio:.2f}×")
print(f"DNA sequence: {len(compressed.dna_sequence)} bases")
```

### 2. Decompress

```python
from qdic import QDICDecoder

decoder = QDICDecoder()
reconstructed = decoder.decompress(compressed)
```

### 3. Evaluate Quality

```python
from qdic.compression.evaluation import compute_ssim, compute_psnr

ssim = compute_ssim(img, reconstructed)
psnr = compute_psnr(img, reconstructed)

print(f"SSIM: {ssim:.3f} (1.0 = perfect)")
print(f"PSNR: {psnr:.2f} dB (higher = better)")
```

## Example Results

| Image Type | Compression Ratio | SSIM | PSNR (dB) |
|------------|-------------------|------|-----------|
| Medical CT | 10.8× | 0.946 | 41.2 |
| Natural    | 8.7× | 0.923 | 38.4 |
| Satellite  | 9.4× | 0.921 | 38.9 |
| Document   | 7.1× | 0.894 | 35.6 |

## Run Experiments

```bash
# Quick test (3 images, ~5 minutes)
python scripts/run_experiments.py --quick

# Full benchmark (15 images, ~6 hours)
python scripts/run_experiments.py --all

# Specific images
python scripts/run_experiments.py --images lena ct_scan
```

## Jupyter Notebooks

Interactive tutorials:

```bash
jupyter lab notebooks/
```

Start with `01_quickstart.ipynb` for a complete walkthrough.

## Use Real Quantum Hardware

```python
from qiskit import IBMQ
from qdic import QDICEncoder

# Setup IBM Quantum account
IBMQ.save_account("YOUR_API_TOKEN")  # Get from quantum-computing.ibm.com
IBMQ.load_account()

# Select backend
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibm_washington')  # 127 qubits

# Run on real quantum computer
encoder = QDICEncoder(variant="nisq", backend=backend)
compressed = encoder.compress(img)

print(f"Quantum hardware result: {compressed.ratio:.2f}×")
```

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'qiskit'`  
**Solution:** Activate virtual environment and reinstall: `pip install -r requirements.txt`

**Problem:** Low compression ratio (<3×)  
**Solution:** Increase `n_clusters` parameter (try 200-500)

**Problem:** Poor SSIM (<0.8)  
**Solution:** Use NISQ variant for better quality-compression trade-off

## Next Steps

- Read [API.md](docs/API.md) for complete API reference
- See [TUTORIAL.md](docs/TUTORIAL.md) for algorithm deep-dive
- Run `notebooks/07_reproduce_paper.ipynb` to reproduce all paper results
- Contribute: see GitHub issues

## Citation

```bibtex
@article{lee2025qdic,
  title={Quantum-Enhanced DNA Image Compression},
  author={Lee, Yong-Hwan and Lee, Wan-Bum},
  journal={Applied Sciences},
  year={2025}
}
```

---

**Any Questions?** Open an issue or email: hwany1458@empal.com
