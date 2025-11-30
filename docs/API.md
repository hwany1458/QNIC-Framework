# Q-DIC API Documentation

## Main Classes

### QDICEncoder

Main compression engine implementing quantum-DNA compression.

```python
from qdic import QDICEncoder

encoder = QDICEncoder(
    variant="nisq",              # "nisq" or "full"
    backend="qasm_simulator",    # Qiskit backend
    n_clusters=200               # Number of pixel clusters
)
```

**Methods:**

#### `compress(image: np.ndarray) -> CompressedData`

Compress grayscale image.

**Parameters:**
- `image` (np.ndarray): Grayscale image, shape (H, W), values [0, 255]

**Returns:**
- `CompressedData`: Object containing:
  - `dna_sequence` (str): DNA representation
  - `codon_dictionary` (Dict): Pixel-to-codon mapping
  - `metadata` (Dict): Image shape, variant, etc.
  - `ratio` (float): Compression ratio

**Example:**
```python
import numpy as np
from qdic import QDICEncoder

img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
encoder = QDICEncoder(variant="nisq")
compressed = encoder.compress(img)
print(f"Compression ratio: {compressed.ratio:.2f}×")
```

---

### QDICDecoder

Decompression engine.

```python
from qdic import QDICDecoder

decoder = QDICDecoder()
```

**Methods:**

#### `decompress(compressed: CompressedData) -> np.ndarray`

Decompress DNA sequence back to image.

**Parameters:**
- `compressed` (CompressedData): Data from encoder

**Returns:**
- `np.ndarray`: Reconstructed grayscale image

**Example:**
```python
from qdic import QDICEncoder, QDICDecoder

encoder = QDICEncoder()
decoder = QDICDecoder()

compressed = encoder.compress(img)
reconstructed = decoder.decompress(compressed)
```

---

## Evaluation Functions

### compute_ssim

Calculate Structural Similarity Index.

```python
from qdic.compression.evaluation import compute_ssim

ssim_score = compute_ssim(original, reconstructed)
```

**Returns:** Float in [0, 1], higher is better (1.0 = perfect)

---

### compute_psnr

Calculate Peak Signal-to-Noise Ratio.

```python
from qdic.compression.evaluation import compute_psnr

psnr_score = compute_psnr(original, reconstructed)
```

**Returns:** Float in dB, higher is better (>40 dB = excellent)

---

### compute_compression_ratio

Calculate compression ratio.

```python
from qdic.compression.evaluation import compute_compression_ratio

ratio = compute_compression_ratio(original, dna_sequence)
```

**Returns:** Float, original_size / compressed_size

---

## Advanced Usage

### Using Real IBM Quantum Hardware

```python
from qiskit import IBMQ
from qdic import QDICEncoder

# Authenticate
IBMQ.save_account("YOUR_API_TOKEN")
IBMQ.load_account()

# Get backend
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibm_washington')

# Use real quantum hardware
encoder = QDICEncoder(variant="nisq", backend=backend)
compressed = encoder.compress(img)
```

### Custom Cost Function Weights

Modify encoder internal weights (advanced):

```python
encoder = QDICEncoder(variant="nisq")

# Access internal method (for research)
# Modify lambda weights in _evaluate_codon_cost()
# Default: λ₁=0.4, λ₂=0.3, λ₃=0.2, λ₄=0.1
```

---

## CompressedData Structure

```python
@dataclass
class CompressedData:
    dna_sequence: str              # Full DNA representation
    codon_dictionary: Dict[int, str]  # Pixel -> 8-base codon
    metadata: Dict                 # Image info
    ratio: float                   # Compression ratio
```

**Example access:**
```python
compressed = encoder.compress(img)

print(f"DNA length: {len(compressed.dna_sequence)} bases")
print(f"Codons: {len(compressed.codon_dictionary)}")
print(f"Original shape: {compressed.metadata['shape']}")
print(f"Ratio: {compressed.ratio:.2f}×")
```
