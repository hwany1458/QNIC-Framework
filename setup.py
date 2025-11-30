from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qdic",
    version="1.0.0",
    author="Yong-Hwan Lee, Wan-Bum Lee",
    author_email="hwany1458@empal.com",
    description="Quantum-Enhanced DNA Image Compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/qdic-compression",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "qiskit>=0.45.0",
        "qiskit-aer>=0.13.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "Pillow>=9.5.0",
        "scikit-image>=0.20.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
