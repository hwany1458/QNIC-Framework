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
        "qiskit>=2.2.3",
        "qiskit-aer>=0.17.2",
        "numpy>=1.26.4",
        "scipy>=1.26.4",
        "pandas>=2.2.3",
        "Pillow>=11.0.0",
        "scikit-image>=0.24.0",
        "scikit-learn>=1.5.2",
        "matplotlib>=3.10.3",
        "tqdm>=4.66.5",
    ],
)
