"""
Reproduce all experiments from the paper

Usage:
    python run_experiments.py --all                    # Run all experiments (~6 hours)
    python run_experiments.py --images lena ct_scan   # Run specific images
    python run_experiments.py --quick                  # Quick test (5 minutes)
"""

import argparse
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdic import QDICEncoder, QDICDecoder
from qdic.compression.evaluation import compute_ssim, compute_psnr
from tqdm import tqdm

def run_single_experiment(image_path, variant="nisq", n_trials=20):
    """Run compression experiment on single image"""
    
    # Load image
    img = np.array(Image.open(image_path).convert("L"))
    
    results = []
    
    for trial in range(n_trials):
        # Compress
        encoder = QDICEncoder(variant=variant, n_clusters=200)
        compressed = encoder.compress(img)
        
        # Decompress
        decoder = QDICDecoder()
        reconstructed = decoder.decompress(compressed)
        
        # Evaluate
        ssim_score = compute_ssim(img, reconstructed)
        psnr_score = compute_psnr(img, reconstructed)
        
        results.append({
            'trial': trial,
            'compression_ratio': compressed.ratio,
            'ssim': ssim_score,
            'psnr': psnr_score,
        })
    
    return pd.DataFrame(results)

def run_all_experiments(output_dir="../data/results"):
    """Run experiments on all 15 test images"""
    
    image_dir = Path("../data/test_images")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    image_files = list(image_dir.glob("*.png"))
    
    all_results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        print(f"\nProcessing {img_path.name}...")
        
        # Run NISQ variant
        df_nisq = run_single_experiment(img_path, variant="nisq", n_trials=20)
        df_nisq['image'] = img_path.stem
        df_nisq['variant'] = 'nisq'
        
        all_results.append(df_nisq)
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Save
    df_all.to_csv(output_dir / "compression_results.csv", index=False)
    
    # Print summary
    summary = df_all.groupby('image').agg({
        'compression_ratio': ['mean', 'std'],
        'ssim': ['mean', 'std'],
        'psnr': ['mean', 'std']
    })
    
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(summary)
    
    summary.to_csv(output_dir / "summary_statistics.csv")
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    return df_all

def quick_test():
    """Quick test on 3 images"""
    print("Running quick test (3 images, 5 trials each)...")
    
    image_dir = Path("../data/test_images")
    test_images = ["lena.png", "ct_scan.png", "gradient.png"]
    
    for img_name in test_images:
        img_path = image_dir / img_name
        print(f"\nTesting {img_name}...")
        
        df = run_single_experiment(img_path, variant="nisq", n_trials=5)
        
        print(f"  Avg compression ratio: {df['compression_ratio'].mean():.2f}×")
        print(f"  Avg SSIM: {df['ssim'].mean():.3f}")
        print(f"  Avg PSNR: {df['psnr'].mean():.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-DIC experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--images", nargs="+", help="Specific images to process")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.all:
        run_all_experiments()
    elif args.images:
        for img_name in args.images:
            img_path = Path(f"../data/test_images/{img_name}.png")
            if img_path.exists():
                df = run_single_experiment(img_path)
                print(f"\nResults for {img_name}:")
                print(df.describe())
    else:
        print("Please specify --all, --quick, or --images")
        parser.print_help()
