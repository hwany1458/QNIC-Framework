"""
Generate test images for Q-DIC experiments

Creates 15 test images across different categories:
- Medical: CT scans, MRI
- Natural: Lena, landscape
- Satellite: aerial imagery
- Documents: text, diagrams
"""

import numpy as np
from PIL import Image
import os

def create_test_images(output_dir="../data/test_images"):
    """Generate 15 test images"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1-2: Medical CT/MRI (high contrast, structured)
    print("Creating medical images...")
    ct_scan = create_ct_scan(256, 256)
    Image.fromarray(ct_scan).save(f"{output_dir}/ct_scan.png")
    
    mri_brain = create_mri_brain(256, 256)
    Image.fromarray(mri_brain).save(f"{output_dir}/mri_brain.png")
    
    # 3-5: Natural images (Lena, landscape, texture)
    print("Creating natural images...")
    lena = create_lena_substitute(512, 512)
    Image.fromarray(lena).save(f"{output_dir}/lena.png")
    
    landscape = create_landscape(512, 512)
    Image.fromarray(landscape).save(f"{output_dir}/landscape.png")
    
    texture = create_texture(256, 256)
    Image.fromarray(texture).save(f"{output_dir}/texture.png")
    
    # 6-7: Satellite imagery
    print("Creating satellite images...")
    satellite1 = create_satellite(512, 512)
    Image.fromarray(satellite1).save(f"{output_dir}/satellite_urban.png")
    
    satellite2 = create_satellite_agriculture(512, 512)
    Image.fromarray(satellite2).save(f"{output_dir}/satellite_agriculture.png")
    
    # 8-10: Documents
    print("Creating document images...")
    document1 = create_document_text(512, 512)
    Image.fromarray(document1).save(f"{output_dir}/document_text.png")
    
    diagram = create_diagram(512, 512)
    Image.fromarray(diagram).save(f"{output_dir}/diagram.png")
    
    # 11-15: Additional test cases
    print("Creating additional test images...")
    gradient = create_gradient(256, 256)
    Image.fromarray(gradient).save(f"{output_dir}/gradient.png")
    
    checkerboard = create_checkerboard(256, 256)
    Image.fromarray(checkerboard).save(f"{output_dir}/checkerboard.png")
    
    noise = create_noise_pattern(256, 256)
    Image.fromarray(noise).save(f"{output_dir}/noise.png")
    
    circle_pattern = create_circles(256, 256)
    Image.fromarray(circle_pattern).save(f"{output_dir}/circles.png")
    
    complex_pattern = create_complex_pattern(512, 512)
    Image.fromarray(complex_pattern).save(f"{output_dir}/complex.png")
    
    print(f"✓ Created 15 test images in {output_dir}/")

def create_ct_scan(h, w):
    """Simulate CT scan with bone structures"""
    img = np.ones((h, w), dtype=np.uint8) * 50  # Dark background
    
    # Add circular structures (skull)
    center = (h // 2, w // 2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    
    # Outer skull
    mask_outer = (dist < h // 2.2) & (dist > h // 2.5)
    img[mask_outer] = 200
    
    # Inner structures
    mask_inner = dist < h // 3
    img[mask_inner] = 80
    
    # Add some variation
    img = img + np.random.normal(0, 10, (h, w)).astype(np.uint8)
    return np.clip(img, 0, 255)

def create_mri_brain(h, w):
    """Simulate MRI brain scan"""
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Brain ellipse
    Y, X = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    ellipse = ((X - center[1]) / (w // 2.5))**2 + ((Y - center[0]) / (h // 2))**2 < 1
    img[ellipse] = 120
    
    # Add internal structures
    for i in range(5):
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        r = np.random.randint(10, 30)
        structure = ((X - cx)**2 + (Y - cy)**2) < r**2
        img[structure] = np.random.randint(80, 180)
    
    return img

def create_lena_substitute(h, w):
    """Create Lena-like image (portrait substitute)"""
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Face ellipse
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 3
    face = ((X - cx) / (w // 4))**2 + ((Y - cy) / (h // 3))**2 < 1
    img[face] = 180
    
    # Hair
    hair = (Y < h // 4) | ((Y < h // 2) & ((X < w // 3) | (X > 2 * w // 3)))
    img[hair] = 60
    
    # Add texture
    img = img + np.random.normal(0, 15, (h, w)).astype(np.uint8)
    return np.clip(img, 0, 255)

def create_landscape(h, w):
    """Create landscape with sky, mountains, ground"""
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Sky gradient
    for i in range(h // 2):
        img[i, :] = 200 - i // 2
    
    # Mountains
    mountain_y = h // 2 + 20 * np.sin(np.linspace(0, 4 * np.pi, w))
    for x in range(w):
        img[int(mountain_y[x]):, x] = 100
    
    # Ground
    img[2 * h // 3:, :] = 80
    
    return img

def create_texture(h, w):
    """Create texture pattern"""
    return (np.random.rand(h, w) * 255).astype(np.uint8)

def create_satellite(h, w):
    """Satellite imagery (urban)"""
    img = np.ones((h, w), dtype=np.uint8) * 100
    
    # Grid pattern (roads)
    for i in range(0, h, 50):
        img[i:i+3, :] = 200
    for j in range(0, w, 50):
        img[:, j:j+3] = 200
    
    # Buildings
    for _ in range(20):
        x, y = np.random.randint(0, w-30), np.random.randint(0, h-30)
        size = np.random.randint(20, 40)
        img[y:y+size, x:x+size] = np.random.randint(60, 140)
    
    return img

def create_satellite_agriculture(h, w):
    """Satellite imagery (agriculture fields)"""
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Create field patches
    for i in range(5):
        for j in range(5):
            y_start, x_start = i * h // 5, j * w // 5
            y_end, x_end = (i + 1) * h // 5, (j + 1) * w // 5
            img[y_start:y_end, x_start:x_end] = np.random.randint(80, 180)
    
    return img

def create_document_text(h, w):
    """Document with text lines"""
    img = np.ones((h, w), dtype=np.uint8) * 255
    
    # Horizontal text lines
    for i in range(20, h-20, 30):
        img[i:i+2, 40:w-40] = 0
    
    return img

def create_diagram(h, w):
    """Technical diagram"""
    img = np.ones((h, w), dtype=np.uint8) * 255
    
    # Draw boxes and arrows
    img[50:150, 50:150] = 0
    img[52:148, 52:148] = 255
    
    img[50:150, w-150:w-50] = 0
    img[52:148, w-148:w-52] = 255
    
    # Arrow
    img[98:102, 150:w-150] = 0
    
    return img

def create_gradient(h, w):
    """Simple gradient"""
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        img[i, :] = int(255 * i / h)
    return img

def create_checkerboard(h, w, square_size=32):
    """Checkerboard pattern"""
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = 255
    return img

def create_noise_pattern(h, w):
    """Random noise"""
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)

def create_circles(h, w):
    """Concentric circles"""
    img = np.zeros((h, w), dtype=np.uint8)
    Y, X = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    
    for r in range(20, min(h, w) // 2, 20):
        mask = (dist > r - 2) & (dist < r + 2)
        img[mask] = 255
    
    return img

def create_complex_pattern(h, w):
    """Complex mixed pattern"""
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Combine multiple frequencies
    for freq in [1, 2, 4, 8]:
        X, Y = np.meshgrid(np.linspace(0, freq * np.pi, w), 
                          np.linspace(0, freq * np.pi, h))
        img += (50 * np.sin(X) * np.cos(Y)).astype(np.uint8)
    
    return np.clip(img, 0, 255)

if __name__ == "__main__":
    create_test_images()
