#!/usr/bin/env python3
"""
Generate masks for images where mask=1 for non-black pixels and mask=0 for black pixels.

Usage:
    python generate_masks.py --input_dir data/DFC2019_processed/JAX_264/outputs_skew/images \
                             --output_dir data/DFC2019_processed/JAX_264/outputs_skew/masks

Arguments:
    --input_dir: Directory containing input PNG images
    --output_dir: Directory to save generated masks
    --save_npy: Save masks in .npy format (default: True)
    --save_png: Save masks in .png format (default: True)
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def generate_mask(image_array):
    """
    Generate a binary mask where 1 = non-black pixel, 0 = black pixel (0,0,0).
    
    Args:
        image_array: numpy array of shape (H, W, C) where C is 3 or 4
    
    Returns:
        Binary mask of shape (H, W) with dtype uint8
    """
    if image_array.shape[2] == 4:  # RGBA
        # Use RGB channels only
        rgb = image_array[:, :, :3]
    else:  # RGB
        rgb = image_array
    
    # Check if any RGB channel is non-zero
    # Pixel is non-black if R > 0 OR G > 0 OR B > 0
    mask = np.any(rgb > 0, axis=2).astype(np.uint8)
    
    return mask


def save_mask_npy(mask, output_path):
    """Save mask in .npy format."""
    np.save(output_path, mask)


def save_mask_png(mask, output_path):
    """Save mask in .png format (0=black, 255=white)."""
    # Convert binary mask to 0-255 range for visualization
    mask_img = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(output_path)


def process_image(image_path, output_dir, save_npy=True, save_png=True):
    """
    Process a single image and generate its mask.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save masks
        save_npy: Whether to save .npy format
        save_png: Whether to save .png format
    
    Returns:
        Tuple of (num_black_pixels, num_non_black_pixels)
    """
    # Read image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Generate mask
    mask = generate_mask(img_array)
    
    # Get base name without extension
    base_name = Path(image_path).stem
    
    # Save mask in requested formats
    if save_npy:
        npy_path = os.path.join(output_dir, f"{base_name}.npy")
        save_mask_npy(mask, npy_path)
    
    if save_png:
        png_path = os.path.join(output_dir, f"{base_name}.png")
        save_mask_png(mask, png_path)
    
    # Calculate statistics
    num_non_black = np.sum(mask)
    num_black = mask.size - num_non_black
    
    return num_black, num_non_black


def main():
    parser = argparse.ArgumentParser(
        description="Generate masks for non-black pixels in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_masks.py --input_dir ./images --output_dir ./masks
  
  # Save only PNG format
  python generate_masks.py --input_dir ./images --output_dir ./masks --no_npy
  
  # Process specific scene
  python generate_masks.py \\
      --input_dir data/DFC2019_processed/JAX_264/outputs_skew/images \\
      --output_dir data/DFC2019_processed/JAX_264/outputs_skew/masks
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input PNG images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save generated masks'
    )
    
    parser.add_argument(
        '--no_npy',
        action='store_true',
        help='Do not save masks in .npy format'
    )
    
    parser.add_argument(
        '--no_png',
        action='store_true',
        help='Do not save masks in .png format'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.png',
        help='File pattern to match (default: *.png)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.no_npy and args.no_png:
        print("Error: At least one output format (npy or png) must be enabled")
        return 1
    
    # Check input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG images
    image_files = sorted(list(input_dir.glob(args.pattern)))
    
    if not image_files:
        print(f"Warning: No images found matching pattern '{args.pattern}' in {input_dir}")
        return 1
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output formats: {', '.join([f for f, enabled in [('npy', not args.no_npy), ('png', not args.no_png)] if enabled])}")
    print()
    
    # Process images
    total_black = 0
    total_non_black = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            num_black, num_non_black = process_image(
                image_path,
                output_dir,
                save_npy=not args.no_npy,
                save_png=not args.no_png
            )
            total_black += num_black
            total_non_black += num_non_black
            
        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            continue
    
    # Print summary
    print()
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total pixels: {total_black + total_non_black:,}")
    print(f"Black pixels: {total_black:,} ({100*total_black/(total_black+total_non_black):.2f}%)")
    print(f"Non-black pixels: {total_non_black:,} ({100*total_non_black/(total_black+total_non_black):.2f}%)")
    print(f"\nMasks saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())