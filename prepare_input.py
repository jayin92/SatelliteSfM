#!/usr/bin/env python3
"""
Organize DFC2019 satellite imagery by scene ID.

This script creates a scene-specific folder structure with images and bounding box metadata.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Optional
import glob


def find_scene_images(track_rgb_dir: Path, scene_id: str, city: str = "JAX") -> List[Path]:
    """
    Find all TIF images matching the scene ID.
    
    Args:
        track_rgb_dir: Path to Track3-RGB directory
        scene_id: Scene ID (e.g., "004")
        city: City code (JAX or OMA)
    
    Returns:
        List of paths to matching TIF files
    """
    pattern = f"{city}_{scene_id}_*_RGB.tif"
    image_files = sorted(track_rgb_dir.glob(pattern))
    
    if not image_files:
        print(f"Warning: No images found matching pattern: {pattern}")
    
    return image_files


def find_scene_bbx(preprocessed_dir: Path, scene_id: str, city: str = "JAX") -> Optional[Path]:
    """
    Find a bounding box JSON file matching the scene ID.
    
    Args:
        preprocessed_dir: Path to preprocessed directory
        scene_id: Scene ID (e.g., "004")
        city: City code (JAX or OMA)
    
    Returns:
        Path to a matching JSON file (returns first match since they're all the same)
    """
    bbx_dir = preprocessed_dir / "latlonalt_bbx"
    pattern = f"{city}_{scene_id}_*_RGB.json"
    bbx_files = sorted(bbx_dir.glob(pattern))
    
    if not bbx_files:
        print(f"Warning: No bounding box files found matching pattern: {pattern}")
        return None
    
    # Return first file since they're all the same
    return bbx_files[0]


def organize_scene(
    scene_id: str,
    track_rgb_dir: Path,
    preprocessed_dir: Path,
    output_base_dir: Path,
    city: str = "JAX",
    copy_images: bool = True
):
    """
    Organize a scene into the target folder structure.
    
    Args:
        scene_id: Scene ID (e.g., "004")
        track_rgb_dir: Path to Track3-RGB directory with original TIF files
        preprocessed_dir: Path to preprocessed directory with bounding boxes
        output_base_dir: Base output directory
        city: City code (JAX or OMA)
        copy_images: If True, copy images; if False, create symlinks
    """
    # Create output directory structure
    scene_name = f"{city}_{scene_id}"
    output_dir = output_base_dir / scene_name / "inputs"
    images_dir = output_dir / "images"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")
    
    # Find all matching images
    print(f"\nSearching for images in: {track_rgb_dir}")
    image_files = find_scene_images(track_rgb_dir, scene_id, city)
    print(f"Found {len(image_files)} images")
    
    # Copy or symlink images
    for img_file in image_files:
        dest_file = images_dir / img_file.name
        
        if copy_images:
            print(f"  Copying: {img_file.name}")
            shutil.copy2(img_file, dest_file)
        else:
            print(f"  Linking: {img_file.name}")
            if dest_file.exists():
                dest_file.unlink()
            dest_file.symlink_to(img_file.absolute())
    
    # Find and copy bounding box
    print(f"\nSearching for bounding box in: {preprocessed_dir / 'latlonalt_bbx'}")
    bbx_file = find_scene_bbx(preprocessed_dir, scene_id, city)
    
    if bbx_file:
        bbx_output_file = output_dir / "latlonalt_bbx.json"
        shutil.copy2(bbx_file, bbx_output_file)
        
        # Load and display bounding box info
        with open(bbx_file, 'r') as f:
            bbx = json.load(f)
        
        print(f"\nCopied bounding box from: {bbx_file.name}")
        print(f"  Output: {bbx_output_file}")
    else:
        print("\n⚠ No bounding box file found for this scene")
    
    print(f"\n✓ Scene {scene_name} organized successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total images: {len(image_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize DFC2019 satellite imagery by scene ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize JAX scene 004
  python organize_scene.py --scene_id 004 --city JAX
  
  # Organize OMA scene 068 with symlinks instead of copying
  python organize_scene.py --scene_id 068 --city OMA --symlink
  
  # Use custom paths
  python organize_scene.py --scene_id 004 \\
      --track_rgb_dir /path/to/Track3-RGB-1 \\
      --preprocessed_dir /path/to/DFC2019_JAX_preprocessed \\
      --output_dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--scene_id',
        type=str,
        required=True,
        help='Scene ID (e.g., "004", "068")'
    )
    
    parser.add_argument(
        '--city',
        type=str,
        default='JAX',
        choices=['JAX', 'OMA'],
        help='City code (default: JAX)'
    )
    
    parser.add_argument(
        '--track_rgb_dir',
        type=Path,
        default=Path('data/Track3-RGB-1'),
        help='Path to Track3-RGB directory (default: data/Track3-RGB-1)'
    )
    
    parser.add_argument(
        '--preprocessed_dir',
        type=Path,
        default=None,
        help='Path to preprocessed directory (default: data/DFC2019_{CITY}_preprocessed)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/DFC2019_processed'),
        help='Output base directory (default: data/DFC2019_processed)'
    )
    
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symlinks instead of copying images (saves disk space)'
    )
    
    args = parser.parse_args()
    
    # Set default preprocessed_dir based on city if not provided
    if args.preprocessed_dir is None:
        args.preprocessed_dir = Path(f'data/DFC2019_{args.city}_preprocessed')
    
    # Validate inputs
    if not args.track_rgb_dir.exists():
        print(f"Error: Track RGB directory does not exist: {args.track_rgb_dir}")
        return 1
    
    if not args.preprocessed_dir.exists():
        print(f"Error: Preprocessed directory does not exist: {args.preprocessed_dir}")
        return 1
    
    # Organize the scene
    organize_scene(
        scene_id=args.scene_id,
        track_rgb_dir=args.track_rgb_dir,
        preprocessed_dir=args.preprocessed_dir,
        output_base_dir=args.output_dir,
        city=args.city,
        copy_images=not args.symlink
    )
    
    return 0


if __name__ == '__main__':
    exit(main())