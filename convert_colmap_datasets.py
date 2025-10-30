"""Convert COLMAP sparse reconstructions to NeRF-style transforms JSON.

Usage:
    python convert_colmap_datasets.py -s /path/to/colmap_output [--skip_calibration]

Arguments:
    -s  Path to a COLMAP dataset directory that contains 'sparse/0' (required).
    --skip_calibration  Skip the camera rotation/centering calibration step.

Example:
    python convert_colmap_datasets.py -s ./JAX_004_colmap

This script reads camera data from the COLMAP sparse model under
`<dataset>/sparse/0` and writes `transforms_train.json` and
`transforms_test.json` into the same dataset directory. It also copies
or converts the points3D file to the output directory.
"""

import os
import argparse
import json
import numpy as np
import shutil
import subprocess

from preprocess_sfm.colmap_sfm_utils import extract_camera_dict

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def convert_bin_to_txt_with_colmap(sparse_dir):
    """
    Use COLMAP binary to convert .bin files to .txt files.
    
    Args:
        sparse_dir: Path to the sparse reconstruction directory containing .bin files
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        print(f"Converting binary model to text format using COLMAP...")
        
        # Run COLMAP model_converter
        cmd = [
            "colmap", "model_converter",
            "--input_path", sparse_dir,
            "--output_path", sparse_dir,
            "--output_type", "TXT"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✓ Successfully converted binary model to text format")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: COLMAP model conversion failed")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: COLMAP binary not found. Please ensure COLMAP is installed and in your PATH")
        return False

def copy_or_convert_points3D(sparse_dir, output_dir):
    """
    Copy or convert points3D file from sparse reconstruction.
    Tries to find points3D.txt first, then points3D.bin.
    If only .bin exists, uses COLMAP to convert it to .txt format.
    """
    points3d_txt = os.path.join(sparse_dir, "points3D.txt")
    points3d_bin = os.path.join(sparse_dir, "points3D.bin")
    output_txt = os.path.join(output_dir, "points3D.txt")
    
    # Case 1: points3D.txt exists - just copy it
    if os.path.exists(points3d_txt):
        print(f"Found points3D.txt, copying to output directory...")
        shutil.copy2(points3d_txt, output_txt)
        print(f"✓ Copied points3D.txt")
        return True
    
    # Case 2: Only points3D.bin exists - convert using COLMAP then copy
    elif os.path.exists(points3d_bin):
        print(f"Found points3D.bin, converting to text format...")
        
        # Convert using COLMAP binary
        if convert_bin_to_txt_with_colmap(sparse_dir):
            # After conversion, points3D.txt should exist
            if os.path.exists(points3d_txt):
                shutil.copy2(points3d_txt, output_txt)
                print(f"✓ Copied converted points3D.txt")
                return True
            else:
                print("Warning: Conversion succeeded but points3D.txt not found")
                return False
        else:
            print("Warning: Failed to convert points3D.bin")
            return False
    
    # Case 3: No points3D file found
    else:
        print(f"Warning: No points3D.txt or points3D.bin found in {sparse_dir}")
        print("Creating empty points3D.txt file...")
        with open(output_txt, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0\n")
        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, required=True, help='Path to COLMAP dataset directory')
    parser.add_argument('--skip_calibration', action='store_true', help='Skip calibration step')
    args = parser.parse_args()

    sparse_dir = os.path.join(args.s, 'sparse', '0')
    
    # Check if sparse directory exists
    if not os.path.exists(sparse_dir):
        print(f"Error: Sparse reconstruction directory not found: {sparse_dir}")
        exit(1)
    
    # Extract camera dictionary
    print(f"Reading camera data from {sparse_dir}...")
    camera_dict = extract_camera_dict(sparse_dir, ext='.bin')
    
    if not camera_dict:
        print("Error: No cameras found in the sparse reconstruction")
        exit(1)
    
    num_image = len(camera_dict)
    num_image = min(num_image, 60)
    num_train = int(num_image)
    train_split = []
    test_split = []
    all_split = []
    camera_dict_keys = sorted(list(camera_dict.keys()))
    for i in range(num_image):
        if True:
            train_split.append(camera_dict_keys[i])
        else:
            # Disable test split for now
            test_split.append(camera_dict_keys[i])
        all_split.append(camera_dict_keys[i])

    output_dict_train = {
        "camera_model": "PINHOLE",
        "frames": [],
    }
    print(f"Number of training images: {len(train_split)}")
    print(f"Number of test images: {len(test_split)}")

    up = np.zeros((3,))
    for name in all_split:
        W2C = np.array(camera_dict[name]['W2C'], dtype=np.float64).reshape((4, 4))
        c2w = np.linalg.inv(W2C)
        up += c2w[0:3,1]

    up /= len(all_split)
    up = up / np.linalg.norm(up)
    up = -up  # Convert from down to up
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for name in train_split:
        K = np.array(camera_dict[name]['K'], dtype=np.float64).reshape((4, 4))
        W2C = np.array(camera_dict[name]['W2C'], dtype=np.float64).reshape((4, 4))
        c2w = np.linalg.inv(W2C)
        frame_dict = {
            "file_path": "./images/" + name,
            "fl_x": K[0, 0],
            "fl_y": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "transform_matrix": c2w,
        }
        output_dict_train["frames"].append(frame_dict)

    if not args.skip_calibration:
        print("\nRotating all cameras to look at the same point...")
        for frame_dict in output_dict_train["frames"]:
            frame_dict["transform_matrix_rotated"] = np.matmul(R, frame_dict["transform_matrix"])
        
        # find a central point they are all looking at
        print("Computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in output_dict_train["frames"]:
            mf = f["transform_matrix_rotated"][0:3,:]
            for g in output_dict_train["frames"]:
                mg = g["transform_matrix_rotated"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(f"Center of attention: {totp}")
        for f in output_dict_train["frames"]:
            f["transform_matrix_rotated"][0:3,3] -= totp

        for f in output_dict_train["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()
            f["transform_matrix_rotated"] = f["transform_matrix_rotated"].tolist()
        
        output_dict_train["R"] = R.tolist()
        output_dict_train["T"] = totp.tolist()
    else:
        for f in output_dict_train["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()

    print("\nWriting transforms_train.json...")
    with open(os.path.join(args.s, "transforms_train.json"), 'w') as f:
        json.dump(output_dict_train, f, indent=2, sort_keys=True)

    output_dict_test = {
        "camera_model": "PINHOLE",
        "frames": [],
    }
    for name in test_split:
        K = np.array(camera_dict[name]['K'], dtype=np.float64).reshape((4, 4))
        frame_dict = {
            "file_path": "./images/" + name,
            "fl_x": K[0, 0],
            "fl_y": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
        }
        w2c = np.array(camera_dict[name]['W2C'], dtype=np.float64).reshape((4, 4))
        c2w = np.linalg.inv(w2c)

        frame_dict["transform_matrix"] = c2w.tolist()
        if not args.skip_calibration:
            c2w_rotated = np.matmul(R, c2w)
            c2w_rotated[0:3,3] -= totp
            frame_dict["transform_matrix_rotated"] = c2w_rotated.tolist()
        output_dict_test["frames"].append(frame_dict)

    if not args.skip_calibration:
        output_dict_test["R"] = R.tolist()
        output_dict_test["T"] = totp.tolist()
    
    print("Writing transforms_test.json...")
    with open(os.path.join(args.s, "transforms_test.json"), 'w') as f:
        json.dump(output_dict_test, f, indent=2, sort_keys=True)
    
    # Copy or convert points3D file
    print("\nProcessing 3D points...")
    success = copy_or_convert_points3D(sparse_dir, args.s)
    
    if success:
        print("\n" + "="*60)
        print("✓ Conversion complete!")
        print("="*60)
        print(f"Output directory: {args.s}")
        print(f"  - transforms_train.json ({len(train_split)} images)")
        print(f"  - transforms_test.json ({len(test_split)} images)")
        print(f"  - points3D.txt")
        print("\nYour dataset is ready for Skyfall-GS!")
        print("Next step: https://github.com/jayin92/Skyfall-GS")
    else:
        print("\n⚠ Conversion completed with warnings")
        print("Please check the output above for details")