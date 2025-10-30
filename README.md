# Dataset Preprocessing for Skyfall-GS

This repository is forked from [Kai-46/SatelliteSfM](https://github.com/Kai-46/SatelliteSfM). Special thanks to Kai Zhang for the original work!

This repository provides tools for preparing datasets for [Skyfall-GS](https://github.com/jayin92/Skyfall-GS) from two different sources:

1. **COLMAP reconstructions** - Convert existing COLMAP sparse reconstructions
2. **Satellite imagery** - Process satellite images with RPC camera models

---

## COLMAP Dataset Conversion

If you already have reconstruction results from [COLMAP](https://github.com/colmap/colmap), you can convert them using the provided script.

### Usage
```bash
python convert_colmap_datasets.py -s /path/to/colmap_output [--skip_calibration]
```

### What it does

This script:
- Reads camera data from the COLMAP sparse model located at `<dataset>/sparse/0`
- Generates `transforms_train.json` and `transforms_test.json` in the dataset directory
- Automatically rotates and centers cameras so that:
  - The look-at point is at the scene origin
  - The up vector is `(0, 0, 1)`

> **✓ Dataset Ready!** After conversion, your dataset is ready for training with [Skyfall-GS](https://github.com/jayin92/Skyfall-GS). Head over to the Skyfall-GS repository to train 3DGS models on your prepared dataset.

---

## Satellite Imagery Processing

### What is RPC (Rational Polynomial Camera) Model?

The RPC (Rational Polynomial Camera) model is a standard camera model used in satellite imagery that describes the relationship between 3D world coordinates and 2D image coordinates using rational polynomial functions.

#### Overview

Unlike traditional pinhole camera models used in computer vision, satellite imagery requires a different approach due to:
- **Orbital motion** - The satellite is moving during image capture
- **Earth curvature** - Large ground coverage areas where Earth's curvature matters
- **Atmospheric effects** - Refraction and distortion through the atmosphere
- **Complex optics** - Push-broom sensors and other specialized imaging systems

The RPC model provides a **generalized, vendor-agnostic** representation that abstracts away these complexities into a mathematical model.

#### Mathematical Formulation

The RPC model uses rational polynomials to map between:
- **(Lat, Lon, Alt)** - 3D world coordinates (latitude, longitude, altitude)
- **(Row, Col)** - 2D image pixel coordinates

The transformation is expressed as:
```
Row = P1(X,Y,Z) / P2(X,Y,Z)
Col = P3(X,Y,Z) / P4(X,Y,Z)
```

Where:
- **P1, P2, P3, P4** are polynomial functions (typically cubic, with up to 20 coefficients each)
- **X, Y, Z** are normalized 3D coordinates derived from Lat, Lon, Alt

For more technical details, see the [RPC specification (GeoTIFF format)](http://geotiff.maptools.org/rpc_prop.html).

---

### Installation

**Requirements:**
- Linux machine with at least one GPU
- Conda package manager

**Setup:**
```bash
. ./env.sh
```

---

### Download DFC2019 Dataset

This repository processes satellite imagery in `.tif` format with RPC camera metadata, following the format used in the public benchmark: [DFC2019 Track 3: Multi-View Semantic Stereo](https://ieee-dataport.org/open-access/data-fusion-contest-2019).

The dataset contains multi-view satellite imagery of two US cities:
- **JAX** - Jacksonville, Florida
- **OMA** - Omaha, Nebraska

**Required downloads:**

1. Track 3 / Training data / RGB images 1/2 (7.6 GB)
2. Track 3 / Training data / RGB images 2/2 (7.6 GB)
3. Track 3 / Training data / Reference (37.46 MB)

**Extract the files** into the `data/` directory with the following structure:
```
data/
├── Track3-RGB-1/*.tif
├── Track3-RGB-2/*.tif
└── Track3-Truth/[*.tif, *.txt]
```

---

### Initial Preprocessing

Run the preprocessing script to convert `.tif` images into the required format:
```bash
# Jacksonville dataset
python preprocess_track3/preprocess_track3.py \
    --base_view_dir data/Track3-RGB-1 \
    --base_dsm_dir data/Track3-Truth \
    --out_dir data/DFC2019_JAX_preprocessed

# Omaha dataset
python preprocess_track3/preprocess_track3.py \
    --base_view_dir data/Track3-RGB-2 \
    --base_dsm_dir data/Track3-Truth \
    --out_dir data/DFC2019_OMA_preprocessed
```

**Output structure:**
```
data/DFC2019_[JAX|OMA]_preprocessed/
├── cameras/          # Camera parameters
├── enu_bbx/          # ENU coordinate bounding boxes
├── enu_observers/    # Observer positions in ENU coordinates
├── groundtruth_u/    # Ground truth data
├── images/           # Converted PNG images
├── latlonalt_bbx/    # Lat/Lon/Alt bounding boxes
└── metas/            # RPC coefficients and metadata (JSON)
```

---

### Prepare Scene Inputs

Organize the preprocessed data by scene ID for SatelliteSfM processing.

#### Usage
```bash
# Basic usage - organize JAX scene 004
python prepare_input.py --scene_id 004

# Organize OMA scene 068
python prepare_input.py --scene_id 068 --city OMA

# Use symlinks to save disk space
python prepare_input.py --scene_id 004 --symlink
```

#### Advanced Options
```bash
# Custom paths
python prepare_input.py --scene_id 004 \
    --track_rgb_dir /custom/path/Track3-RGB-1 \
    --preprocessed_dir /custom/path/DFC2019_JAX_preprocessed \
    --output_dir /custom/output

# Process OMA with Track3-RGB-2
python prepare_input.py --scene_id 068 --city OMA \
    --track_rgb_dir data/Track3-RGB-2
```

#### What it does

This script:
- Finds all `.tif` images matching the scene ID pattern (e.g., `JAX_004_*_RGB.tif`)
- Copies or symlinks images to the organized structure
- Copies the bounding box JSON file for the scene
- Creates the following output structure:
```
data/DFC2019_processed/{CITY}_{SCENE_ID}/inputs/
├── images/
│   ├── {CITY}_{SCENE_ID}_*.tif
│   └── ...
└── latlonalt_bbx.json
```

---

### Run SatelliteSfM

Execute the SatelliteSfM pipeline to perform structure-from-motion reconstruction on the satellite imagery.

#### Single Scene
```bash
python satellite_sfm.py \
    --input_folder data/DFC2019_processed/JAX_004/inputs \
    --output_folder data/DFC2019_processed/JAX_004/outputs_srtm \
    --run_sfm \
    --use_srtm4 \
    --enable_debug
```

#### Parameters

- `--input_folder`: Path to the prepared scene inputs
- `--output_folder`: Path where reconstruction outputs will be saved
- `--run_sfm`: Enable structure-from-motion reconstruction
- `--use_srtm4`: Use SRTM4 (Shuttle Radar Topography Mission) elevation data for better initialization
- `--enable_debug`: Enable debug mode for additional logging and visualizations

#### Output Structure
```
data/DFC2019_processed/{CITY}_{SCENE_ID}/outputs_srtm/
├── colmap_triangulate_postba/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.txt
├── camera_poses/
├── depth_maps/
└── [other reconstruction artifacts]
```

---
### Post-Processing

After running SatelliteSfM, post-process the results to perform skew correction, convert to the final format, and generate masks for valid pixels.

#### Usage
```bash
# Process specific scenes
./postprocess_scenes.sh JAX_004 OMA_068 JAX_214

# Process a single scene
./postprocess_scenes.sh JAX_004

# Mix of JAX and OMA scenes
./postprocess_scenes.sh JAX_004 JAX_068 OMA_001 OMA_175

# Use default scenes (edit the script to customize)
./postprocess_scenes.sh
```

#### What it does

For each scene, the script performs four steps:

1. **Skew Correction** - Corrects geometric distortions using SRTM elevation data
```bash
   python skew_correct.py \
       --input_folder data/DFC2019_processed/{SCENE}/outputs_srtm \
       --output_folder data/DFC2019_processed/{SCENE}/outputs_skew
```

2. **Dataset Conversion** - Converts to the final format required by Skyfall-GS
```bash
   python convert_datasets.py \
       --input_folder data/DFC2019_processed/{SCENE}/outputs_skew
```

3. **Copy 3D Points** - Copies the reconstructed 3D point cloud
```bash
   cp data/DFC2019_processed/{SCENE}/outputs_srtm/colmap_triangulate_postba/points3D.txt \
      data/DFC2019_processed/{SCENE}/outputs_skew/
```

4. **Generate Masks** - Creates binary masks for valid (non-black) pixels
```bash
   python generate_masks.py \
       --input_dir data/DFC2019_processed/{SCENE}/outputs_skew/images \
       --output_dir data/DFC2019_processed/{SCENE}/outputs_skew/masks
```

#### Advanced Options
```bash
# Skip specific steps
./postprocess_scenes.sh --skip-skew JAX_004      # Skip skew correction
./postprocess_scenes.sh --skip-convert JAX_004   # Skip dataset conversion
./postprocess_scenes.sh --skip-copy JAX_004      # Skip points3D.txt copy
./postprocess_scenes.sh --skip-mask JAX_004      # Skip mask generation

# Dry run (see what would be executed without running)
./postprocess_scenes.sh --dry-run JAX_004

# Use custom base directory
./postprocess_scenes.sh --base-dir /custom/path JAX_004

# Get help
./postprocess_scenes.sh --help
```

Make the script executable:
```bash
chmod +x postprocess_scenes.sh
```

#### Final Output Structure
```
data/DFC2019_processed/{CITY}_{SCENE_ID}/outputs_skew/
├── cameras/                 # Camera parameters
├── images/                  # Converted PNG images
│   ├── *.png
│   └── ...
├── masks/                   # Binary masks for valid pixels
│   ├── *.npy               # NumPy format (for processing)
│   ├── *.png               # PNG format (for visualization)
│   └── ...
├── transforms_train.json    # Training camera transforms
├── transforms_test.json     # Testing camera transforms
└── points3D.txt            # 3D point cloud
```

---

## Complete Workflow Example

Here's a complete example processing JAX scene 004 from start to finish:
```bash
# 1. Download and extract DFC2019 dataset (manual step)
#    Place files in data/Track3-RGB-1, data/Track3-RGB-2, data/Track3-Truth

# 2. Initial preprocessing
python preprocess_track3/preprocess_track3.py \
    --base_view_dir data/Track3-RGB-1 \
    --base_dsm_dir data/Track3-Truth \
    --out_dir data/DFC2019_JAX_preprocessed

# 3. Prepare scene inputs
python prepare_input.py --scene_id 004 --city JAX

# 4. Run SatelliteSfM reconstruction
python satellite_sfm.py \
    --input_folder data/DFC2019_processed/JAX_004/inputs \
    --output_folder data/DFC2019_processed/JAX_004/outputs_srtm \
    --run_sfm \
    --use_srtm4 \
    --enable_debug

# 5. Post-process results
./postprocess_scenes.sh JAX_004

# 6. Final outputs ready for Skyfall-GS training
ls data/DFC2019_processed/JAX_004/outputs_skew/
```

---

## Batch Processing Multiple Scenes

Process multiple scenes efficiently:
```bash
# Define scenes to process
scenes=(JAX_004 JAX_068 JAX_214 OMA_001 OMA_175)

# Step 1: Prepare all scene inputs
for scene in "${scenes[@]}"; do
    city=$(echo $scene | cut -d'_' -f1)
    scene_id=$(echo $scene | cut -d'_' -f2)
    
    python prepare_input.py --scene_id $scene_id --city $city --symlink
done

# Step 2: Run SatelliteSfM for all scenes
for scene in "${scenes[@]}"; do
    python satellite_sfm.py \
        --input_folder data/DFC2019_processed/${scene}/inputs \
        --output_folder data/DFC2019_processed/${scene}/outputs_srtm \
        --run_sfm \
        --use_srtm4 \
        --enable_debug
done

# Step 3: Post-process all scenes at once
./postprocess_scenes.sh "${scenes[@]}"
```

---

## Troubleshooting

### Common Issues

**Issue: "No images found matching pattern"**
- Check that the scene ID format is correct (e.g., "004" not "4")
- Verify that `.tif` files exist in the Track3-RGB directory
- Ensure preprocessing was run successfully

**Issue: "Input folder does not exist"**
- Make sure you've run `prepare_input.py` before `satellite_sfm.py`
- Check that the scene ID and city code are correct

**Issue: SatelliteSfM fails with SRTM errors**
- Ensure you have internet connectivity (SRTM data is downloaded automatically)
- Try running without `--use_srtm4` flag as a fallback

**Issue: Post-processing script fails**
- Make sure the script is executable: `chmod +x postprocess_scenes.sh`
- Verify that SatelliteSfM completed successfully
- Check that all required Python dependencies are installed

---

## Citation

If you use this preprocessing pipeline, please cite the original SatelliteSfM work, COLMAP, the DFC2019 dataset and our work.

```
@inproceedings{VisSat-2019,
  title={Leveraging Vision Reconstruction Pipelines for Satellite Imagery},
  author={Zhang, Kai and Sun, Jin and Snavely, Noah},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}

@inproceedings{schoenberger2016sfm,
  author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
  title={Structure-from-Motion Revisited},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016},
}

@data{c6tm-vw12-19,
  doi = {10.21227/c6tm-vw12},
  url = {https://dx.doi.org/10.21227/c6tm-vw12},
  author = {Le Saux, Bertrand and Yokoya, Naoto and Hänsch, Ronny and Brown, Myron},
  publisher = {IEEE Dataport},
  title = {Data Fusion Contest 2019 ({DFC2019})},
  year = {2019},
}

@article{lee2025SkyfallGS,
  title = {{Skyfall-GS}: Synthesizing Immersive {3D} Urban Scenes from Satellite Imagery},
  author = {Jie-Ying Lee and Yi-Ruei Liu and Shr-Ruei Tsai and Wei-Cheng Chang and Chung-Ho Wu and Jiewen Chan and Zhenjun Zhao and Chieh Hubert Lin and Yu-Lun Liu},
  journal = {arXiv preprint},
  year = {2025},
  eprint = {2510.15869},
  archivePrefix = {arXiv}
}
```

> Below is the original README
---

# Satellite Structure from Motion

Maintained by [Kai Zhang](https://kai-46.github.io/website/).

## Why this repo?

I started my computer vision research journey with satellite stereo being my first project. Working on that problem makes me feel that there seems to be an unnesseary[?] gap between how the stereo problems are approached in the computer vision community and remote sensing community. And moreover, satellite images seem to attract relatively less attention from the vast computer vision community. I was guessing perhaps this was due to the limited satellite image availability, which seems to be improving these days. With the increasing availability of satellite datasets, I am hoping to further simplify the access to satellite stereo problems for computer vision researchers' and practitioners' with this repo.

## Development roadmaps (Open-source contributions are always welcome!)

- [x] release SatelliteSfM
- [x] release [SatelliteNeRF](https://github.com/Kai-46/SatelliteNeRF) as downstream neural rendering applications
- [x] release scripts to visualize SatelliteSfM output cameras in 3D
- [x] release [TRACK 3: MULTI-VIEW SEMANTIC STEREO](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) data preprocessed by SatelliteSfM
- [x] re-write [ColmapForVisSat](https://github.com/Kai-46/ColmapForVisSat) as patches to latest [Colmap](https://github.com/colmap/colmap): SfM first, followed by MVS, and finally meshing. You can find the re-written version [ColmapForVisSatPatched](https://github.com/SBCV/ColmapForVisSatPatched). Thanks to @SBCV. 
- [x] release [SatelliteNeuS](https://github.com/Kai-46/SatelliteNeuS) that can reconstruct meshes from multi-date satellite images with varying illuminations 
- [x] draw a road map
- [ ] improve documentations of the [SatellitePlaneSweep](https://github.com/Kai-46/SatellitePlaneSweep) and [SatelliteNeRF](https://github.com/Kai-46/SatelliteNeRF) and [SatelliteNeuS](https://github.com/Kai-46/SatelliteNeuS)
- [ ] port [SatelliteSurfaceReconstruction](https://github.com/SBCV/SatelliteSurfaceReconstruction) meshing algorithm to the new API
- [ ] release Deep Satellite Stereo as downstream MVS algorithms
- [ ] release code to rectify satellite stereo pairs based on the SatelliteSfM outputs
- [ ] release code to run stereo matching on rectified stereo pairs, including both classical and deep ones

<img width="1359" alt="roadmap" src="https://user-images.githubusercontent.com/21653654/164472067-abb78d47-deab-4b95-8a62-5ce38a78966d.png">

## Relevant repos for downstream applications

- [Satellite-based Neural Radiance Fields](https://github.com/Kai-46/SatelliteNeRF)

## Overview

- This is a library dedicated to solving the satellite structure from motion problem.
- It's a wrapper of the [VisSatSatelliteStereo repo](https://github.com/Kai-46/VisSatSatelliteStereo) for easier use.
- The outputs are png images and **OpenCV-compatible** pinhole cameras readily deployable to multi-view stereo pipelines targetting ground-level images.

## Installation

Assume you are on a Linux machine with at least one GPU, and have conda installed. Then to install this library, simply by:

```bash
. ./env.sh
```

## Inputs

We assume the inputs to be a set of .tif images encoding the 3-channel uint8 RGB colors, and the metadata like RPC cameras.
This data format is to align with the public satellite benchmark: [TRACK 3: MULTI-VIEW SEMANTIC STEREO](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019).
Download one example data from this [google drive](https://drive.google.com/drive/folders/11UeurSa-dyfaRUIdUZFfNBAyd3jN7D46?usp=sharing); folder structure look like below:

```
- examples/inputs
    - images/
        - *.tif
        - *.tif
        - *.tif
        - ...
    - latlonalt_bbx.json
```

, where ```latlonalt_bbx.json``` specifies the bounding box for the site of interest in the global (latitude, longitude, altitude) coordinate system.

If you are not sure what is a reasonably good altitude range, you can put random numbers in the json file, but you have to enable the ```--use_srtm4``` option below.  

## Run Structure from Motion

```bash
python satellite_sfm.py --input_folder examples/inputs --output_folder examples/outputs --run_sfm [--use_srtm4] [--enable_debug]
```

The ```--enable_debug``` option outputs some visualization helpful debugging the structure from motion quality.

## Outputs

- ```{output_folder}/images/``` folder contains the png images
- ```{output_folder}/cameras_adjusted/``` folder contains the bundle-adjusted pinhole cameras; each camera is represented by a pair of 4x4 K, W2C matrices that are OpenCV-compatible.
- ```{output_folder}/enu_bbx_adjusted.json``` contains the scene bounding box in the local ENU Euclidean coordinate system.
- ```{output_folder}/enu_observer_latlonalt.json``` contains the observer coordinate for defining the local ENU coordinate; essentially, this observer coordinate is only necessary for coordinate conversion between local ENU and global latitude-longitude-altitude.

If you turn on the ```--enable_debug``` option, you might want to dig into the folder ```{output_folder}/debug_sfm``` for visuals, etc.

## Citations

```
@inproceedings{VisSat-2019,
  title={Leveraging Vision Reconstruction Pipelines for Satellite Imagery},
  author={Zhang, Kai and Sun, Jin and Snavely, Noah},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}

@inproceedings{schoenberger2016sfm,
  author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
  title={Structure-from-Motion Revisited},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016},
}
```

## Example results

### input images

![Input images](./readme_resources/example_data.gif)

### sparse point cloud ouput by SfM

![Sparse point cloud](./readme_resources/example_data_sfm.gif)

### Visualize cameras

```
python visualize_satellite_cameras.py
```

Red, Green, Blue axes denote east, north, up directions, respectively. For simplicity, each camera is represented by a line pointing from origin to that camera center.
![Visualize cameras](./readme_resources/visualize_camera.png)

### homograhpy-warp one view, then average with another by a plane sequence

![Sweep plane](./readme_resources/sweep_plane.gif)
[high-res video](https://drive.google.com/file/d/13TshDCsHTx0J7X6UFd0zglutQkD8NgyK/view?usp=sharing)

### inspect epipolar geometry

```
python inspect_epipolar_geometry.py
```

![inspect epipolar](./readme_resources/debug_epipolar.png)

### get zero-skew intrincics marix

```
python skew_correct.py --input_folder ./examples/outputs ./examples/outputs_zeroskew
```

![skew correct](./readme_resources/skew_correct.png)

## Downstream applications

One natural task following this SatelliteSfM is to acquire the dense reconstruction by classical patch-based MVS, or mordern deep MVS, or even neural rendering like NeRF. When working with these downstream algorithms, be careful of the float32 pitfall caused by the huge depth values as a result of **satellite cameras being distant from the scene**; this is particularly worthy of attention with the prevalent float32 GPU computing.  

[Note: this SatelliteSfM library doesn't have such issue for the use of float64.]

### pitfall of float32 arithmetic

![numeric precison](./readme_resources/numeric_precision.png)

### overcome float32 pitfall for NeRF

Center and scale scene to be inside unit sphere by:

```bash
python normalize_sfm_reconstruction.py
```

Modify how ```pixel2ray``` is computed for NeRF-based models, while keeping the other parts unchanged:

```python
import torch

def pixel2ray(col: torch.Tensor, row: torch.Tensor, K: torch.DoubleTensor, W2C: torch.DoubleTensor):
    '''
    Assume scene is centered and inside unit sphere.

    col, row: both [N, ]; float32
    K, W2C: 4x4 opencv-compatible intrinsic and W2C matrices; float64

    return:
        ray_o, ray_d: [N, 3]; float32
    '''
    C2W = torch.inverse(W2C)  # float64
    px = torch.stack((col, row, torch.ones_like(col)), axis=-1).unsqueeze(-1)  # [N, 3, 1]; float64
    K_inv = torch.inverse(K[:3, :3]).unsqueeze(0).expand(px.shape[0], -1, -1)  # [N, 3, 3]; float64
    c2w_rot = C2W[:3, :3].unsqueeze(0).expand(px.shape[0], -1, -1) # [N, 3, 3]; float64
    ray_d = torch.matmul(c2w_rot, torch.matmul(K_inv, px.double())) # [N, 3, 1]; float64
    ray_d = (ray_d / ray_d.norm(dim=1, keepdims=True)).squeeze(-1) # [N, 3]; float64

    ray_o = C2W[:3, 3].unsqueeze(0).expand(px.shape[0], -1) # [N, 3]; float64
    # shift ray_o along ray_d towards the scene in order to shrink the huge depth
    shift = torch.norm(ray_o, dim=-1) - 5.  # [N, ]; float64; 5. here is a small margin
    ray_o = ray_o + ray_d * shift.unsqueeze(-1)  # [N, 3]; float64
    return ray_o.float(), ray_d.float()
```
<!-- ![novel view](./readme_resources/novel_view.gif) -->
<https://user-images.githubusercontent.com/21653654/153779703-36b50265-ae3b-41ac-8139-2e0bf081f28d.mp4>

<https://user-images.githubusercontent.com/21653654/153779789-93f68ce9-9cc4-4947-81de-d6d2104ae0ac.mp4>

<https://user-images.githubusercontent.com/21653654/153779889-8116d7ee-8a4d-474c-8d52-3b1f0e175104.mp4>

<https://user-images.githubusercontent.com/21653654/153779898-dba46433-143e-499a-9315-4316747e6e59.mp4>

<https://user-images.githubusercontent.com/21653654/153779906-b4196d7d-afd7-4fde-b691-0d7c6a785f8b.mp4>

<https://user-images.githubusercontent.com/21653654/153779913-36931e65-2739-4d35-8901-22808d8eaced.mp4>

<https://user-images.githubusercontent.com/21653654/153779919-5157b7df-d59b-48f8-a66c-7982b600e01d.mp4>

<https://user-images.githubusercontent.com/21653654/153779930-9902b8b3-b035-4c78-ac51-ff97dbb0f266.mp4>

<https://user-images.githubusercontent.com/21653654/153779937-74ef0a19-8ce8-4d87-84a3-357513c419ed.mp4>

<https://user-images.githubusercontent.com/21653654/153780455-939f4e36-794b-4282-b9a1-c1b98e8a1866.mp4>

### overcome float32 pitfall for neural point based graphics

to be filled...

### overcome float32 pitfall for plane sweep stereo, or patch-based stereo, or deep stereo

to be filled...

## preprocessed satellite multi-view stereo dataset with ground-truth

This dataset can be used for evaluating multi-view stereo, running neural rendering, etc. You can download it from [google drive](https://drive.google.com/drive/folders/1Do7oF36sCEBWrcIiCzgHbhF5kQMawjVo?usp=sharing).

## More handy scripts are coming

Stay tuned :-)
