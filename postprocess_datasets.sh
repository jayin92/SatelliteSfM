#!/bin/bash

# This script post-processes the datasets for specified scenes
# It performs skew correction using SRTM data, converts the datasets into the desired format,
# and generates masks for non-black pixels.

set -e  # Exit on error (optional, remove if you want to continue on errors)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
BASE_DIR="data/DFC2019_processed"
default_scenes=(JAX_004 JAX_068 JAX_214 JAX_260 JAX_164 JAX_168 JAX_175 JAX_264)

# Parse command line arguments
SKIP_SKEW=false
SKIP_CONVERT=false
SKIP_COPY=false
SKIP_MASK=false
DRY_RUN=false

usage() {
    echo "Usage: $0 [OPTIONS] [SCENE1 SCENE2 ...]"
    echo ""
    echo "Options:"
    echo "  --skip-skew       Skip skew correction step"
    echo "  --skip-convert    Skip dataset conversion step"
    echo "  --skip-copy       Skip points3D.txt copy step"
    echo "  --skip-mask       Skip mask generation step"
    echo "  --dry-run         Print commands without executing them"
    echo "  --base-dir DIR    Set base directory (default: data/DFC2019_processed)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Scene format: CITY_SCENEID (e.g., JAX_004, OMA_068)"
    echo ""
    echo "Examples:"
    echo "  $0 JAX_004 OMA_068           # Process specific scenes"
    echo "  $0                            # Process default scenes"
    echo "  $0 --skip-skew JAX_004        # Skip skew correction"
    echo "  $0 --dry-run JAX_004          # Show what would be done"
}

# Parse options
scenes=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-skew)
            SKIP_SKEW=true
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-copy)
            SKIP_COPY=true
            shift
            ;;
        --skip-mask)
            SKIP_MASK=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            exit 1
            ;;
        *)
            scenes+=("$1")
            shift
            ;;
    esac
done

# Use default scenes if none specified
if [ ${#scenes[@]} -eq 0 ]; then
    scenes=("${default_scenes[@]}")
    echo -e "${YELLOW}No scenes specified, using default scenes: ${scenes[@]}${NC}"
else
    echo -e "${GREEN}Processing scenes: ${scenes[@]}${NC}"
fi

# Function to run command
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} $@"
    else
        "$@"
    fi
}

# Process each scene
success_count=0
fail_count=0
failed_scenes=()

for scene in "${scenes[@]}"; do
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Post-processing scene: ${scene}${NC}"
    echo "=========================================="
    
    # Extract city and scene_id from scene name
    if [[ ! $scene =~ ^([A-Z]+)_([0-9]+)$ ]]; then
        echo -e "${RED}Error: Invalid scene format '$scene'. Expected format: CITY_ID (e.g., JAX_004, OMA_068)${NC}"
        ((fail_count++))
        failed_scenes+=("$scene")
        continue
    fi
    
    city="${BASH_REMATCH[1]}"
    scene_id="${BASH_REMATCH[2]}"
    
    # Define paths
    input_folder="${BASE_DIR}/${scene}/outputs_srtm"
    output_folder="${BASE_DIR}/${scene}/outputs_skew"
    points3d_src="${BASE_DIR}/${scene}/outputs_srtm/colmap_triangulate_postba/points3D.txt"
    points3d_dst="${BASE_DIR}/${scene}/outputs_skew/points3D.txt"
    images_dir="${BASE_DIR}/${scene}/outputs_skew/images"
    masks_dir="${BASE_DIR}/${scene}/outputs_skew/masks"
    
    # Check if input folder exists
    if [ ! -d "$input_folder" ]; then
        echo -e "${RED}Warning: Input folder does not exist: $input_folder${NC}"
        echo "Skipping scene ${scene}"
        ((fail_count++))
        failed_scenes+=("$scene")
        continue
    fi
    
    # Track if this scene succeeded
    scene_success=true
    
    # Step 1: Skew correction
    if [ "$SKIP_SKEW" = false ]; then
        echo ""
        echo "Step 1: Running skew correction..."
        if ! run_cmd python skew_correct.py \
            --input_folder "$input_folder" \
            --output_folder "$output_folder"; then
            echo -e "${RED}Error: skew_correct.py failed for scene ${scene}${NC}"
            scene_success=false
        fi
    else
        echo -e "${YELLOW}Step 1: Skipping skew correction${NC}"
    fi
    
    # Step 2: Dataset conversion
    if [ "$scene_success" = true ] && [ "$SKIP_CONVERT" = false ]; then
        echo ""
        echo "Step 2: Converting dataset..."
        if ! run_cmd python convert_datasets.py \
            --input_folder "$output_folder"; then
            echo -e "${RED}Error: convert_datasets.py failed for scene ${scene}${NC}"
            scene_success=false
        fi
    elif [ "$SKIP_CONVERT" = true ]; then
        echo -e "${YELLOW}Step 2: Skipping dataset conversion${NC}"
    fi
    
    # Step 3: Copy points3D.txt
    if [ "$scene_success" = true ] && [ "$SKIP_COPY" = false ]; then
        echo ""
        echo "Step 3: Copying points3D.txt..."
        if [ -f "$points3d_src" ]; then
            run_cmd cp "$points3d_src" "$points3d_dst"
            echo "Copied: $points3d_src -> $points3d_dst"
        else
            echo -e "${YELLOW}Warning: Source file not found: $points3d_src${NC}"
        fi
    elif [ "$SKIP_COPY" = true ]; then
        echo -e "${YELLOW}Step 3: Skipping points3D.txt copy${NC}"
    fi
    
    # Step 4: Generate masks
    if [ "$scene_success" = true ] && [ "$SKIP_MASK" = false ]; then
        echo ""
        echo "Step 4: Generating masks for non-black pixels..."
        if [ -d "$images_dir" ]; then
            if ! run_cmd python generate_masks.py \
                --input_dir "$images_dir" \
                --output_dir "$masks_dir"; then
                echo -e "${RED}Error: generate_masks.py failed for scene ${scene}${NC}"
                scene_success=false
            fi
        else
            echo -e "${YELLOW}Warning: Images directory not found: $images_dir${NC}"
            echo "Skipping mask generation"
        fi
    elif [ "$SKIP_MASK" = true ]; then
        echo -e "${YELLOW}Step 4: Skipping mask generation${NC}"
    fi
    
    # Update counters
    if [ "$scene_success" = true ]; then
        echo ""
        echo -e "${GREEN}✓ Successfully processed scene: ${scene}${NC}"
        ((success_count++))
    else
        ((fail_count++))
        failed_scenes+=("$scene")
    fi
done

# Final summary
echo ""
echo "=========================================="
echo "Processing Summary"
echo "=========================================="
echo -e "${GREEN}Successful: ${success_count}${NC}"
echo -e "${RED}Failed: ${fail_count}${NC}"

if [ ${#failed_scenes[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed scenes:${NC}"
    for failed_scene in "${failed_scenes[@]}"; do
        echo "  - $failed_scene"
    done
fi

echo ""
echo "=========================================="
echo "All scenes processed!"
echo "=========================================="

# Exit with error if any scene failed
if [ $fail_count -gt 0 ]; then
    exit 1
fi