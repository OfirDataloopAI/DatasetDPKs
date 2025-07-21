SRC: https://www.kaggle.com/datasets/dulanim/retailgaze

# RetailGaze Dataset

## Overview
RetailGaze is a dataset for gaze estimation in real-world retail environments. It contains 3,922 images of individuals looking at products in retail settings, captured from 12 different camera angles.

**Dataset Details:**
- Total images: 7,844 (across all splits)
- Train set: 2,728 images (70%)
- Test set: 609 images (15%) 
- Validation set: 585 images (15%)
- Complete dataset: 3,922 unique images

## Data Structure
Each entry contains:
- `filename`: Image path (e.g., `1/060433/0.jpg`)
- `width`, `height`: Image dimensions (640x480)
- `gaze_cx`, `gaze_cy`: Gaze point coordinates in the image
- `hbox`: Head bounding box `[xmin, ymin, xmax, ymax]`
- `seg_mask`: Path to segmentation mask of gazed product area

## Files in this Directory

### Original Data
- `RetailGaze_V2_seg/` - Contains original pickle files and image data
  - `RetailGaze_V3_2.pickle` - Complete dataset (3,922 entries)
  - `RetailGaze_V3_2_train.pickle` - Training split (2,728 entries)
  - `RetailGaze_V3_2_test.pickle` - Test split (609 entries) 
  - `RetailGaze_V3_2_valid.pickle` - Validation split (585 entries)

### Extracted Data
- `extracted_data/` - Processed data in multiple formats
  - `*.csv` - Tabular format for easy analysis
  - `*.json` - Structured format preserving all data
  - `dataset_summary.json` - Complete analysis report with statistics

### Processing Scripts
- `extract_retailgaze_pickles.py` - Main extraction script
- `simple_csv_export.py` - JSON to CSV converter
- `upload_data.py` - Upload data to Dataloop platform
- `upload_data_test.py` - Test version (uploads first 5 samples)

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Data Analysis
```python
import pandas as pd

# Load training data
train_data = pd.read_csv('extracted_data/RetailGaze_V3_2_train.csv')

# Access gaze coordinates
gaze_points = train_data[['gaze_cx', 'gaze_cy']]

# Access head bounding boxes  
head_boxes = train_data[['hbox_xmin', 'hbox_ymin', 'hbox_xmax', 'hbox_ymax']]
```

### Upload to Dataloop
```python
import dtlpy as dl

# First test with 5 samples
python upload_data_test.py

# Then upload full dataset
python upload_data.py
```

**What gets uploaded:**
- 🖼️ **Images**: All training images with proper paths
- 📦 **Head Bounding Boxes**: `dl.Box` annotations for head detection
- 📍 **Gaze Points**: `dl.Point` annotations for gaze coordinates  
- 🎭 **Segmentation Masks**: `dl.Polygon` annotations for gazed objects (merged from all mask files)

## References
- **IEEE DataPort**: https://ieee-dataport.org/documents/retail-gaze-gaze-estimation-retail-environment
- **GitHub Repository**: https://github.com/PrimeshShamilka/RetailGazeDataset