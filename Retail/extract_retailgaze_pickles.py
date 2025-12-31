#!/usr/bin/env python3
"""
RetailGaze Dataset Pickle Extractor

This script extracts and analyzes the RetailGaze dataset pickle files.
Based on the IEEE DataPort documentation, each pickle file contains a list of 
dictionary objects with the following structure:

object = {
    'filename': "the filename tree of this image",
    'width': "640width",
    'height': "480height", 
    'gaze_cx': "gaze point x coordinate in the image",
    'gaze_cy': "gaze point y coordinate in the image",
    'ann': {
        'hbox': "Bounding box of the head [xmin, ymin, xmax, ymax]"
    },
    'seg_mask': 'the ground truth segmentation mask tree'
}
"""

import pickle
import json
import csv
import os
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠ pandas not available, CSV export will be limited")

def load_pickle_file(pickle_path):
    """Load and return the contents of a pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded {pickle_path}")
        print(f"  - Type: {type(data)}")
        print(f"  - Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        return data
    except Exception as e:
        print(f"✗ Failed to load {pickle_path}: {str(e)}")
        return None

def analyze_pickle_data(data, filename):
    """Analyze the structure and content of pickle data."""
    print(f"\n=== Analysis of {filename} ===")
    
    if not data:
        print("No data to analyze")
        return
        
    if isinstance(data, list) and len(data) > 0:
        print(f"Number of entries: {len(data)}")
        
        # Analyze first entry
        first_entry = data[0]
        print(f"First entry type: {type(first_entry)}")
        
        if isinstance(first_entry, dict):
            print("Keys in first entry:")
            for key in first_entry.keys():
                value = first_entry[key]
                value_type = type(value).__name__
                
                if isinstance(value, (str, int, float)):
                    print(f"  - {key}: {value} ({value_type})")
                elif isinstance(value, dict):
                    print(f"  - {key}: {dict(list(value.items())[:3])} ({value_type}, {len(value)} keys)")
                elif isinstance(value, (list, tuple)):
                    print(f"  - {key}: {value[:3] if len(value) > 3 else value} ({value_type}, length {len(value)})")
                else:
                    print(f"  - {key}: {value_type}")
                    
        # Show sample of more entries
        print(f"\nSample filenames (first 5):")
        for i, entry in enumerate(data[:5]):
            if isinstance(entry, dict) and 'filename' in entry:
                print(f"  {i+1}. {entry['filename']}")
                
    else:
        print(f"Data type: {type(data)}")
        print(f"Data preview: {str(data)[:200]}...")

def export_to_csv(data, output_path):
    """Export pickle data to CSV format."""
    if not data or not isinstance(data, list):
        print(f"Cannot export to CSV: invalid data type")
        return
        
    try:
        # Flatten the data for CSV export
        flattened_data = []
        
        for entry in data:
            if isinstance(entry, dict):
                flat_entry = {}
                
                # Basic fields
                for key in ['filename', 'width', 'height', 'gaze_cx', 'gaze_cy', 'seg_mask']:
                    flat_entry[key] = entry.get(key, '')
                
                # Handle nested 'ann' field
                if 'ann' in entry and isinstance(entry['ann'], dict):
                    if 'hbox' in entry['ann']:
                        hbox = entry['ann']['hbox']
                        if isinstance(hbox, (list, tuple)) and len(hbox) >= 4:
                            flat_entry['hbox_xmin'] = hbox[0]
                            flat_entry['hbox_ymin'] = hbox[1] 
                            flat_entry['hbox_xmax'] = hbox[2]
                            flat_entry['hbox_ymax'] = hbox[3]
                        else:
                            flat_entry['hbox_raw'] = str(hbox)
                    
                    # Add any other keys from ann
                    for key, value in entry['ann'].items():
                        if key != 'hbox':
                            flat_entry[f'ann_{key}'] = value
                
                flattened_data.append(flat_entry)
        
        # Create DataFrame and save
        df = pd.DataFrame(flattened_data)
        df.to_csv(output_path, index=False)
        print(f"✓ Exported {len(flattened_data)} entries to {output_path}")
        
        # Show column info
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        
    except Exception as e:
        print(f"✗ Failed to export to CSV: {str(e)}")

def export_to_json(data, output_path):
    """Export pickle data to JSON format."""
    if not data:
        print(f"Cannot export to JSON: no data")
        return
        
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✓ Exported to {output_path}")
    except Exception as e:
        print(f"✗ Failed to export to JSON: {str(e)}")

def create_summary_report(all_data, output_dir):
    """Create a summary report of all datasets."""
    summary = {
        'dataset_info': {
            'description': 'RetailGaze Dataset - Gaze estimation in retail environments',
            'source': 'https://www.kaggle.com/datasets/dulanim/retailgaze',
            'total_files_processed': len(all_data)
        },
        'files': {}
    }
    
    total_images = 0
    
    for filename, data in all_data.items():
        if data and isinstance(data, list):
            file_info = {
                'num_entries': len(data),
                'sample_filenames': []
            }
            
            # Collect sample filenames
            for entry in data[:10]:  # First 10
                if isinstance(entry, dict) and 'filename' in entry:
                    file_info['sample_filenames'].append(entry['filename'])
            
            # Analyze gaze points distribution
            gaze_cx_values = []
            gaze_cy_values = []
            
            for entry in data:
                if isinstance(entry, dict):
                    if 'gaze_cx' in entry and entry['gaze_cx'] is not None:
                        try:
                            gaze_cx_values.append(float(entry['gaze_cx']))
                        except (ValueError, TypeError):
                            pass
                    if 'gaze_cy' in entry and entry['gaze_cy'] is not None:
                        try:
                            gaze_cy_values.append(float(entry['gaze_cy']))
                        except (ValueError, TypeError):
                            pass
            
            if gaze_cx_values:
                file_info['gaze_stats'] = {
                    'gaze_cx_min': min(gaze_cx_values),
                    'gaze_cx_max': max(gaze_cx_values),
                    'gaze_cy_min': min(gaze_cy_values),
                    'gaze_cy_max': max(gaze_cy_values),
                    'valid_gaze_points': len(gaze_cx_values)
                }
            
            summary['files'][filename] = file_info
            total_images += len(data)
    
    summary['dataset_info']['total_images'] = total_images
    
    # Save summary
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Created summary report: {summary_path}")
    print(f"  Total images across all files: {total_images}")

def main():
    folder_root_path = Path(__file__).parent

    # Define paths
    pickle_dir = str(folder_root_path.joinpath("RetailGaze_V2_seg"))
    output_dir = str(folder_root_path.joinpath("RetailGaze_V2_seg/extracted_data"))
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # List of pickle files to process
    pickle_files = [
        "RetailGaze_V3_2.pickle",
        "RetailGaze_V3_2_train.pickle", 
        "RetailGaze_V3_2_test.pickle",
        "RetailGaze_V3_2_valid.pickle"
    ]
    
    print("=== RetailGaze Dataset Pickle Extractor ===\n")
    
    all_data = {}
    
    # Process each pickle file
    for pickle_file in pickle_files:
        pickle_path = pickle_dir / pickle_file
        
        if not pickle_path.exists():
            print(f"⚠ File not found: {pickle_path}")
            continue
            
        print(f"\nProcessing {pickle_file}...")
        
        # Load pickle file
        data = load_pickle_file(pickle_path)
        if data is None:
            continue
            
        # Store for summary
        all_data[pickle_file] = data
            
        # Analyze the data
        analyze_pickle_data(data, pickle_file)
        
        # Export to different formats
        base_name = pickle_file.replace('.pickle', '')
        
        # Export to CSV
        csv_path = output_dir / f"{base_name}.csv"
        export_to_csv(data, csv_path)
        
        # Export to JSON
        json_path = output_dir / f"{base_name}.json" 
        export_to_json(data, json_path)
        
        print("-" * 50)
    
    # Create summary report
    if all_data:
        create_summary_report(all_data, output_dir)
    
    print(f"\n✓ Processing complete! Check the '{output_dir}' directory for extracted files.")
    print(f"\nFiles created:")
    for file_path in sorted(output_dir.glob("*")):
        print(f"  - {file_path.name}")

if __name__ == "__main__":
    main() 