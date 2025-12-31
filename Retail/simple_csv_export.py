#!/usr/bin/env python3
"""
Simple CSV exporter for RetailGaze JSON files
"""

import json
import csv
from pathlib import Path

def json_to_csv(json_file, csv_file):
    """Convert JSON file to CSV without pandas."""
    try:
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not data or not isinstance(data, list):
            print(f"⚠ No valid data in {json_file}")
            return
        
        # Define CSV headers
        headers = [
            'filename', 'width', 'height', 'gaze_cx', 'gaze_cy', 'seg_mask',
            'hbox_xmin', 'hbox_ymin', 'hbox_xmax', 'hbox_ymax'
        ]
        
        # Write CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for entry in data:
                if isinstance(entry, dict):
                    row = {}
                    
                    # Basic fields
                    for key in ['filename', 'width', 'height', 'gaze_cx', 'gaze_cy', 'seg_mask']:
                        row[key] = entry.get(key, '')
                    
                    # Handle hbox
                    if 'ann' in entry and isinstance(entry['ann'], dict) and 'hbox' in entry['ann']:
                        hbox = entry['ann']['hbox']
                        if isinstance(hbox, (list, tuple)) and len(hbox) >= 4:
                            row['hbox_xmin'] = hbox[0]
                            row['hbox_ymin'] = hbox[1]
                            row['hbox_xmax'] = hbox[2]
                            row['hbox_ymax'] = hbox[3]
                        else:
                            row['hbox_xmin'] = row['hbox_ymin'] = row['hbox_xmax'] = row['hbox_ymax'] = ''
                    
                    writer.writerow(row)
        
        print(f"✓ Converted {json_file} to {csv_file}")
        print(f"  {len(data)} entries written")
        
    except Exception as e:
        print(f"✗ Failed to convert {json_file}: {str(e)}")

def main():
    folder_root_path = Path(__file__).parent

    json_dir = str(folder_root_path.joinpath("RetailGaze_V2_seg/extracted_data"))
    
    # Find all JSON files
    json_files = list(json_dir.glob("RetailGaze_*.json"))
    
    print("=== Converting JSON files to CSV ===\n")
    
    for json_file in json_files:
        csv_file = json_file.with_suffix('.csv')
        print(f"Converting {json_file.name}...")
        json_to_csv(json_file, csv_file)
        print()
    
    print("✓ All conversions complete!")

if __name__ == "__main__":
    main() 