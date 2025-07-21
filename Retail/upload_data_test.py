import dtlpy as dl
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def merge_masks(masks_dir, mask_filename):
    """
    Merge all mask files in the directory using logical OR operation.
    Returns the merged mask as a numpy array.
    """
    masks_path = Path(masks_dir)
    mask_files = list(masks_path.glob("*.png"))
    
    if not mask_files:
        print(f"⚠ No mask files found in {masks_dir}")
        return None
    
    # Load first mask to get dimensions
    first_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        print(f"⚠ Could not load mask: {mask_files[0]}")
        return None
    
    # Initialize merged mask
    merged_mask = np.zeros_like(first_mask)
    
    # Merge all masks using logical OR
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Binarize mask (assuming non-zero pixels are mask areas)
            _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            merged_mask = cv2.bitwise_or(merged_mask, binary_mask)
    
    return merged_mask


def upload_data_test(dataset, extracted_data_path, binaries_path, max_samples=5):
    """TEST VERSION: Upload only first few RetailGaze samples for testing."""
    
    # Step 1: Read all the train images paths and their annotations
    print("🧪 TEST MODE: Processing only first 5 samples...")
    print("📖 Step 1: Reading training data and annotations...")
    csv_path = os.path.join(extracted_data_path, "RetailGaze_V3_2_train.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Load annotations data
    df = pd.read_csv(csv_path)
    # Limit to first few samples for testing
    df = df.head(max_samples)
    print(f"✅ Loaded {len(df)} training samples (TEST MODE)")
    
    success_count = 0
    
    # Process each image
    for idx, row in df.iterrows():
        print(f"\n🔄 Processing TEST image {idx + 1}/{len(df)}: {row['filename']}")
        
        # Construct image path
        image_path = os.path.join(binaries_path, row['filename'])
        
        if not os.path.exists(image_path):
            print(f"⚠ Image not found: {image_path}")
            continue
        
        try:
            # Step 2: Upload the image using dataset.items.upload()
            print(f"📤 Step 2: Uploading image...")
            item = dataset.items.upload(local_path=image_path, remote_name=f"test_{row['filename']}")
            print(f"✅ Uploaded: {item.name}")
            
            # Step 3: Get the item and call builder = item.annotations.builder()
            print(f"🔧 Step 3: Creating annotation builder...")
            builder = item.annotations.builder()
            
            # Step 4: Add bounding box annotation (dl.Box)
            print(f"📦 Step 4: Adding head bounding box...")
            head_box = dl.Box(
                top=row['hbox_ymin'],
                left=row['hbox_xmin'],
                bottom=row['hbox_ymax'],
                right=row['hbox_xmax'],
                label='head'
            )
            builder.add(annotation_definition=head_box)
            
            # Add gaze point as a point annotation
            gaze_point = dl.Point(
                x=row['gaze_cx'],
                y=row['gaze_cy'],
                label='gaze_point'
            )
            builder.add(annotation_definition=gaze_point)
            
            # Step 5: Merge masks and add segmentation (dl.Segmentation)
            print(f"🎭 Step 5: Processing segmentation masks...")
            mask_dir = os.path.join(binaries_path, os.path.dirname(row['seg_mask']).replace('/', os.sep))
            
            if os.path.exists(mask_dir):
                # Merge all masks in the directory
                merged_mask = merge_masks(mask_dir, row['seg_mask'])
                
                if merged_mask is not None:
                    # Convert mask to polygon format for Dataloop
                    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    polygon_count = 0
                    for contour in contours:
                        if len(contour) >= 3:  # Need at least 3 points for polygon
                            # Convert contour to list of points
                            polygon_points = []
                            for point in contour:
                                x, y = point[0]
                                polygon_points.extend([float(x), float(y)])
                            
                            if len(polygon_points) >= 6:  # At least 3 points (6 coordinates)
                                segmentation = dl.Polygon(
                                    geo=polygon_points,
                                    label='gazed_object'
                                )
                                builder.add(annotation_definition=segmentation)
                                polygon_count += 1
                    
                    print(f"  Added {polygon_count} segmentation polygons")
                else:
                    print(f"⚠ Could not process masks in {mask_dir}")
            else:
                print(f"⚠ Mask directory not found: {mask_dir}")
            
            # Upload all annotations
            print(f"📤 Uploading annotations...")
            item.annotations.upload(builder)
            print(f"✅ Annotations uploaded for {row['filename']}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {row['filename']}: {str(e)}")
            continue
    
    # Step 6: Profit! 🎉
    print(f"\n🎉 Step 6: Test Complete!")
    print(f"📊 Dataset: {dataset.name}")
    print(f"✅ Successfully processed: {success_count}/{len(df)} samples")
    print(f"📁 Total items processed in TEST MODE: {len(df)}")


def main():
    dataset_id = "64b800000000000000000000"
    dataset = dl.datasets.get(dataset_id=dataset_id)

    extracted_data_path = "RetailGaze_V2_seg/extracted_data"
    binaries_path = "RetailGaze_V2_seg/RetailGaze_V2"
    
    # Test with first 5 samples
    upload_data_test(
        dataset=dataset,
        extracted_data_path=extracted_data_path,
        binaries_path=binaries_path,
        max_samples=5
    )


if __name__ == "__main__":
    main() 