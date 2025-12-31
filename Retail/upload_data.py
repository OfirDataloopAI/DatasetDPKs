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
    
    merged_mask = np.where(merged_mask > 0, 1, 0)
    return merged_mask


def upload_data(dataset, extracted_data_path, binaries_path, samples_per_dir=5):
    """Upload RetailGaze data to Dataloop dataset with sampling from each directory."""
    
    # Step 1: Read all the train images paths and their annotations
    print("📖 Step 1: Reading training data and annotations...")
    csv_path = os.path.join(extracted_data_path, "RetailGaze_V3_2_train.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Load annotations data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} training samples")
    
    # Group by directory (first part of filename path)
    print(f"🗂️ Grouping images by directory...")
    df['directory'] = df['filename'].str.split('/').str[0]
    
    # Sample approximately 5 images from each directory
    sampled_dfs = []
    directories = df['directory'].unique()
    
    print(f"📁 Found {len(directories)} unique directories")
    
    for directory in directories:
        dir_df = df[df['directory'] == directory]
        # Take up to samples_per_dir images from this directory
        sampled_dir_df = dir_df.head(samples_per_dir)
        sampled_dfs.append(sampled_dir_df)
        print(f"  📂 {directory}: {len(dir_df)} images → sampling {len(sampled_dir_df)}")
    
    # Combine all sampled data
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\n🎯 Final sample: {len(df_sampled)} images from {len(directories)} directories")
    print(f"📊 Reduction: {len(df)} → {len(df_sampled)} images ({len(df_sampled)/len(df)*100:.1f}%)")
    
    success_count = 0
    
    # Process each image
    for idx, row in df_sampled.iterrows():
        print(f"\n🔄 Processing image {idx + 1}/{len(df_sampled)}: {row['filename']}")
        
        # Construct image path
        image_path = os.path.join(binaries_path, row['filename'])
        
        if not os.path.exists(image_path):
            print(f"⚠ Image not found: {image_path}")
            continue
        
        try:
            # Step 2: Upload the image using dataset.items.upload()
            print(f"📤 Step 2: Uploading image...")
            item = dataset.items.upload(local_path=image_path, remote_name=row['filename'])
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
            
            # Check if seg_mask is NaN or empty
            if pd.isna(row['seg_mask']) or row['seg_mask'] == '' or row['seg_mask'] is None:
                print(f"⚠ seg_mask is NaN/empty for {row['filename']}, skipping segmentation")
            else:
                mask_dir = os.path.join(binaries_path, os.path.dirname(row['seg_mask']).replace('/', os.sep))
                
                if os.path.exists(mask_dir):
                    # Merge all masks in the directory
                    merged_mask = merge_masks(mask_dir, row['seg_mask'])
                    
                    if merged_mask is not None:
                        segmentation = dl.Segmentation(
                            geo=merged_mask,
                            label='gazed_object'
                        )
                        builder.add(annotation_definition=segmentation)
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
    print(f"\n🎉 Step 6: Profit! Upload complete!")
    print(f"📊 Dataset: {dataset.name}")
    print(f"✅ Successfully processed: {success_count}/{len(df_sampled)} samples")
    print(f"📁 Directories covered: {len(directories)}")
    print(f"📈 Average samples per directory: {len(df_sampled)/len(directories):.1f}")


def main():
    dataset_id = "687e43853715c705e1b03bfc"
    dataset = dl.datasets.get(dataset_id=dataset_id)

    folder_root_path = Path(__file__).parent
    extracted_data_path = str(folder_root_path.joinpath("RetailGaze_V2_seg/extracted_data"))
    binaries_path = str(folder_root_path.joinpath("RetailGaze_V2_seg/RetailGaze_V2"))
    
    # Upload ~5 samples from each directory
    upload_data(
        dataset=dataset,
        extracted_data_path=extracted_data_path,
        binaries_path=binaries_path,
        samples_per_dir=5  # Adjust this number as needed
    )


if __name__ == "__main__":
    main()
