from ultralytics import YOLO
import cv2
from datetime import datetime
from PIL import Image
import numpy as np
import io
import torch
import pandas as pd

# Fixed class thresholds for pose detection
class_thresholds = {
    "face_forward": 0.8411,
    "body_forward": 0.3784,
    "Upper_body": 0.5709
}

def generate_excel_filename(num_files):
    """Generate Excel filename with current timestamp and number of files processed"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"passport_analysis_{current_time}_{num_files}files.xlsx"

def load_models():
    """Load and configure YOLO models for pose and segmentation"""
    pose_model = YOLO("models/tishan_model.pt")
    seg_model = YOLO("models/yolo11x-seg.pt")
    if torch.cuda.is_available():
        pose_model.to('cuda')
        seg_model.to('cuda')
    return pose_model, seg_model

def analyze_pose(image_array, pose_model):
    """Analyze pose in the image using YOLO model"""
    results = pose_model(image_array)
    pose_data = []
    
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = pose_model.names[class_id]
                confidence = float(box.conf[0])
                if confidence > class_thresholds.get(class_name, 0):
                    pose_data.append({
                        'Class': class_name,
                        'Confidence': confidence
                    })
    
    return pose_data

def is_neutral_color(image):
    """Analyze if the background color is neutral (white or light blue)"""
    # Create a mask for non-black pixels (actual background)
    non_black_mask = cv2.inRange(image, np.array([1, 1, 1]), np.array([255, 255, 255]))
    
    if cv2.countNonZero(non_black_mask) == 0:
        return False, {
            'mean_color': [0, 0, 0],
            'color_variation': 0,
            'is_light': False,
            'background_percentage': 0,
            'background_type': 'none'
        }
    
    masked_image = cv2.bitwise_and(image, image, mask=non_black_mask)
    background_pixels = masked_image[masked_image != 0].reshape(-1, 3)
    
    if len(background_pixels) == 0:
        return False, {
            'mean_color': [0, 0, 0],
            'color_variation': 0,
            'is_light': False,
            'background_percentage': 0,
            'background_type': 'none'
        }
    
    mean_color = np.mean(background_pixels, axis=0)
    hsv_pixels = cv2.cvtColor(background_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv_pixels.reshape(-1, 3)
    
    h_std = np.std(hsv_pixels[:, 0])
    s_std = np.std(hsv_pixels[:, 1])
    color_variation = np.std(background_pixels, axis=0).mean()
    
    total_pixels = image.shape[0] * image.shape[1]
    background_percentage = (cv2.countNonZero(non_black_mask) / total_pixels) * 100
    
    HUE_THRESHOLD = 15
    SATURATION_THRESHOLD = 20
    COLOR_VARIATION_THRESHOLD = 25
    BRIGHTNESS_THRESHOLD = 150
    
    is_light = np.mean(mean_color) > BRIGHTNESS_THRESHOLD
    
    mean_rgb = [mean_color[2], mean_color[1], mean_color[0]]
    
    is_blue = (mean_rgb[2] > mean_rgb[0] + 20 and 
              mean_rgb[2] > mean_rgb[1] + 20 and 
              mean_rgb[2] > 150)
    
    is_white = (abs(mean_rgb[0] - mean_rgb[1]) < 20 and
               abs(mean_rgb[1] - mean_rgb[2]) < 20 and
               abs(mean_rgb[0] - mean_rgb[2]) < 20 and
               np.mean(mean_rgb) > 180)
    
    if is_blue:
        background_type = 'blue'
    elif is_white:
        background_type = 'white'
    else:
        background_type = 'other'
    
    is_neutral = ((is_blue or is_white) and 
                 color_variation < COLOR_VARIATION_THRESHOLD and
                 h_std < HUE_THRESHOLD and 
                 s_std < SATURATION_THRESHOLD and
                 background_percentage < 50) 
    
    return is_neutral, {
        'mean_color': mean_color,
        'color_variation': color_variation,
        'is_light': is_light,
        'background_percentage': background_percentage,
        'hue_variation': h_std,
        'saturation_variation': s_std,
        'background_type': background_type
    }

def check_image_size(image):
    """Check if image meets the required dimensions"""
    try:
        width, height = image.size
        return width == 1050 and height == 1500
    except Exception as e:
        return False

def prepare_image(uploaded_file):
    """Prepare uploaded image for processing"""
    # Read the file into bytes
    bytes_data = uploaded_file.read()
    
    # Convert bytes to numpy array using cv2
    nparr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB (cv2 loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def process_batch_photos(uploaded_files, pose_model, seg_model):
    """Process multiple photos in batch mode"""
    results = []
    
    for file in uploaded_files:
        try:
            # Prepare image
            img = prepare_image(file)
            
            # Open image for size check
            file.seek(0)  # Reset file pointer
            pil_image = Image.open(file)
            
            # Check size
            is_correct_size = check_image_size(pil_image)
            
            # Convert to BGR for processing
            image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Analyze pose
            pose_data = analyze_pose(img, pose_model)
            pose_check = any(pd['Class'] in class_thresholds and pd['Confidence'] >= class_thresholds[pd['Class']] for pd in pose_data)
            
            # Process with segmentation model
            seg_results = seg_model(image_bgr, classes=[0, 27])
            
            # Initialize background check result
            is_neutral_background = False
            background_percentage = 0
            
            # Process segmentation results
            for r in seg_results:
                background_mask = np.ones(image_bgr.shape[:2], np.uint8) * 255
                
                for c in r:
                    contour = c.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(background_mask, [contour], -1, 0, cv2.FILLED)
                
                background_mask_3ch = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
                background_only = cv2.bitwise_and(image_bgr, background_mask_3ch)
                
                # Check background
                is_neutral_background, background_stats = is_neutral_color(background_only)
                background_percentage = background_stats['background_percentage']
            
            # Pose checks
            face_forward_check = any(pd['Class'] == 'face_forward' and pd['Confidence'] >= class_thresholds['face_forward'] for pd in pose_data)
            body_forward_check = any(pd['Class'] == 'body_forward' and pd['Confidence'] >= class_thresholds['body_forward'] for pd in pose_data)
            upper_body_check = any(pd['Class'] == 'Upper_body' and pd['Confidence'] >= class_thresholds['Upper_body'] for pd in pose_data)
            
            # Background Percentage as boolean (True if below 50%, False if above 50%)
            background_percentage_check = background_percentage < 50
            
            # Overall check (all conditions must be met)
            overall_pass = is_correct_size and is_neutral_background and pose_check
            
            results.append({
                'Filename': file.name,
                'Passport Size': is_correct_size,
                'Neutral Background': is_neutral_background,
                'Background Percentage': background_percentage_check,  # Changed to use boolean check
                'Face Forward': face_forward_check,
                'Body Forward': body_forward_check,
                'Upper Body': upper_body_check,
                'Overall': overall_pass
            })
            
        except Exception as e:
            results.append({
                'Filename': file.name,
                'Passport Size': False,
                'Neutral Background': False,
                'Background Percentage': False,
                'Face Forward': False,
                'Body Forward': False,
                'Upper Body': False,
                'Overall': False
            })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    excel_file = io.BytesIO()
    df.to_excel(excel_file, index=False, engine='openpyxl')
    excel_file.seek(0)
    
    return excel_file, df

def analyze_single_photo(img, pil_image, pose_model, seg_model):
    """Analyze a single photo and return all relevant data"""
    # Convert PIL Image to numpy array
    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Analyze pose
    pose_data = analyze_pose(img, pose_model)
    pose_check = any(pd['Class'] in class_thresholds and pd['Confidence'] >= class_thresholds[pd['Class']] for pd in pose_data)
    
    # Process with segmentation model
    results = seg_model(image_bgr, classes=[0, 27])
    
    # Process segmentation results
    background_data = None
    for r in results:
        background_mask = np.ones(image_bgr.shape[:2], np.uint8) * 255
        
        for c in r:
            contour = c.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(background_mask, [contour], -1, 0, cv2.FILLED)
        
        background_mask_3ch = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
        background_only = cv2.bitwise_and(image_bgr, background_mask_3ch)
        background_only_rgb = cv2.cvtColor(background_only, cv2.COLOR_BGR2RGB)
        
        # Analyze background color
        is_neutral, color_stats = is_neutral_color(background_only)
        
        background_data = {
            'mask': background_mask_3ch,
            'only': background_only,
            'only_rgb': background_only_rgb,
            'is_neutral': is_neutral,
            'color_stats': color_stats
        }
    
    return {
        'pose_data': pose_data,
        'pose_check': pose_check,
        'background_data': background_data,
        'image_np': image_np,
        'image_bgr': image_bgr
    }
