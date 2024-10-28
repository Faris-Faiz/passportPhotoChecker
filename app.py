import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import io
import time
import os
import torch
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Passport Photo Analyzer",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stat-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #262730;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .stat-box h3 {
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .stat-box p {
        color: #FFFFFF;
        font-size: 1.2rem;
        margin: 0;
    }
    .metric-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        color: #FFFFFF;
        font-weight: bold;
    }
    .metric-value {
        color: #00FF00;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📷 Settings & Info")
    
    # Move the model to CUDA
    if torch.cuda.is_available():
        #model.to('cuda')
        st.write("Model moved to CUDA")
    else:
        st.write("CUDA is not available. Running on CPU.")
    
    # Model selection with better description
    st.header("Model Configuration")
    model_info = """
    Choose the model based on your needs:
    - x: Highest accuracy, slower
    - m: Balanced performance
    - l: Faster, slightly lower accuracy
    """
    st.info(model_info)
    
    available_models = ["yolo11x-seg.pt", "yolo11l-seg.pt", "yolo11m-seg.pt"]
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        help="Choose the model for person segmentation"
    )
    
    # Guidelines
    st.header("📋 Photo Guidelines")
    st.markdown("""
    **Passport Photo Requirements:**
    - Neutral white or light blue background
    - Subject centered and facing forward
    - No shadows on background
    - Proper lighting on face
    - Neutral facial expression
    - No accessories (except required ones)
    """)

# Main content
st.title("🔍 Passport Photo Background Analyzer")
st.markdown("""
This tool analyzes your passport photo's background to ensure it meets official requirements.
Choose between single photo analysis or batch processing below.
""")

# Load model
@st.cache_resource
def load_model(model_path):
    return YOLO(f"models/{model_path}")

model = load_model(selected_model)

if torch.cuda.is_available():
    model.to('cuda')

def is_neutral_color(image):
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
    
    mean_hsv = np.mean(hsv_pixels, axis=0)
    
    HUE_THRESHOLD = 20
    SATURATION_THRESHOLD = 30
    COLOR_VARIATION_THRESHOLD = 20
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
                 s_std < SATURATION_THRESHOLD)
    
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
    try:
        # Get the dimensions of the image
        width, height = image.size
        
        # Check if the dimensions match the required size
        return width == 1050 and height == 1500
    except Exception as e:
        return False

def process_batch_photos(uploaded_files):
    results = []
    
    for file in uploaded_files:
        try:
            # Open image
            pil_image = Image.open(file)
            
            # Check size
            is_correct_size = check_image_size(pil_image)
            
            # Convert to numpy array for processing
            image_np = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Process with YOLO model
            yolo_results = model(image_bgr, classes=[0, 27])
            
            # Initialize background check result
            is_neutral_background = False
            
            # Process results
            for r in yolo_results:
                background_mask = np.ones(image_bgr.shape[:2], np.uint8) * 255
                
                for c in r:
                    contour = c.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(background_mask, [contour], -1, 0, cv2.FILLED)
                
                background_mask_3ch = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
                background_only = cv2.bitwise_and(image_bgr, background_mask_3ch)
                
                # Check background
                is_neutral_background, _ = is_neutral_color(background_only)
            
            # Overall check (all conditions must be met)
            overall_pass = is_correct_size and is_neutral_background
            
            results.append({
                'Filename': file.name,
                'Passport Size': is_correct_size,
                'Neutral Background': is_neutral_background,
                'Overall': overall_pass
            })
            
        except Exception as e:
            results.append({
                'Filename': file.name,
                'Passport Size': False,
                'Neutral Background': False,
                'Overall': False
            })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    excel_file = io.BytesIO()
    df.to_excel(excel_file, index=False, engine='openpyxl')
    excel_file.seek(0)
    
    return excel_file, df

# Tab selection
tab1, tab2 = st.tabs(["Single Photo Analysis", "Batch Processing"])

with tab1:
    # Single photo upload and analysis
    uploaded_file = st.file_uploader(
        "Choose a passport photo to analyze",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear front-facing photo with a solid background",
        key="single_upload"
    )

    if uploaded_file is not None:
        try:
            start_time = time.time()
            
            # Check image size
            pil_image = Image.open(uploaded_file)
            if check_image_size(pil_image):
                with st.spinner("Analyzing your photo..."):
                    # Image size validation
                    width, height = pil_image.size
                    aspect_ratio = width / height
                    
                    if not (0.7 <= aspect_ratio <= 1.3):
                        st.warning("⚠️ Image aspect ratio should be close to 1:1 for optimal passport photos")
                    
                    # Convert PIL Image to numpy array
                    image_np = np.array(pil_image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # Create two columns for image display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📸 Original Image")
                        st.image(image_np, use_column_width=True)
                    
                    # Make prediction
                    results = model(image_bgr, classes=[0, 27])
                    
                    # Process results
                    for r in results:
                        background_mask = np.ones(image_bgr.shape[:2], np.uint8) * 255
                        
                        for c in r:
                            contour = c.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
                            cv2.drawContours(background_mask, [contour], -1, 0, cv2.FILLED)
                        
                        background_mask_3ch = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
                        background_only = cv2.bitwise_and(image_bgr, background_mask_3ch)
                        background_only_rgb = cv2.cvtColor(background_only, cv2.COLOR_BGR2RGB)
                        
                        with col2:
                            st.subheader("🎯 Detected Background")
                            st.image(background_only_rgb, use_column_width=True)
                        
                        # Analyze background color
                        is_neutral, color_stats = is_neutral_color(background_only)
                        
                        # Calculate processing time
                        processing_time = time.time() - start_time
                        
                        # Display overall result first
                        st.header("📊 Analysis Results")
                        if is_neutral:
                            st.success("✅ Background meets passport photo requirements!")
                        else:
                            st.error("❌ Background does not meet passport photo requirements")
                        
                        # Create three columns for stats
                        stat_cols = st.columns(3)
                        
                        # Background Type
                        with stat_cols[0]:
                            st.markdown(f"""
                            <div class="stat-box">
                                <h3>🎨 Background Type</h3>
                                <p>{color_stats['background_type'].capitalize()}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Brightness
                        with stat_cols[1]:
                            st.markdown(f"""
                            <div class="stat-box">
                                <h3>💡 Brightness</h3>
                                <p>{"Good" if color_stats['is_light'] else "Too Dark"}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Coverage
                        with stat_cols[2]:
                            st.markdown(f"""
                            <div class="stat-box">
                                <h3>📏 Coverage</h3>
                                <p>{color_stats['background_percentage']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed Analysis Expander
                        with st.expander("See Detailed Analysis"):
                            st.markdown("### 🔍 Technical Details")
                            
                            # RGB Values
                            st.markdown("""
                            <div class="metric-container">
                                <p class="metric-label">RGB Color Values</p>
                                <p class="metric-value">""")
                            rgb_values = f"R={color_stats['mean_color'][2]:.1f}, G={color_stats['mean_color'][1]:.1f}, B={color_stats['mean_color'][0]:.1f}"
                            st.code(rgb_values)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Color Variation Metrics
                            st.markdown("<p class='metric-label'>Color Uniformity Metrics</p>", unsafe_allow_html=True)
                            metric_cols = st.columns(3)
                            
                            with metric_cols[0]:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <p class="metric-label">Color Variation</p>
                                    <p class="metric-value">{color_stats['color_variation']:.1f}</p>
                                    <small style="color: #888888;">< 20 is good</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_cols[1]:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <p class="metric-label">Hue Variation</p>
                                    <p class="metric-value">{color_stats['hue_variation']:.1f}</p>
                                    <small style="color: #888888;">< 20 is good</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_cols[2]:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <p class="metric-label">Saturation Variation</p>
                                    <p class="metric-value">{color_stats['saturation_variation']:.1f}</p>
                                    <small style="color: #888888;">< 30 is good</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Processing Information
                            st.info(f"⚡ Processing Time: {processing_time:.2f} seconds")
            else:
                st.error("Uploaded Image Is Not 1050px by 1500px.", icon="🚨")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please ensure you've uploaded a valid image file.")

with tab2:
    st.header("📦 Batch Photo Processing")
    st.markdown("""
    Upload multiple passport photos for batch analysis. The tool will generate an Excel report 
    with analysis results for all photos.
    """)
    
    uploaded_files = st.file_uploader(
        "Choose passport photos to analyze",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        with st.spinner("Processing photos..."):
            excel_file, df = process_batch_photos(uploaded_files)
            
            # Display results in the UI
            st.success(f"✅ Processed {len(uploaded_files)} photos")
            st.dataframe(df)
            
            # Provide download button for Excel file
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_file,
                file_name="passport_photo_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        # Show example/placeholder
        st.info("👆 Upload photos to begin analysis")
        
        # Display guidelines in a more visual way
        st.markdown("""
        ### 📝 What makes a good passport photo?
        
        1. **Background**
            - Solid white or light blue
            - No patterns or shadows
            - Even lighting
            
        2. **Subject Position**
            - Centered in frame
            - Eyes at 2/3 height of image
            - Full head and top of shoulders visible
            
        3. **Technical Requirements**
            - High resolution
            - Sharp focus
            - No digital alterations
        """)

footer_html = """<div style='text-align: center;'>
  <p>Developed with ❤️ by <a href="https://www.linkedin.com/in/muhammad-faris-ahmad-faiz-ab9b35212/" target="_blank">Faris</a> and <a href="https://www.linkedin.com/in/tishanprakash-sivarajah-32362820a/" target="_blank">Tishan</a></p>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)
