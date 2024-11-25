import streamlit as st
import time
from PIL import Image
import torch
import pandas as pd
from utils import (
    load_models, prepare_image, check_image_size, process_batch_photos,
    analyze_single_photo, class_thresholds, generate_excel_filename
)

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
    
    # Move the models to CUDA if available
    if torch.cuda.is_available():
        st.write("Models moved to CUDA")
    else:
        st.write("CUDA is not available. Running on CPU.")
    
    # Model information
    st.header("Model Configuration")
    model_info = """
    Using two models for comprehensive analysis:
    1. Pose Detection Model (tishan_model.pt)
    2. Background Segmentation Model (yolo11x-seg.pt)
    """
    st.info(model_info)
    
    # Guidelines
    st.header("📋 Steps in Determining Passport Photo Validity")
    st.markdown("""
    1. **Use YOLO11** to segment the person out, alongside the tie.
    2. **Create a mask** of the detected person.
    3. **Use OpenCV** to remove the person from the photo.
    4. **Analyze**:
    - **Color Variation**
    - **Hue**
    - **Saturation**
    - **Background Percentage** of the photo by calculating the percentage of black pixels (left behind after removing the detected person) compared to other colors in the photo.

    ---

    ### Current Criteria for Valid Passport Photos

    1. **Color Variation**: Less than **25** (calculated using OpenCV).
    2. **Hue**: Less than **15** (calculated using OpenCV).
    3. **Saturation**: Less than **20** (calculated using OpenCV).
    4. **Background Percentage**: Less than **50%**.

    ---

    ### Future Enhancements

    1. **Detect the pose of the person in the photo** and determine the probability that:
    - The person's body is facing forward.
    - The upper body is present.
    """)

# Main content
st.title("🔍 Passport Photo Analyzer")
st.markdown("*:orange[PPDM Experimental Feature - 2024]*")
st.markdown("""
This tool utilizes YOLOv11 model to analyze your passport photo for both pose and background requirements.
Choose between single photo analysis or batch processing below.
""")

# Load models
@st.cache_resource
def get_models():
    return load_models()

pose_model, seg_model = get_models()

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
            img = prepare_image(uploaded_file)

            if img is not None:
                # Check image size
                uploaded_file.seek(0)  # Reset file pointer
                pil_image = Image.open(uploaded_file)
                #if check_image_size(pil_image):
                if True:  # Removed size check temporarily
                    with st.spinner("Analyzing your photo..."):
                        # Image size validation
                        width, height = pil_image.size
                        aspect_ratio = width / height
                        
                        if not (0.7 <= aspect_ratio <= 1.3):
                            st.warning("⚠️ Image aspect ratio should be close to 1:1 for optimal passport photos")
                        
                        # Analyze photo
                        analysis_result = analyze_single_photo(img, pil_image, pose_model, seg_model)
                        
                        # Display pose analysis
                        st.subheader("🎯 Pose Analysis")
                        if analysis_result['pose_data']:
                            pose_df = pd.DataFrame(analysis_result['pose_data'])
                            st.dataframe(pose_df)
                            if analysis_result['pose_check']:
                                st.success("✅ Pose meets requirements!")
                            else:
                                st.error("❌ Pose does not meet requirements")
                        else:
                            st.warning("No pose elements detected")
                        
                        # Display background analysis
                        st.subheader("🎨 Background Analysis")
                        
                        # Create two columns for image display
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📸 Original Image")
                            st.image(analysis_result['image_np'], use_column_width=True)
                        
                        if analysis_result['background_data']:
                            with col2:
                                st.subheader("🎯 Detected Background")
                                st.image(analysis_result['background_data']['only_rgb'], use_column_width=True)
                            
                            # Calculate processing time
                            processing_time = time.time() - start_time
                            
                            # Display background result
                            if analysis_result['background_data']['is_neutral']:
                                st.success("✅ Background meets passport photo requirements!")
                            else:
                                st.error("❌ Background does not meet passport photo requirements")
                            
                            color_stats = analysis_result['background_data']['color_stats']
                            
                            # Create three columns for stats
                            stat_cols = st.columns(3)
                            
                            # Background Type
                            with stat_cols[0]:
                                st.markdown(f"""
                                <div class="stat-box">
                                    <h3>🎨 Background Type</h3>
                                    <p>{color_stats['background_type'].capitalize()}</p>
                                    <small style="color: #888888;">.</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Brightness
                            with stat_cols[1]:
                                st.markdown(f"""
                                <div class="stat-box">
                                    <h3>💡 Brightness</h3>
                                    <p>{"Good" if color_stats['is_light'] else "Too Dark"}</p>
                                    <small style="color: #888888;">.</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Coverage
                            with stat_cols[2]:
                                st.markdown(f"""
                                <div class="stat-box">
                                    <h3>📏 Coverage</h3>
                                    <p>{color_stats['background_percentage']:.1f}%</p>
                                    <small style="color: #888888;">< 50 is good</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Overall Result
                            st.header("📊 Overall Analysis")
                            # if analysis_result['background_data']['is_neutral'] and analysis_result['pose_check']:
                            if analysis_result['background_data']['is_neutral']:
                                st.success("✅ Photo meets all requirements!")
                            else:
                                st.error("❌ Photo does not meet all requirements")
                            
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
                                        <small style="color: #888888;">< 25 is good</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with metric_cols[1]:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <p class="metric-label">Hue Variation</p>
                                        <p class="metric-value">{color_stats['hue_variation']:.1f}</p>
                                        <small style="color: #888888;">< 15 is good</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with metric_cols[2]:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <p class="metric-label">Saturation Variation</p>
                                        <p class="metric-value">{color_stats['saturation_variation']:.1f}</p>
                                        <small style="color: #888888;">< 20 is good</small>
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
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose passport photos to analyze",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        # Process Batch button
        if st.button("🔄 Process Batch"):
            # Create a progress bar and text
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            with st.spinner("Processing photos..."):
                # Process photos with progress tracking
                excel_file, df = process_batch_photos(uploaded_files, pose_model, seg_model, 
                                                    progress_bar=progress_bar,
                                                    progress_text=progress_text)
                st.session_state.batch_results = (excel_file, df)
                
                # Ensure progress bar reaches 100%
                progress_bar.progress(1.0)
                progress_text.text("✅ Processing complete!")
                st.dataframe(df)
                
                # Provide download button for Excel file
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_file,
                    file_name=generate_excel_filename(len(uploaded_files)),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Show previous results if they exist
        elif 'batch_results' in st.session_state:
            excel_file, df = st.session_state.batch_results
            st.success(f"✅ Previously processed {len(uploaded_files)} photos")
            st.dataframe(df)
            
            # Provide download button for Excel file
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_file,
                file_name="passport_photo_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        # Reset session state when no files are uploaded
        if 'batch_results' in st.session_state:
            st.session_state.batch_results = None
            
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
  <p>Developed with ❤️ by <a href="https://www.linkedin.com/in/muhammad-faris-ahmad-faiz-ab9b35212/" target="_blank">Faris</a> and <a href="https://www.linkedin.com/in/tishanprakash-sivarajah-32362820a/" target="_blank">Tishan</a> for PPDM</p>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)
