# Passport Photo Background Analyzer

This application uses AI-powered image segmentation to analyze passport photo backgrounds, ensuring they meet standard requirements for official documents. It checks if the background is neutral, light-colored, and suitable for passport photos.

## Features

- Person segmentation using YOLO
- Background color analysis
- Detailed statistics about background properties
- Real-time visual feedback
- Support for common image formats (JPG, JPEG, PNG)

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a passport photo using the file uploader

4. The application will:
   - Display the original image
   - Show the extracted background
   - Provide analysis results including:
     - Whether the background is suitable for a passport photo
     - Average RGB values
     - Color variation
     - Background percentage

## Background Analysis Algorithm

The application uses a sophisticated algorithm to determine if a background is neutral and suitable for passport photos. Here's how it works:

1. **Person Segmentation**:
   - YOLO model identifies and segments the person from the image
   - This creates a mask where the person was, filled with black pixels (RGB: [0,0,0])

2. **Background Isolation**:
   - The algorithm specifically excludes the black pixels (created from person removal) from the analysis
   - Only considers actual background pixels (non-black) for color analysis
   - Creates a mask for pixels with values above [1,1,1] in RGB

3. **Color Analysis**:
   - Converts background pixels to HSV color space for better color analysis
   - Calculates several metrics:
     - Mean color (average RGB values)
     - Color variation (standard deviation of colors)
     - Background percentage (ratio of non-black pixels to total image size)
     - Lightness check (average RGB > 180)

4. **Neutrality Determination**:
   The background is considered neutral if it meets all these criteria:
   - Hue variation < 20 (consistent color)
   - Saturation variation < 30 (not too colorful)
   - Color variation < 20 (uniform background)
   - Is light colored (mean RGB > 180)

## Model Information

The application uses YOLO11 for segmentation, specifically the `yolo11x-seg.pt` model, which is optimized for person segmentation. The model files should be placed in the `models/` directory.

## Directory Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── models/            # Directory containing YOLO models
│   └── yolo11x-seg.pt # YOLO segmentation model
└── photos/            # Sample photos directory
