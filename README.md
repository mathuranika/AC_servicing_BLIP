# AC Servicing BLIP

An AI-powered video captioning system for AC (Air Conditioning) installation and servicing videos.  This application automatically detects scene changes in videos and generates descriptive captions using the BLIP (Bootstrapping Language-Image Pre-training) model.

## Overview

This project combines computer vision and natural language processing to automatically caption AC installation and servicing videos. It detects scene changes using OpenCV, generates captions for each scene using the BLIP model, and burns the captions into the video as subtitles.

## Features

- **Automatic Scene Detection**: Uses OpenCV to detect scene changes based on frame differences
- **AI-Powered Captioning**: Leverages Salesforce's BLIP model for image captioning
- **Subtitle Generation**:  Creates SRT subtitle files with timestamps
- **Video Processing**: Burns subtitles directly into the video using FFmpeg
- **User-Friendly Interface**: Built with Streamlit for easy video upload and processing
- **Download Support**: Download the captioned video directly from the web interface

## Technology Stack

- **Python 3.x**
- **Streamlit**:  Web application framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for BLIP model
- **OpenCV**: Computer vision library for scene detection
- **FFmpeg**: Video processing and subtitle burning
- **Pillow (PIL)**: Image processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mathuranika/AC_servicing_BLIP.git
cd AC_servicing_BLIP
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure FFmpeg is installed on your system: 
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

### Running the Web Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload a video file (supported formats: MP4, MOV, MKV)

4. Wait for the processing to complete

5. View the generated captions and download the captioned video

### Testing the BLIP Model

Run the test script to verify the BLIP model is working: 

```bash
python test_blip.py
```

Note: You'll need a test image named `test.jpeg` in the project directory. 

## How It Works

1. **Video Upload**: User uploads an AC installation/servicing video through the Streamlit interface

2. **Scene Detection**: The system analyzes the video frame-by-frame using OpenCV to detect scene changes based on: 
   - Frame difference threshold (default: 30.0)
   - Minimum scene length (default: 15 frames)

3. **Frame Extraction**: Extracts a representative frame from each detected scene

4. **Caption Generation**:  Each frame is passed through the BLIP model to generate descriptive captions

5. **Subtitle Creation**: Generates an SRT subtitle file with timestamps for each scene

6. **Video Processing**: Uses FFmpeg to burn the subtitles into the original video

7. **Output**: Returns a captioned video ready for download

## Configuration

You can adjust scene detection parameters in `app.py`:

```python
detect_scenes_opencv(
    video_path,
    output_dir,
    diff_threshold=30.0,      # Sensitivity for scene changes
    min_scene_length=15       # Minimum frames between scenes
)
```

## Project Structure

```
AC_servicing_BLIP/
├── app.py              # Main Streamlit application
├── test_blip.py        # BLIP model testing script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

```
torch
transformers
streamlit
timm
opencv-python
moviepy
scenedetect
```

## Model Information

This project uses the **Salesforce BLIP** (Bootstrapping Language-Image Pre-training) model:
- Model:  `Salesforce/blip-image-captioning-base`
- Task: Image-to-text captioning
- Framework: Hugging Face Transformers

## Limitations

- Processing time depends on video length and hardware capabilities
- Runs on CPU by default (GPU support can be added for faster processing)
- Scene detection accuracy depends on video quality and scene variation
- Caption accuracy is limited by the BLIP model's training data

## Future Enhancements

- [ ] GPU acceleration support
- [ ] Custom scene detection parameters in UI
- [ ] Support for batch video processing
- [ ] Fine-tuned model for AC-specific terminology
- [ ] Multi-language caption support
- [ ] Video preview with timeline
- [ ] Advanced subtitle styling options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for use.  Please check with the repository owner for specific licensing terms. 

## Acknowledgments

- [Salesforce BLIP Model](https://github.com/salesforce/BLIP)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)

