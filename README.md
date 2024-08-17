# Hybrid Detection System

This project implements a **Hybrid Detection System** for detecting number plates, helmets, and triple riding violations on uploaded images. The system leverages pre-trained models hosted on **RoboFlow** for object detection and **TrOCR (Large Printed)** from Hugging Face for text extraction from number plates.

![](/data/project/Flowchart.png)
## Features
- **Number Plate Detection**: Identifies vehicle number plates in images
- **Helmet Detection**: Detects whether a rider is wearing a helmet.
- **Triple Ride Detection**: Identifies instances of triple riding on two-wheelers.
- **Text Extraction**: Extracts text from detected number plates using TrOCR.
- **Authentication** : Allows you to Sign up and Login
- **Admin Panel** : shows databases data on dashboard.

## Prerequisites
- **Python**: Version 3.12.1
- A working internet connection (for accessing RoboFlow models and Hugging Face APIs).
- Optional: GPU support for faster inference (if available).

## Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd hybrid-detection-system
   ```

2. **Create a Virtual Environment**  
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**  
   - On Windows:  
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:  
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**  
   ```bash
   streamlit run app.py
   ```

   This will launch the Streamlit web interface where you can upload images or use a live feed for detection.

## Dependencies
The required Python packages are listed in `requirements.txt`. Key libraries include:
- `streamlit` - For the web interface.
- `roboflow` - For accessing detection models.
- `transformers` - For TrOCR text extraction.
- Other dependencies (e.g., `opencv-python`, `numpy`) for image processing.

## Model Details
- **Object Detection**: Models for number plate, helmet, and triple ride detection are hosted on RoboFlow. Ensure you have the API key
- **Text Extraction**: Uses `TrOCR Large Printed` from Hugging Face for high-accuracy text recognition from number plates.

## Usage
1. Open the Streamlit app in your browser (typically at `http://localhost:8501`).
2. Upload an image
3. View the detection results and extracted text in real-time.

## Project Images
<p float="left">
   <img src="/data/project/Signin.png" width="33%" />
   <img src="/data/project/admin.png" width="33%" />
   <img src="/data/project/admindashboard.png" width="33%" />
</p>

<p float="left">
   <img src="/data/project/numberplate.png" width="33%" />
   <img src="/data/project/helmet.png" width="33%" />
   <img src="/data/project/triple.png" width="33%" />
</p>

## License
This project is licensed under the [MIT License](LICENSE).
