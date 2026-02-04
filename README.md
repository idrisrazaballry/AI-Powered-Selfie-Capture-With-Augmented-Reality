AI Selfie Camera with Augmented Reality

Project Overview
The "AI Selfie Camera with Augmented Reality" is a desktop application developed in Python that facilitates "hands-free selfie capturing" through the use of "facial gestures and voice commands". This application incorporates computer vision, facial landmark detection, and real-time AR filters to deliver an engaging and intelligent selfie experience.

The system accommodates "smile capture", "blink capture", "voice-triggered capture", and "manual capture", while also applying various AR filters in real time.

Key Features
- Real-time webcam feed utilizing OpenCV  
- Smile detection based on mouth curvature ratio  
- Blink detection employing Eye Aspect Ratio (EAR)  
- Voice command capture (“hey sefi”, “take selfie”)  
- Augmented Reality filters:
  - Glasses
  - Mustache
  - Dog face
  - Oil paint
  - Brighten
  - Black & White
  - Blur
- Live preview of the most recently captured selfie  
- Cooldown mechanism to prevent multiple captures  
- User-friendly GUI designed with Tkinter  

Technologies Used
- Python 3.10+
- OpenCV
- MediaPipe Face Mesh
- Tkinter
- SpeechRecognition
- NumPy
- Pillow (PIL)

Capture Modes
- Smile -> Captures a selfie upon detection of a smile.
- Blink -> Captures a selfie during an intentional blink.
- Voice -> Captures a selfie via voice commands.
- Manual -> Capture initiated by button press.
- Disable -> Deactivates detection.

Project Structure
AI-Selfie-Camera/
├── main.py
├── glass.py
├── mustache.py
├── dog_filter.py
├── oil_paint.py
├── brightener.py
├── bw.py
├── blur.py
├── README.md

How to Run
Install dependencies
```bash
pip install opencv-python mediapipe numpy pillow SpeechRecognition
```

Run the application
```bash
python main.py
```

Output
Captured images are automatically saved in the project directory in the following format:
```
captured_YYYYMMDD_HHMMSS.jpg
```

Future Enhancements
- Support for mobile applications
- Face recognition for login
- Filters based on emotional recognition
- Integration for cloud uploads
