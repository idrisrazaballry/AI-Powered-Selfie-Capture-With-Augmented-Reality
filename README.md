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

## 🚀 How to Download & Run (Instant Windows App)

You can run this application instantly without installing Python, setting up an IDE, or managing dependencies.

1. Go to the [Latest Releases Page]().
2. Download the **`AI-Selfie-Camera-Windows.zip`** file under the Assets dropdown menu.
3. Locate the file on your computer, right-click it, and select **Extract All...**.
4. Open the extracted folder and double-click **`main.exe`** to start the application.

---

## 🖥️ Alternative Setup: Run from Source

### Prerequisites
* Python 3.10+ installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com
cd AI-Powered-Selfie-Capture-With-Augmented-Reality
```

### 2. Install Required Packages
```bash
pip install opencv-python mediapipe numpy pillow SpeechRecognition
```

### 3. Run the Application
```bash
python main.py
```

---

## 🔥 Key Features

* **Real-Time Video Analytics**: Live interactive processing using OpenCV webcam streaming feeds.
* **Intelligent Gesture Capturing**: 
  * **Smile Capture**: Uses custom mouth curvature math detection to take a photo.
  * **Blink Capture**: Monitors tracking using Eye Aspect Ratio (EAR) mapping logic.
* **Voice Activation Command System**: Recognizes spoken capture triggers like *"hey selfie"* or *"take selfie"*.
* **Augmented Reality Filters**: Fast real-time matrix transformations to overlay textures:
  * **Face Overlays**: Dynamic tracking for glasses, mustache, and dog face layers.
  * **Post-Processing Shaders**: Live oil paint engine, blurring algorithms, brightness masks, and monochrome matrix transformations.
* **Anti-Spam Optimization**: Integrated state tracking cooldown buffers to block accidental continuous frames.

---

## 🛠️ Built With

* **Python 3.10+** - Core application runtime engine
* **OpenCV** - Matrix frame processing and webcam IO capture
* **MediaPipe Face Mesh** - 468-point landmark tracking model mesh mapping
* **Tkinter** - Multi-threaded responsive desktop graphical dashboard interface
* **SpeechRecognition** - Background audio pipeline processing engine

---

## 📸 Automated Output

Photos are saved directly to the root project runtime workspace folder using structured timestamp formats:
`captured_YYYYMMDD_HHMMSS.jpg`

---

## 🗺️ Future Enhancements

* Cross-platform porting onto mobile system frameworks.
* Biometric authentication profile face recognition login layers.
* Real-time emotional facial mesh analytics for contextual adaptive filters.
