# VERO 2.0: Smart Glasses for Blind People 
Visually impaired individuals face significant challenges in navigating their surroundings safely, especially in unfamiliar environments. Traditional mobility aids like white canes have limitations, as they primarily detect obstacles at ground level but fail to identify upper-body hazards. Guide dogs, while helpful, are expensive and require extensive training. High-tech solutions such as smart navigation systems exist but are often costly and not widely accessible. There is a need for an affordable, hands-free, and intuitive solution that enhances mobility without reliance on complex technology. 

Our smart glasses detect nearby obstacles and provide feedback to alert the user. This hands-free solution is lightweight, easy to use, and helps detect upper-body hazards that traditional mobility aids miss. Unlike expensive smart canes or guide dogs, our design is affordable and accessible, making independent navigation safer and more intuitive.

**Key Benefits:**
- Real-time obstacle detection with simple audio alerts
- No learning curve, just wear and use like regular sunglasses
- Cost-effective alternative to high-tech mobility aids


## Features of our project
- **Real-Time Object Detection**
  - YOLOv8n model (95% mAP50 on COCO)
  - Processes at ~50 FPS on Raspberry Pi 5
- **Instant Audio Feedback**
  - Offline text-to-speech (pyttsx3)
  - Bluetooth audio output
- **Edge Optimized**
  - Autonomous operation (no external monitor)
  - Low-power consumption design
 
## Hardware 
| Component               | Model                          | Key Specs                          |
|-------------------------|--------------------------------|------------------------------------|
| Microcontroller         | Raspberry Pi 5                 | 8GB RAM, Quad-core 2.4GHz         |
| Camera                  | Raspberry Pi Camera Module 3   | 12MP, Autofocus                   |
| Audio Output            | Generic Bluetooth Earpiece     | Hands-free operation              |

## Software 
**Core Stack:**
- **Computer Vision:** OpenCV + Picamera2
- **AI Inference:** YOLOv8n + TensorFlow Lite
- **Audio:** pyttsx3 (Text-to-Speech)

**Dependencies:**
Python 3.8+
opencv-python>=4.8
ultralytics>=8.0
pyttsx3>=2.90

**Future Work**
- Train custom model to attain 90%+ accuracy
- Add OCR for text reading

**Contributors**
[Amey Jawale](https://github.com/ameyjawale)
[Pushkar Sanap](link)
[Ritvik Jeeda](link)
