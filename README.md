# 🖐️ Hand Gesture Controlled Mouse with MediaPipe & OpenCV

Control your mouse using just your hand! This Python project uses **MediaPipe**, **OpenCV**, and **PyAutoGUI** to recognize hand gestures through a webcam and perform mouse actions like move, click, scroll, and drag.

## 🚀 Features

- 🎯 **Move Mouse** – Move your cursor by raising your index finger.
- 👈 **Left Click** – Bring index and middle fingers close together.
- 👉 **Right Click** – Bring thumb and index finger close.
- 🖱️ **Double Click** – Perform left click twice quickly.
- ✊ **Drag & Drop** – Pinch with thumb and index finger and move.
- 🖐️ **Scroll** – Show all five fingers and move your hand up or down.

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **PyAutoGUI**
- **NumPy**
- **Math**

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-mouse.git
   cd hand-gesture-mouse
   ```
2. Install dependencies:

   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```
3. How to Run
    ```bash
    python mouse.py
    ```
> Make sure your webcam is on and you have proper lighting.

---

## ✋ Gestures & Modes

| Gesture                           | Action        |
|-----------------------------------|---------------|
| ☝️ Index finger up                | Move cursor   |
| ✌️ Index + Middle close together  | Left click    |
| 👍 Thumb + Index close together    | Right click   |
| 👌 Thumb + Index far apart         | Drag/Drop     |
| 🖐️ All fingers up                 | Scroll        |
| ✌️ Double click fast              | Double click  |

--- 

📷 UI Highlights
- 🟢 Green dot: Cursor movement.

- 🔴 Red dot: Left click.

- 💛 Yellow dot: Double click.

- 🔵 Blue + other colors: Finger landmarks.

---

## 🧠 Behind the Scenes
- Hand detection and tracking powered by MediaPipe.

- Gesture recognition using custom logic.

- Screen mapping and movement smoothing for smooth interaction.

- Real-time gesture mode switching with cooldowns and thresholds.

---

## 📌 To Do
✅ Add double-click detection

✅ Add scroll gesture

⏳ Multi-hand support

⏳ Customize sensitivity and gestures via config

⏳ Support for virtual drawing (Air Canvas mode)

---

## 📸 Demo
(Add GIF or screenshots here)
