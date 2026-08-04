# 📷 Object Classification and Recognition Using Images and Live Webcam

This project is an AI-based Object Classification System developed using **Python, TensorFlow, Flask, and OpenCV**.

The model can classify three objects:

- 📖 Book
- 🍼 Bottle
- 🖊️ Pen

Users can classify objects in two ways:

- Upload an image through the web application.
- Detect objects using a live webcam.

---

# 🚀 Features

- Image Upload Classification
- Live Webcam Object Detection
- Deep Learning Model (TensorFlow/Keras)
- Simple Flask Web Interface
- Real-time Prediction
- Confidence Score Display

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Flask
- OpenCV
- NumPy
- HTML
- CSS

---

# 📂 Project Structure

```
Project
│
├── train/
│   ├── book/
│   ├── bottle/
│   └── pen/
│
├── test/
│   ├── book/
│   ├── bottle/
│   └── pen/
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── camera.py
├── predict.py
├── train.py
├── object_classifier.keras
├── README.md
```

---

# 📖 Project Workflow

```
Dataset
      ↓
Model Training
      ↓
Saved Model (.keras)
      ↓
Image Upload / Live Webcam
      ↓
Object Prediction
      ↓
Prediction + Confidence
```

---

# 📸 Objects Classified

- Book
- Bottle
- Pen

---

# ▶️ How to Run

### 1. Install Required Libraries

```bash
pip install tensorflow flask opencv-python numpy
```

### 2. Train Model

```bash
python train.py
```

### 3. Run Website

```bash
python app.py
```

Open Browser:

```
http://127.0.0.1:5000
```

### 4. Run Webcam Detection

```bash
python camera.py
```

---

# 📊 Model Output

The system displays:

- Predicted Object
- Confidence Score

Example:

```
Prediction : Bottle

Confidence : 98.75%
```

---

# 🎯 Applications

- Smart Object Detection
- Educational Projects
- AI Learning
- Computer Vision
- Real-Time Object Recognition

---

# 👨‍💻 Developed By

Saad Pathan

Diploma in Artificial Intelligence & Machine Learning (AIML)

---

# ⭐ Future Improvements

- More Object Classes
- Mobile Application
- Voice Output
- Cloud Deployment
- Better UI De-sign

---

## 📄 License

This project is developed for educational purposes.
