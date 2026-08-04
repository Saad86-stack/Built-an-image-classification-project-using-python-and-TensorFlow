import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Trained model load 
model = tf.keras.models.load_model("object_classifier.keras")

# Test image path
img_path = "test/pen/0ded849413cd1dfc_jpg.rf.a1a72e4a1114b5093032b23c6206ecf5.jpg" 

# Image load 
img = image.load_img(img_path, target_size=(224, 224))

# Image  array  convert 
img_array = image.img_to_array(img)

# Normalize (0-255 -> 0-1)
img_array = img_array / 255.0

# Batch dimension add 
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

# Classes
classes = ["book", "bottle", "pen"]

# Highest probability class
predicted_class = classes[np.argmax(prediction)]

print("Prediction :", predicted_class)
print("Confidence :", np.max(prediction) * 100, "%")