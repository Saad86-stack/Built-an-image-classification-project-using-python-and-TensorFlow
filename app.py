from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Model load
model = tf.keras.models.load_model("object_classifier.keras")

# Classes
classes = ["book", "bottle", "pen"]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = ""

    if request.method == "POST":

        # Uploaded image
        file = request.files["image"]

        # Image save
        file.save("temp.jpg")

        # Image load
        img = image.load_img("temp.jpg", target_size=(224, 224))

        img_array = image.img_to_array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)

        prediction = classes[np.argmax(result)]

        confidence = round(np.max(result) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)