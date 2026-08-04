import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("object_classifier.keras")

classes = [
    "book",
    "bottle",
    "pen"
]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    img = cv2.resize(frame,(224,224))

    img = img/255.0

    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img,verbose=0)

    index = np.argmax(prediction)

    confidence = prediction[0][index]*100

    text = f"{classes[index]} ({confidence:.2f}%)"

    cv2.putText(frame,text,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Object Classification",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()

cv2.destroyAllWindows()