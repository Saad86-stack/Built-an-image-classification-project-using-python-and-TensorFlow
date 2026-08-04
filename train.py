import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image resize , normalize
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training dataset load
train_data = train_datagen.flow_from_directory(
    "train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Testing dataset load
test_data = test_datagen.flow_from_directory(
    "test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print("Classes:", train_data.class_indices)

#//////////////////////////////////////////////////////

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(3, activation='softmax')
])

#///////////////////////////////////////////////////////

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

model.save("object_classifier.keras")