import os
import tensorflow as tf
import tensorflow_datasets as tfds

# ============================================================
# CHARACTER RECOGNITION USING CNN
# ============================================================

print("=" * 60)
print("CHARACTER RECOGNITION USING CONVOLUTIONAL NEURAL NETWORK")
print("=" * 60)

# -----------------------------
# Configuration
# -----------------------------

IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 8
NUM_CLASSES = 26

MODEL_PATH = "model/character_cnn.keras"

# -----------------------------
# Load EMNIST Letters
# -----------------------------

print("\n[1/5] Loading EMNIST Letters dataset...")

(train_ds, test_ds), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

print("Dataset loaded successfully!")
print(
    "Training samples:",
    ds_info.splits["train"].num_examples
)
print(
    "Testing samples:",
    ds_info.splits["test"].num_examples
)

# -----------------------------
# Preprocessing
# -----------------------------

print("\n[2/5] Preprocessing dataset...")

def preprocess(image, label):

    # Convert image to float32
    image = tf.cast(image, tf.float32)

    # Normalize pixel values
    image = image / 255.0

    # EMNIST labels are 1-26
    # Convert them to 0-25
    label = label - 1

    return image, label


train_ds = train_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(10000)

train_ds = train_ds.batch(
    BATCH_SIZE
).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.batch(
    BATCH_SIZE
).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# Build CNN
# -----------------------------

print("\n[3/5] Building CNN model...")

model = tf.keras.Sequential([

    # Input Layer
    tf.keras.layers.Input(
        shape=(28, 28, 1)
    ),

    # Convolution Block 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    # Convert feature maps to vector
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Reduce overfitting
    tf.keras.layers.Dropout(
        0.3
    ),

    # Output: 26 English characters
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# -----------------------------
# Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN architecture:")
model.summary()

# -----------------------------
# Training
# -----------------------------

print("\n[4/5] Training CNN model...")
print("Please wait...\n")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# Evaluation
# -----------------------------

print("\n[5/5] Evaluating model...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

# -----------------------------
# Save Model
# -----------------------------

os.makedirs(
    "model",
    exist_ok=True
)

model.save(
    MODEL_PATH
)

print("\nModel saved successfully!")
print(
    f"Model location: {MODEL_PATH}"
)

print("=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)