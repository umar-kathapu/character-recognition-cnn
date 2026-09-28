import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# Load EMNIST Letters dataset
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

print("Dataset loaded successfully!")
print("Training samples:", ds_info.splits["train"].num_examples)
print("Testing samples:", ds_info.splits["test"].num_examples)