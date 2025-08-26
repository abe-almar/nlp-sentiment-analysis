import os

import keras
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Dense, Dropout, Embedding,
                                     GlobalAveragePooling1D, TextVectorization)

from src.config import label_to_int, names
from src.formatting import format_input


def train_and_save_model(path: str) -> keras.Model:
    """Loads training and validation data from CSV files, preprocesses the text input,
    builds and trains a neural network model for sentiment classification, and saves
    the trained model including the preprocessing pipeline.

    Args:
        path (str): Path to the directory containing 'twitter_training.csv' and
                    'twitter_validation.csv'.

    This function performs the following steps:
    - Loads and labels sentiment data (mapping sentiment categories to integers).
    - Preprocesses text data by formatting inputs using entity-context pairs.
    - Applies text vectorization using Keras' TextVectorization layer.
    - Builds and compiles a Sequential model using embedding and dense layers.
    - Trains the model with early stopping and model checkpoint callbacks.
    - Wraps the trained model with the text preprocessing layer.
    - Saves the full model (preprocessing + trained model) in a local 'models' directory
      as 'full_model_1D.keras'.
    """
    # Load and preprocess data
    train_df = pd.read_csv(
        filepath_or_buffer=path + "/twitter_training.csv", names=names
    )
    val_df = pd.read_csv(
        filepath_or_buffer=path + "/twitter_validation.csv", names=names
    )

    train_df["label"] = train_df["sentiment"].map(label_to_int)
    val_df["label"] = val_df["sentiment"].map(label_to_int)

    train_df["input_text"] = train_df.apply(
        lambda row: format_input(row["entity"], row["text"]), axis=1
    )

    val_df["input_text"] = val_df.apply(
        lambda row: format_input(row["entity"], row["text"]), axis=1
    )

    train_texts = train_df["input_text"].values
    train_labels = train_df["label"].values

    val_texts = val_df["input_text"].values
    val_labels = val_df["label"].values

    # Create tf.data.Dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
        .shuffle(1000)
        .batch(32)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)).batch(32)

    # Text vectorization layer
    text_vectorization = TextVectorization(output_mode="int", output_sequence_length=64)
    text_vectorization.adapt(train_texts)

    # Define and compile model
    vocab_size = len(text_vectorization.get_vocabulary())
    model = Sequential(
        [
            text_vectorization,
            Embedding(vocab_size, 64, mask_zero=True),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # ------------------ Callbacks ------------------
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        os.path.join(models_dir, "full_model_1D.keras"), save_best_only=True
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_accuracy", restore_best_weights=True
    )
    callbacks = [checkpoint_cb, early_stopping_cb]

    # Train model
    model.fit(train_ds, validation_data=val_ds, epochs=6, callbacks=callbacks)

    return model


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    data_path = os.getenv("DATA_FOLDER")
    if not data_path:
        raise ValueError(
            "DATA_FOLDER environment variable is not set in the .env file."
        )

    train_and_save_model(data_path)
