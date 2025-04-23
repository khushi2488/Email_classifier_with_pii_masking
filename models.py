"""
Model definition, training, and evaluation for email classification using LSTM.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def build_model(vocab_size, num_classes):
    """
    Build a Sequential LSTM-based neural network for classification.

    Args:
        vocab_size (int): The size of the vocabulary (input dimension for the Embedding layer).
        num_classes (int): The number of output classes for classification.

    Returns:
        model (tensorflow.keras.Model): A compiled Keras model.
    """
    model = Sequential([
        # Embedding layer to map words to vector representations
        Embedding(input_dim=vocab_size, output_dim=128, input_length=128),

        # Bidirectional LSTM layer with dropout for regularization
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)),

        # Dense layer with ReLU activation for additional learning capacity
        Dense(128, activation='relu'),

        # Dropout layer to prevent overfitting
        Dropout(0.5),

        # Output layer with softmax activation for classification
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer=Adam(0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train):
    """
    Train the model using the provided training data.

    Args:
        model (tensorflow.keras.Model): The model to train.
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.

    Returns:
        model (tensorflow.keras.Model): The trained model.
    """
    # Early stopping callback to stop training when validation loss stops improving
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Learning rate scheduler to reduce the learning rate when the validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6)

    # Train the model with validation split and callbacks for early stopping and learning rate adjustment
    model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32,
              callbacks=[early_stop, lr_scheduler], verbose=2)

    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model on the test data and print performance metrics.

    Args:
        model (tensorflow.keras.Model): The trained model to evaluate.
        X_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): Test data labels.
        label_encoder (sklearn.preprocessing.LabelEncoder): Label encoder to map class indices to labels.

    Prints:
        - Test accuracy
        - Classification report
        - Confusion matrix
    """
    # Evaluate the model on test data
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Predict the labels for the test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Print classification report and confusion matrix
    print("\nClassification Report:\n", classification_report(y_true, y_pred,
                                                              target_names=label_encoder.classes_))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
