import re
import nltk
import numpy as np
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and encoders only once
model = tf.keras.models.load_model("saved_model/optimized_email-fc-gru-model.h5")

with open("saved_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("saved_model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 128


def clean_text(text):
    """
    Cleans the input text by converting it to lowercase, removing non-alphanumeric characters,
    and removing stopwords.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def mask_email_string(input_email_body):
    """
    Masks specific entities such as full names, emails, and phone numbers in the input email body.
    It also predicts the category of the email using a pre-trained model.

    Args:
        input_email_body (str): The raw email text to be processed.

    Returns:
        str: A JSON string with the masked email, the list of masked entities, and the predicted category.
    """
    masked_email = input_email_body
    entities = []
    replacements = []

    # Find full name (e.g., "My name is John Doe")
    name_pattern = r"(My name is)\s+([A-Za-z]+)\s+([A-Za-z]+)([^\w\s]*)"
    for match in re.finditer(name_pattern, input_email_body, flags=re.IGNORECASE):
        full_entity = match.group(2) + " " + match.group(3)
        start, end = match.start(2), match.end(3)
        entities.append({
            "position": [start, end],
            "classification": "full_name",
            "entity": full_entity
        })
        replacements.append((start, end, "[full_name]"))

    # Find email (e.g., "You can reach me at john.doe@example.com")
    email_pattern = r"(You can reach me at)\s+(\S+)"
    for match in re.finditer(email_pattern, input_email_body, flags=re.IGNORECASE):
        email_entity = match.group(2)
        start, end = match.start(2), match.end(2)
        entities.append({
            "position": [start, end],
            "classification": "email",
            "entity": email_entity
        })
        replacements.append((start, end, "[email]"))

    # Find phone number (e.g., "My contact number is +91-9876543210")
    phone_pattern = r"(My contact number is)\s+(\+\d[\d\-]*)"
    for match in re.finditer(phone_pattern, input_email_body, flags=re.IGNORECASE):
        phone_entity = match.group(2)
        start, end = match.start(2), match.end(2)
        entities.append({
            "position": [start, end],
            "classification": "phone_number",
            "entity": phone_entity
        })
        replacements.append((start, end, "[phone_number]"))

    # Sort replacements in reverse order to avoid shifting of positions
    replacements.sort(reverse=True)
    for start, end, repl in replacements:
        masked_email = masked_email[:start] + repl + masked_email[end:]

    # Predict category using the pre-trained model
    cleaned_text = clean_text(masked_email)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded_seq, verbose=0)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    # Prepare the result in JSON format
    result_json = {
        "input_email_body": input_email_body,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": predicted_class
    }
    json_output = json.dumps(result_json, indent=2)

    return json_output
