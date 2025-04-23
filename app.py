"""
Main entry point for training and evaluating the email classification model,
as well as demonstrating the masking of sensitive entities in a sample email.
"""

from models import build_model, train_model, evaluate_model
from utils import load_and_clean_data, mask_email_string


def main():
    """
    Train the model, evaluate it, and run a demo on masking email content.
    """

    # Load and preprocess data from the Excel file
    (
        X_train, X_test,
        y_train, y_test,
        tokenizer, label_encoder,
        num_classes
    ) = load_and_clean_data("masked_data_final.xlsx")

    # Build the model using the specified vocabulary size and number of classes
    model = build_model(vocab_size=20000, num_classes=num_classes)

    # Train the model on the training dataset
    model = train_model(model, X_train, y_train)

    # Evaluate the model on the test dataset
    evaluate_model(model, X_test, y_test, label_encoder)

    # Demo: Test masking functionality on a sample email
    print("\n--- Email Masking Demo ---")
    sample_email = (
        "My name is John Doe. You can reach me at john.doe@example.com. "
        "My contact number is +91-9876543210."
    )

    masked_result = mask_email_string(sample_email)

    # Display the results
    print("\nOriginal Email:\n", masked_result["input_email_body"])
    print("\nMasked Email:\n", masked_result["masked_email"])
    print("\nMasked Entities:\n", masked_result["list_of_masked_entities"])


if __name__ == "__main__":
    main()
