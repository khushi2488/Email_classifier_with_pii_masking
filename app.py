"""
Main entry point for processing the email text and demonstrating the masking
of sensitive entities in an email using an API request.
"""

from utils import mask_email_string
import json
def main():
    """
    Run a demo on masking email content by processing input text.
    """
    
    # Example email string to mask sensitive information
    sample_email = (
        "My name is John Doe. You can reach me at john.doe@example.com. "
        "My contact number is +91-9876543210."
    )

    result_dict = mask_email_string(sample_email)

    # Display the results in the required format
    print(json.dumps(result_dict, indent=2))
    
if __name__ == "__main__":
    main()
