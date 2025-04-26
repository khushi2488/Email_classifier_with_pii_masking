from fastapi import FastAPI
from pydantic import BaseModel
import json

from utils import mask_email_string

# Create a FastAPI app instance
app = FastAPI()


class EmailRequest(BaseModel):
    """Input model containing the email text."""
    email: str


class EmailResponse(BaseModel):
    """Output model for masked email response."""
    input_email_body: str
    list_of_masked_entities: list


@app.post("/classify")
def classify_email(request: EmailRequest):
    """
    API endpoint to classify and mask sensitive information in an email.

    Args:
        request (EmailRequest): The email input.

    Returns:
        dict: Masked email body, list of masked entities, and category.
    """
    # Call the masking function
    result_json_str = mask_email_string(request.email)

    # Convert JSON string to Python dict
    result_dict = json.loads(result_json_str)

    return result_dict
