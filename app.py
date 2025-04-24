from fastapi import FastAPI
from pydantic import BaseModel
import json

from utils import mask_email_string

# Create a FastAPI app instance
app = FastAPI()


# Define input model
class EmailRequest(BaseModel):
    email: str


# Define output model (optional for clarity)
class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: list


# Route for POST API
@app.post("/classify")
def classify_email(request: EmailRequest):
    # Call the masking function (your logic)
    result_json_str = mask_email_string(request.email)

    # Convert JSON string to Python dict
    result_dict = json.loads(result_json_str)

    # Return it as JSON response
    return result_dict
