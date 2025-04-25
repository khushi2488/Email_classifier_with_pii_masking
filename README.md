📧 Email Classification API for Support Teams
A backend-only FastAPI project that classifies support emails into predefined categories after masking sensitive personal information like names, emails, phone numbers, etc. Includes PII detection, masking, classification, and demasking — all exposed via a clean API endpoint. Ready for deployment on Hugging Face Spaces.
# Email Classifier with PII Masking

## Overview

The **Email Classifier with PII Masking** is a Python-based project designed to classify incoming support emails into predefined categories while masking personally identifiable information (PII) for privacy and compliance. This tool ensures that sensitive information like names, email addresses, phone numbers, and others are protected before processing the email content.

📌 Features
Built with FastAPI
Supports email classification and masking of PII
RESTful POST endpoint at /classify
Interactive API documentation via Swagger UI

## 🚀 Deployment

- **🔗 Live API:** [https://khushi2488-email-classifier-with-pii-masking.hf.space](https://khushi2488-email-classifier-with-pii-masking.hf.space)
- **📄 Swagger UI:** [https://khushi2488-email-classifier-with-pii-masking.hf.space/docs](https://khushi2488-email-classifier-with-pii-masking.hf.space/docs)

🧪 How to Use the API
Make a POST request to: https://khushi2488-email-classifier-with-pii-masking.hf.space/classify

⚙️ Setup Instructions
1.Clone the repository
git clone https://github.com/khushi2488/email-classifier-api.git
cd email-classifier-api
2.Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3.Install dependencies:
pip install -r requirements.txt
4.Run the app locally:
uvicorn app:app --reload


