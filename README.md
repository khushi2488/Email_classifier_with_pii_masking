## 📧 Email Classification API for Support Teams
A backend-only FastAPI project that classifies support emails into predefined categories after masking sensitive personal information like names, emails, phone numbers, etc. Includes PII detection, masking, classification, and demasking — all exposed via a clean API endpoint. Ready for deployment on Hugging Face Spaces.
# Email Classifier with PII Masking

## Overview

The **Email Classifier with PII Masking** is a Python-based project designed to classify incoming support emails into predefined categories while masking personally identifiable information (PII) for privacy and compliance. This tool ensures that sensitive information like names, email addresses, phone numbers, and others are protected before processing the email content.

## 📌 Features

1)Built with FastAPI

2)Supports email classification and masking of PII

3)RESTful POST endpoint at /classify

4)Interactive API documentation via Swagger UI

## 🚀 Deployment

- **🔗 Live API:** [https://khushi2488-email-classifier-with-pii-masking.hf.space](https://khushi2488-email-classifier-with-pii-masking.hf.space)
- **📄 Swagger UI:** [https://khushi2488-email-classifier-with-pii-masking.hf.space/docs](https://khushi2488-email-classifier-with-pii-masking.hf.space/docs)

🧪 How to Use the API
Make a POST request to: https://khushi2488-email-classifier-with-pii-masking.hf.space/classify

## ⚙️ Setup Instructions

1.Clone the repository : git clone https://github.com/khushi2488/email-classifier-api.git

2.cd email-classifier-api

3.Create a virtual environment (optional but recommended):
python -m venv venv

4.source venv/bin/activate  # On Windows: venv\Scripts\activate

5.Install dependencies:pip install -r requirements.txt

6.Run the app locally:uvicorn app:app --reload

## 📄 Project Report

Download or view the report: [Email_Classification_System_Report.pdf](./Email_Classification_System_Report.pdf)


