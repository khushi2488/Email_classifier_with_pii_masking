📧 Email Classification API for Support Teams
A backend-only FastAPI project that classifies support emails into predefined categories (e.g., Billing, Technical, Account Issues) after masking sensitive personal information like names, emails, phone numbers, etc. Includes PII detection, masking, classification, and demasking — all exposed via a clean API endpoint. Ready for deployment on Hugging Face Spaces.
# Email Classifier with PII Masking

## Overview

The **Email Classifier with PII Masking** is a Python-based project designed to classify incoming support emails into predefined categories while masking personally identifiable information (PII) for privacy and compliance. This tool ensures that sensitive information like names, email addresses, phone numbers, and others are protected before processing the email content.

## Features

- **PII Masking**: Automatically identifies and masks PII like names, emails, phone numbers, and more.
- **Email Classification**: Categorizes support emails into predefined categories, such as Billing, Technical Support, Account Management, etc.
- **Data Privacy**: The system ensures that PII is masked during processing, allowing for safe handling of sensitive data.
- **API Integration**: The system provides an API endpoint that can be used to process email content programmatically.

## Prerequisites

Before setting up and running the project, ensure you have the following installed:

- Python 3.7 or higher
- Git (for version control)
- A virtual environment (optional but recommended)

## Installation and Setup

1. **Clone the Repository**

   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/khushi2488/Email_classifier_with_pii_masking.git
   cd Email_classifier_with_pii_masking

