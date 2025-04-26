"""
Flask API to mask sensitive entities in email text using NLP and return
the masked result along with metadata and classification.
"""

from flask import Flask, request, jsonify, Response
from utils import mask_email_string

app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def mask_text():
    """
    API endpoint to mask sensitive information in a given email body.

    Expects:
        JSON request with a key 'email' containing the email body.

    Returns:
        JSON response with:
            - input_email_body: original text
            - masked_email: text with masked entities
            - list_of_masked_entities: entities that were masked
            - category_of_the_email: predicted category
    """
    content = request.get_json()

    if not content or 'email' not in content:
        return jsonify({
            "error": "Please provide 'email' in the request body"
        }), 400

    result_json = mask_email_string(content['email'])

    return Response(result_json, mimetype='application/json')


if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)