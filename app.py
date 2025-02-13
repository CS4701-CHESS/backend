from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


class APIError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code


@app.errorhandler(APIError)
def handle_api_error(error):
    response = jsonify({"error": error.message})
    response.status_code = error.status_code
    return response


@app.route("/", methods=["GET"])
def root():
    return "Hello World! Welcome to the Flask Backend"


@app.route("/api/hello", methods=["GET", "POST"])
def hello():
    if request.method == "GET":
        return jsonify({"message": "Hello, World!"})

    elif request.method == "POST":
        data = request.get_json()
        if not data or "name" not in data:
            raise APIError("Name is required in the request body")

        return jsonify({"message": f"Hello, {data['name']}!"})


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(host="0.0.0.0", port=port, debug=True)
