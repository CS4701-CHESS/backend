from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import minimax

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


@app.route("/api/test", methods=["POST"])
def test_endpoint():
    data = request.get_json()

    if not data:
        raise APIError("Request body is required")

    # Extract the message from the request body
    if "message" in data:
        received_message = data["message"]
    else:
        # If no specific message field, use the first string value found
        received_message = next(
            (v for v in data.values() if isinstance(v, str)), "No string found"
        )

    # Return the message with greeting
    return jsonify(
        {
            "status": "success",
            "received": received_message,
            "response": f"Hello from the backend! You sent: {received_message}",
        }
    )


# takes a fen string and returns the recommended move in san notation. Expects
# a json object with a "fen" key, and will return a json object with a "san" key.
@app.route("/api/move", methods=["POST"])
def fen_to_san():
    data = request.get_json()

    if not data:
        raise APIError("Request body is required")

    # Extract the message from the request body
    if "fen" in data:
        fen = data["message"]
        san = minimax.base_minimax(fen, 1, True)
    else:
        # if no fen is given, raise error
        raise APIError("Fen string is required")

    # Return the message with greeting
    return jsonify(
        {
            "status": "success",
            "received": fen,
            "response": san,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
