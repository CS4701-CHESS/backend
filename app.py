from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import eval as model
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    logger.debug(f"Test endpoint received data: {data}")

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
# a json object with a "fen" key, and will return a json object with a "move" key and eval key.
@app.route("/api/move", methods=["POST"])
def fen_to_san():
    data = request.get_json()
    logger.debug(f"Move endpoint received data: {data}")

    if not data:
        logger.error("No request body provided")
        raise APIError("Request body is required")

    # Extract the FEN from the request body
    if "fen" in data:

        fen = data["fen"]

        # Get optional parameters with defaults
        depth = int(data.get("depth", 2))
        is_white = bool(data.get("isWhite", True))

        logger.debug(
            f"Processing move for FEN: {fen}, depth: {depth}, isWhite: {is_white}"
        )

        try:
            move = model.predict_move_fen(fen)
            eval_score = None

            logger.debug(f"AI returned move: {move}, eval: {eval_score}")

            # Return the move and evaluation
            return jsonify(
                {"status": "success", "received": fen, "move": move, "eval": eval_score}
            )

        except Exception as e:
            logger.error(f"Error calculating move: {str(e)}")
            raise APIError(f"Error calculating move: {str(e)}")
    else:
        # if no fen is given, raise error
        logger.error("No FEN string provided in request")
        raise APIError("FEN string is required")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
