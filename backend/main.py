from flask import Flask, request, jsonify, send_from_directory
from flask import abort
# from flask import render_template
import json
import os
import functions as func

# from markupsafe import escape

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ALLOWED_IP_ADDRESSES = ["192.168.40.23", "127.0.0.1", "192.168.40.46"]
os.makedirs(DATA_DIR, exist_ok=True)

counter = func.get_next_index(DATA_DIR)


@app.before_request
def limit_remote_addr():
    if request.remote_addr not in ALLOWED_IP_ADDRESSES:
        abort(403)
@app.route('/')
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/<path:path>")
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/submit-points", methods=["POST"])
def submit_points():
    global counter #global - use from global scope
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400


    filename = f"points_{counter}.json"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as file:
        json.dump(data, file, indent=2)  # type: ignore

    counter += 1

    func.preprocess_data_for_model(DATA_DIR)

    return jsonify(
        {"status": "success",
         "file": filename,
         "points received": len(data)
         }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)