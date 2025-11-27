from flask import Flask, request, jsonify, send_from_directory
from flask import abort
# from flask import render_template
# from markupsafe import escape
import os
import functions as func
import constants as const
import model_related_functions as mrf

app = Flask(__name__, static_folder=const.FRONTEND_DIR, template_folder=const.FRONTEND_DIR)

os.makedirs(const.DATA_DIR, exist_ok=True)

@app.before_request
def limit_remote_addr():
    if request.remote_addr not in const.ALLOWED_IP_ADDRESSES:
        abort(403)
@app.route('/')
def index():
    return send_from_directory(app.template_folder, "index.html")

@app.route("/<path:path>")
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/predict", methods=["POST"])
def predict():

    image_data = request.files["image"].read()

    save_path = func.get_next_filename()
    with open(save_path, "wb") as f:
        f.write(image_data)
    print(f"Saved img to {save_path}")

    pred = mrf.quickdraw_predict_img(image_data)
    print(f'Prediction from quickdraw model: {pred}')
    return jsonify({
        "prediction": pred
    })

@app.route("/submit-img", methods=["POST"])
def submit_img():

    file = request.files["image"]
    print(f"Request files:\n{request.files}")

    file_path = func.get_next_filename()
    file.save(file_path)
    return {
        "status": "success",
        "path": file_path
    }

if __name__ == "__main__":
    # func.preprocess_data_for_model()
    # mrf.run_all_models()
    # mrf.build_and_run_all_models()
    app.run(host="0.0.0.0", port=5000,debug=True)


