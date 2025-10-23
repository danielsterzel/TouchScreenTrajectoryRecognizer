from flask import Flask, render_template, request, jsonify, send_from_directory
import json
from markupsafe import escape

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")
@app.route('/')
def index():
    return send_from_directory(app.template_folder, "index.html")

@app.route("/<path:path>")
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)



if __name__ == "__main__":
    app.run(debug=True)