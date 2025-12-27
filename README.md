# TouchScreenTrajectoryRecognizer

This project implements a system for recognizing hand-drawn symbols using
convolutional neural networks (CNNs). The application consists of a frontend
(HTML/JavaScript/CSS) and a backend implemented using the Flask framework,
which integrates a trained machine learning model for inference.

## Requirements
- Python 3.10+
- pip
- virtual environment tool (recommended)

## Installation
It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```
Install required dependencies:
```bash
pip install -r requirements.txt
```
## Run
In order to run the application:
```bash
python main.py
```
After starting the server, application will be available on:
```
http://localhost:5000/
```
The server listens on 0.0.0.0, which allows access from other devices
within the same local network, for example:
```
http://<host_IP_address>:5000/
```

## Usage

1. Open the application in a web browser.
2. Draw a symbol on the canvas.
3. Click the Send button to submit the drawing to the backend.
4. The predicted class label is returned and displayed in the user interface.

## Data

The project uses the Google QuickDraw dataset.
The dataset is not included directly in the repository and must be obtained
according to the project documentation.
