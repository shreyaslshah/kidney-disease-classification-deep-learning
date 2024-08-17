from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger

# Set environment variables
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

app = Flask(__name__)
CORS(app)


class ClientApp:
  def __init__(self):
    self.filename = "inputImage.jpg"
    self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
  try:
    return render_template('index.html')
  except Exception as e:
    logger.error(f"Error rendering home page: {e}")
    return str(e), 500


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
  try:
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"
  except Exception as e:
    logger.error(f"Error during training: {e}")
    return str(e), 500


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
  try:
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)
  except Exception as e:
    logger.error(f"Error during prediction: {e}")
    return str(e), 500


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8080))
  app.run(host='0.0.0.0', port=port)
