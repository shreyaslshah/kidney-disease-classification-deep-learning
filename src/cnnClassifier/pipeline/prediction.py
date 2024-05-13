import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import os


class PredictionPipeline:
  def __init__(self, filename):
    self.filename = filename

  def predict(self):
    # load model
    model = load_model(os.path.join("model", "model.h5"))

    filename = self.filename

    # Load and preprocess the single image
    datagenerator_kwargs = dict(
        rescale=1./255
    )

    dataflow_kwargs = dict(
        target_size=[224, 224],
        interpolation="bilinear"
    )

    image_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagenerator_kwargs
    )

    image_generator = image_datagenerator.flow_from_dataframe(
        dataframe=pd.DataFrame({"filename": [filename]}),
        directory=None,
        x_col="filename",
        y_col=None,
        class_mode=None,
        shuffle=False,
        **dataflow_kwargs
    )

    # Make predictions
    predictions = model.predict(image_generator)

    result = np.argmax(predictions, axis=1)

    if result[0] == 1:
      prediction = 'Tumor'
      return [{"image": prediction}]
    else:
      prediction = 'Normal'
      return [{"image": prediction}]
