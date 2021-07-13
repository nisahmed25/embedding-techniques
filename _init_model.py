"""
Models
"""
import warnings
# keras imports
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# pylint: disable=bad-indentation

# filter warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def keras_models(model_name, weights):
  """
  Model selection
  """
  # pylint: disable=import-outside-toplevel

  if model_name == "inceptionv3":
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    base_model = InceptionV3(include_top=False,
                             weights=weights,
                             input_tensor=Input(shape=(299, 299, 3)))
    image_size = (299, 299)
  elif model_name == "mobilenet":
    from tensorflow.keras.applications.mobilenet import preprocess_input
    base_model = MobileNet(include_top=False,
                           weights=weights,
                           input_tensor=Input(shape=(224, 224, 3)))
    image_size = (224, 224)
  elif model_name == "efficientnetb3":
    from tensorflow.keras.applications.efficientnet import preprocess_input
    base_model = EfficientNetB3(include_top=False,
                                weights=weights,
                                input_tensor=Input(shape=(300, 300, 3)))
    image_size = (300, 300)
  else:
    base_model = None

  _x = base_model.output
  _x = GlobalAveragePooling2D()(_x)
  model_embed = Model(inputs=base_model.input, outputs=_x)

  return model_embed, preprocess_input, image_size


def get_model(config):
  """
    config variables
    """
  model_name = config["model"]
  weights = config["weights"]
  return keras_models(model_name, weights)
