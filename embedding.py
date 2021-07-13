"""
Embeddings
"""
import warnings
import json
import glob
import pandas as pd
from _init_model import get_model
from input_img import inp_img
from padded_image import crop_pad

# pylint: disable=bad-indentation

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

configs = [item for item in glob.glob('./config/*') if item.endswith('.json')]

for config in configs:
  with open(config) as f:
    config = json.load(f)
  print(config['model'])
  model_embed, preprocess_input, image_size = get_model(config)
  img, pts = inp_img()
  padded_img = crop_pad(image_size, img, pts)
  embeddings = model_embed.predict(padded_img)
  df = pd.DataFrame(embeddings)
  print(df)
  df.to_csv(r'./saved_csv/{}test.csv'.format(config['model']), index=False)
