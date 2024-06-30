import torch
os.chdir('/home/dominik/Documents/ra_hannes/war-destruction/test/mwd/destruction/')
from destruction_utilities import *
from destruction_models import *


os.listdir()
# Load the pre-trained model
model_path = f'{paths.models}/Aerial_SwinB_SI.pth'
image_encoder = ImageEncoder(feature_extractor=torch.load(model_path))

# If the model is saved as a state_dict, you may need to load it differently:
# state_dict = torch.load(model_path)
# image_encoder.load_state_dict(state_dict)
