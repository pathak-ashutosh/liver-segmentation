import numpy as np
from skimage.segmentation import mark_boundaries
from ..model.model import get_unet
from .nifti_utils import load_and_preprocess_nifti

def predict_nifti(nifti_path, weights_path, mean=0, std=1):
    """Predict liver segmentation for a single NIfTI file."""
    # Load and preprocess the NIfTI file
    img = load_and_preprocess_nifti(nifti_path)
    img = (img - mean) / std

    # Load the model and weights
    model = get_unet()
    model.load_weights(weights_path)

    # Predict
    mask = model.predict(np.expand_dims(img, axis=0))[0]
    mask = (mask > 0.5).astype('uint8')

    # Overlay the mask on the original image
    segmented = mark_boundaries(img, mask[:,:,0], color=(1,0,0), mode='thick')
    return segmented