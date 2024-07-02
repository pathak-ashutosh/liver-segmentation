import numpy as np
from skimage.segmentation import mark_boundaries
from ..model.model import get_unet
from .dicom_utils import load_and_preprocess_dicom

def predict_dicom(dicom_path, weights_path, mean=0, std=1):
    """Predict liver segmentation for a single DICOM file."""
    # Load and preprocess the DICOM file
    img = load_and_preprocess_dicom(dicom_path)
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