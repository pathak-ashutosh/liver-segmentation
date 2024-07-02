import pydicom
import numpy as np
from skimage.transform import resize

def load_dicom(file_path):
    """Load a DICOM file and return it as a numpy array."""
    dicom = pydicom.dcmread(file_path)
    return dicom.pixel_array

def preprocess_dicom(dicom_array, target_size=(256, 256)):
    """Preprocess the DICOM array by resizing and normalizing."""
    # Resize the image
    resized = resize(dicom_array, target_size, mode='constant', preserve_range=True)
    
    # Normalize the image
    normalized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))
    
    return normalized.astype(np.float32)

def load_and_preprocess_dicom(file_path, target_size=(256, 256)):
    """Load a DICOM file, preprocess it, and return as a numpy array."""
    dicom_array = load_dicom(file_path)
    return preprocess_dicom(dicom_array, target_size)