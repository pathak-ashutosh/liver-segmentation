import nibabel as nib
import numpy as np
from skimage.transform import resize

def load_nifti(file_path):
    """Load a NIfTI file and return it as a numpy array."""
    nifti = nib.load(file_path)
    return nifti.get_fdata()

def preprocess_nifti(nifti_array, target_size=(256, 256)):
    """Preprocess the NIfTI array by selecting a middle slice, resizing and normalizing."""
    # Select the middle slice if it's a 3D volume
    if nifti_array.ndim == 3:
        middle_slice = nifti_array.shape[2] // 2
        nifti_array = nifti_array[:, :, middle_slice]
    
    # Resize the image
    resized = resize(nifti_array, target_size, mode='constant', preserve_range=True)
    
    # Normalize the image
    normalized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))
    
    return normalized.astype(np.float32)

def load_and_preprocess_nifti(file_path, target_size=(256, 256)):
    """Load a NIfTI file, preprocess it, and return as a numpy array."""
    nifti_array = load_nifti(file_path)
    return preprocess_nifti(nifti_array, target_size)