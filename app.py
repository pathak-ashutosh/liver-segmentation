import gradio as gr
from src.model.model import get_unet
# from src.dicom_handler.dicom_predictor import predict_dicom
from src.nifti_handler.nifti_predictor import predict_nifti

# Load the model and weights
model = get_unet()
model.load_weights('models/weights.h5')

# Define mean and std (should have saved these during training)
mean = 0  # Replace with actual mean
std = 1   # Replace with actual std

def predict_liver_segmentation(file):
    # segmented = predict_dicom(file.name, 'models/weights.h5')
    segmented = predict_nifti(file.name, 'models/weights.h5')
    return segmented

iface = gr.Interface(
    fn=predict_liver_segmentation,
    inputs=gr.File(label="Upload NIfTI file (.nii.gz)"),
    outputs=gr.Image(label="Segmentation Result"),
    title="Liver Segmentation from NIfTI",
    description="Upload a liver CT scan NIfTI file (.nii.gz) to get the segmentation result."
)

if __name__ == "__main__":
    iface.launch(share=True)