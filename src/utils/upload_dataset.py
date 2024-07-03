from datasets import Dataset, Features, Value
import os
import nibabel as nib

# Define the paths to your raw data files
raw_data_path = './data/raw'

# Function to load NIfTI files
def load_nifti(file_path):
    nifti = nib.load(file_path)
    return nifti.get_fdata()

# Create a list to hold the data
data = []

# Iterate over the files in the raw data directory
for file_name in os.listdir(raw_data_path):
    if file_name.endswith('.nii.gz'):
        file_path = os.path.join(raw_data_path, file_name)
        data.append({
            'file_name': file_name,
            'data': load_nifti(file_path).tolist()  # Convert to list for serialization
        })

# Define the features of the dataset
features = Features({
    'file_name': Value('string'),
    'data': Value('float32', id='data')
})

# Create Dataset object
dataset = Dataset.from_dict({'file_name': [d['file_name'] for d in data], 'data': [d['data'] for d in data]}, features=features)

if __name__ == "__main__":
    # Push the dataset to Hugging Face
    dataset.push_to_hub("ashutosh-pathak/liver-segmentation")