import h5py

with h5py.File(r'C:\Users\ayaan\OneDrive\Desktop\hp\SignComm\Phase_1\keras_model.h5', 'r+') as f:
    model_config = f.attrs['model_config']
    # Modify the configuration to remove 'groups'
    model_config = model_config.replace('"groups": 1,', '')  # Remove the 'groups' field
    f.attrs['model_config'] = model_config.encode('utf-8')

print("Model configuration updated successfully!")
