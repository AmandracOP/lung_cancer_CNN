# src/train.py
import os
import tensorflow as tf
from utils import load_data, build_model, get_base_models

# Paths
train_dir = 'data/train'
validation_dir = 'data/validation'
model_save_dir = 'models'

# Ensure the model save directory exists
os.makedirs(model_save_dir, exist_ok=True)

# Load the data
train_gen, val_gen, _ = load_data(train_dir, validation_dir, '')

# Train each model
for model_name, base_model in get_base_models().items():
    print(f"Training model: {model_name}")
    model = build_model(base_model)
    
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,  # Increase epochs for better accuracy
    )
    
    # Save the model
    model_path = os.path.join(model_save_dir, f"{model_name.lower()}.h5")
    model.save(model_path)
    print(f"Model saved: {model_path}")
