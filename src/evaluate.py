import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from utils import load_data
import tensorflow as tf

# Paths
test_dir = 'data/test'  # Replace with your actual test data directory path
model_save_dir = 'models'  # Replace with your actual model directory path
results_dir = 'results'

# Ensure results directories exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'classification_reports'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'confusion_matrices'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

# Load the test data
_, _, test_gen = load_data(None, None, test_dir)  # Use None for train and validation directories

# Initialize results list
results = []

# Evaluate each model
for model_file in os.listdir(model_save_dir):
    model_name = model_file.split('.')[0].capitalize()
    model_path = os.path.join(model_save_dir, model_file)
    print(f"Evaluating model: {model_name}")
    
    # Load the model
    model = load_model(model_path)
    
    # Measure latency
    start_time = time.time()
    predictions = model.predict(test_gen)
    latency = time.time() - start_time
    
    # Model size (in millions of parameters)
    model_size = model.count_params() / 1e6
    
    # Convert predictions to class labels
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys()), output_dict=True)
    report_path = os.path.join(results_dir, 'classification_reports', f"{model_name}_classification_report.csv")
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Classification report saved: {report_path}")
    
    # Confusion matrix and heatmap
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(results_dir, 'confusion_matrices', f"{model_name}_confusion_matrix.csv")
    pd.DataFrame(cm).to_csv(cm_path)
    print(f"Confusion matrix saved: {cm_path}")
    
    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(results_dir, 'plots', f'{model_name}_confusion_matrix.png'))
    plt.close()

    # TensorFlow Lite conversion and model compression size
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model
    tflite_path = os.path.join(model_save_dir, f"{model_name}_compressed.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    compressed_size = os.path.getsize(tflite_path) / 1e6  # in MB
    
    # Save metrics
    results.append({
        'Model': model_name,
        'Model Size (M params)': model_size,
        'Compressed Size (MB)': compressed_size,
        'Latency (s)': latency,
        'Accuracy': report['accuracy']
    })

# Save all metrics to a CSV file
results_df = pd.DataFrame(results)
results_path = os.path.join(results_dir, 'model_metrics.csv')
results_df.to_csv(results_path, index=False)
print(f"All model metrics saved: {results_path}")
