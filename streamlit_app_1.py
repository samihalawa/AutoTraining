# -*- coding: utf-8 -*-
"""
Production-Ready LLM and Vision Model Data Generation, Training, Evaluation, and Deployment Application using Streamlit
"""

# --------------------------- 1. Import Necessary Libraries ---------------------------
import os
import sys
import subprocess
import threading
import time
import queue
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from huggingface_hub import login, HfApi, HfFolder
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import psutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.express as px
from ray import tune
import onnx
import torch.onnx
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import zipfile
from torchvision import transforms
from torch import nn
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import atexit

# --------------------------- 2. Dependency Management ---------------------------
def install_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        st.error(f"Error installing packages: {e}")
        sys.exit(1)

# Uncomment the following lines to install dependencies programmatically
# try:
#     install_packages()
# except Exception as e:
#     st.error(f"Error installing packages: {e}")
#     sys.exit(1)

# --------------------------- 3. Environment Variables ----------------------------
load_dotenv()

# --------------------------- 4. Logging Configuration -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use logger.info(), logger.warning(), logger.error() throughout the app

# --------------------------- 5. Utility Functions ----------------------------------



# Define the missing function `set_hf_credentials`
def set_hf_credentials(token, username):
    if not token or not username:
        st.warning("Please provide both token and username.")
        return
    os.environ["HF_USERNAME"] = username
    encryption_key = Fernet.generate_key()
    os.environ["HF_TOKEN_ENCRYPTED"] = encrypt_data(token, encryption_key).decode()
    st.session_state["encryption_key"] = encryption_key
    st.success("Credentials set successfully.")

# Define the missing function `get_model_list` for selecting a model in the UI
def get_model_list():
    return ["gpt2", "facebook/maskrcnn_resnet50_fpn", "distilgpt2"]

# Define the missing function `aggregate_training_params`
def aggregate_training_params(lr, epochs, batch_size, gradient_accumulation, mixed_precision, peft, quantization, weight_decay, warmup_steps, logging_steps):
    return {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "mixed_precision": mixed_precision,
        "peft": peft,
        "quantization": quantization,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "logging_steps": logging_steps,
    }

# 5.1. Check if CUDA is available
def is_cuda_available():
    available = torch.cuda.is_available()
    logger.debug(f"CUDA available: {available}")
    return available

# 5.2. Encrypt Sensitive Data
def encrypt_data(data, key):
    try:
        cipher_suite = Fernet(key)
        encrypted_data = cipher_suite.encrypt(data.encode())
        logger.info("Data encrypted successfully.")
        return encrypted_data
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        return None

# 5.3. Decrypt Sensitive Data
def decrypt_data(encrypted_data, key):
    try:
        cipher_suite = Fernet(key)
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        logger.info("Data decrypted successfully.")
        return decrypted_data
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        return None

# 5.4. Generate synthetic text data using a specified model and prompt
def generate_text_data(prompt, num_rows, model_name):
    try:
        device = 0 if is_cuda_available() else -1
        generator = pipeline('text-generation', model=model_name, device=device)
        generated_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(int(num_rows)):
            result = generator(prompt, max_length=100, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
            generated_text = result[0]['generated_text']
            generated_data.append({"text": generated_text})
            progress_bar.progress((i + 1) / int(num_rows))
            status_text.text(f"Generating data... {i + 1}/{num_rows}")
            time.sleep(0.1)  # To simulate processing time
        st.session_state.generated_data = pd.DataFrame(generated_data)
        return st.session_state.generated_data
    except Exception as e:
        logger.error(f"Error during text data generation: {e}")
        st.error(f"Error during text data generation: {e}")
        return pd.DataFrame({"text": [f"Error: {e}"]})

# 5.5. Save generated text data to train_text.csv
def save_text_data(generated_data):
    try:
        data_dir = Path('data/text')
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / 'train_text.csv'
        df_new = pd.DataFrame(generated_data)
        if data_file.exists():
            df_existing = pd.read_csv(data_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['text'])
        else:
            df_combined = df_new
        df_combined.to_csv(data_file, index=False)
        logger.info("Generated text data saved to data/text/train_text.csv successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving generated text data: {e}")
        st.error(f"Error saving generated text data: {e}")
        return False

# 5.6. Add user-provided text data to train_text.csv
def add_text_data(input_text):
    if not input_text.strip():
        logger.warning("No input provided for adding text data.")
        st.warning("No input provided.")
        return "No input provided."
    try:
        logger.info("Adding text data to data/text/train_text.csv...")
        data_dir = Path('data/text')
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / 'train_text.csv'
        df_new = pd.DataFrame([{ "text": input_text }])
        if data_file.exists():
            df_existing = pd.read_csv(data_file)
            if input_text in df_existing['text'].values:
                logger.info("Text data is already present in train_text.csv. Skipping addition.")
                st.info("Text data is already present in train_text.csv.")
                return "Text data is already present in train_text.csv."
            df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['text'])
        else:
            df_combined = df_new
        df_combined.to_csv(data_file, index=False)
        logger.info("Text data added and saved successfully.")
        st.success("Text data added and saved successfully.")
        return "Text data added and saved."
    except Exception as e:
        logger.error(f"Error adding text data: {e}")
        st.error(f"Error adding text data: {e}")
        return f"Error adding text data: {e}"

# 5.7. Upload text data file
def upload_text_data(file):
    if file is None:
        logger.warning("No text file uploaded.")
        st.warning("No file uploaded.")
        return "No file uploaded."
    try:
        valid, message = validate_text_uploaded_file(file)
        if not valid:
            st.error(message)
            return message
        logger.info(f"Uploading text data from {file.name}...")
        data_dir = Path('data/text')
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / 'train_text.csv'
        df_new = pd.read_csv(file)
        if data_file.exists():
            df_existing = pd.read_csv(data_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['text'])
        else:
            df_combined = df_new
        df_combined.to_csv(data_file, index=False)
        logger.info("Uploaded text data saved to data/text/train_text.csv successfully.")
        st.success("Text data uploaded and saved successfully.")
        return "Text data uploaded and saved successfully."
    except Exception as e:
        logger.error(f"Error uploading text data: {e}")
        st.error(f"Error uploading text data: {e}")
        return f"Error uploading text data: {e}"

# 5.8. Validate Uploaded Text File
def validate_text_uploaded_file(file):
    try:
        if file is None:
            logger.warning("No file uploaded.")
            return False, "No file uploaded."
        if not file.name.endswith('.csv'):
            logger.warning("Uploaded text file is not a CSV.")
            return False, "Uploaded file must be a CSV."
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            logger.warning("CSV does not contain 'text' column.")
            return False, "CSV must contain a 'text' column."
        logger.info("Uploaded text file validated successfully.")
        return True, "File is valid."
    except Exception as e:
        logger.error(f"Error validating uploaded text file: {e}")
        return False, f"Error validating file: {e}"

def preprocess_image(file):
    img = Image.open(file)
    img_tensor = mae_preprocess(img)
    return img_tensor

def upload_image_data(files):
    images_dir = Path('data/images/images')
    images_dir.mkdir(parents=True, exist_ok=True)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for file in files:
            if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                future = executor.submit(preprocess_image, file)
                futures.append((file.name, future))
        
        for filename, future in futures:
            img_tensor = future.result()
            torch.save(img_tensor, images_dir / f"{filename}.pt")

# 5.10. Validate Uploaded Image Files
def validate_image_uploaded_files(images, masks):
    try:
        if images is None or masks is None:
            logger.warning("No images or masks uploaded.")
            return False, "Both images and masks must be uploaded."
        if len(images) != len(masks):
            logger.warning("Number of images and masks do not match.")
            return False, "Number of images and masks must be the same."
        for img, mask in zip(images, masks):
            if not img.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                logger.warning(f"Image file {img.name} is not a supported format.")
                return False, f"Unsupported image format: {img.name}"
            if not mask.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                logger.warning(f"Mask file {mask.name} is not a supported format.")
                return False, f"Unsupported mask format: {mask.name}"
        logger.info("Uploaded image files validated successfully.")
        return True, "Files are valid."
    except Exception as e:
        logger.error(f"Error validating uploaded image files: {e}")
        return False, f"Error validating files: {e}"

# 5.11. Preview Image Dataset
def preview_image_data():
    try:
        images_dir = Path('data/images/images')
        masks_dir = Path('data/images/masks')
        if not images_dir.exists() or not masks_dir.exists():
            st.info("No image data found.")
            return
        images = sorted(list(images_dir.glob('*')))
        masks = sorted(list(masks_dir.glob('*')))
        if not images:
            st.info("No images found in the dataset.")
            return
        sample_images = images[:5]
        sample_masks = masks[:5]
        for img, mask in zip(sample_images, sample_masks):
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(img), caption=f"Image: {img.name}", use_column_width=True)
            with col2:
                st.image(Image.open(mask), caption=f"Mask: {mask.name}", use_column_width=True)
        st.success("Image data preview loaded successfully.")
    except Exception as e:
        logger.error(f"Error previewing image data: {e}")
        st.error(f"Error previewing image data: {e}")

# 5.12. Fetch a list of vision models for segmentation from Hugging Face
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_vision_model_list():
    try:
        api = HfApi()
        models = api.list_models(
            filter="segmentation",
            sort="downloads",
            direction=-1,
            limit=50
        )
        model_names = [model.modelId for model in models]
        return model_names
    except Exception as e:
        logger.error(f"Error fetching vision model list: {e}")
        raise

# 5.13. Create AutoTrain Configuration for Vision Models
def create_autotrain_vision_configuration(model_name, project_name, push_to_hub, training_params, save_directory='output'):
    try:
        logger.info("Creating AutoTrain configuration for vision model...")
        params_yaml = "\n".join([f"  {key}: {value}" for key, value in training_params.items()])
        token_decrypted = decrypt_data(os.environ.get("HF_TOKEN_ENCRYPTED", "").encode(), st.session_state.get("encryption_key")) if os.getenv("HF_TOKEN_ENCRYPTED") else ""
        conf = f"""
task: image-segmentation
base_model: {model_name}
project_name: {project_name}
log: tensorboard
backend: local
data:
  path: data/images/
  train_split: train
  valid_split: valid
  chat_template: null
  column_mapping:
    image_column: images
    mask_column: masks
params:
{params_yaml}
hub:
  username: {os.environ.get("HF_USERNAME", "")}
  token: {token_decrypted}
  push_to_hub: {str(push_to_hub).lower()}
output_dir: {save_directory}
"""
        conf_file = Path("conf_vision.yaml")
        if conf_file.exists():
            backup_conf = conf_file.with_name(f"conf_vision_backup_{int(time.time())}.yaml")
            conf_file.rename(backup_conf)
            logger.info(f"Existing vision configuration backed up as {backup_conf.name}.")
        with conf_file.open("w") as f:
            f.write(conf)
        logger.info("AutoTrain vision configuration written to conf_vision.yaml.")
        st.success("AutoTrain vision configuration created successfully.")
        start_autotrain_vision_process(conf_file)
        return "AutoTrain vision configuration created and training started."
    except Exception as e:
        logger.error(f"Error creating AutoTrain vision configuration: {e}")
        st.error(f"Error creating AutoTrain vision configuration: {e}")
        return f"Error creating configuration: {e}"

# 5.14. Start AutoTrain Process for Vision Models
def start_autotrain_vision_process(conf_file):
    try:
        logger.info("Starting AutoTrain vision process...")
        process = subprocess.Popen(
            ["autotrain", "--config", str(conf_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy()
        )
        st.session_state.autotrain_vision_process = process
        threading.Thread(target=process_autotrain_logs, args=(process,), daemon=True).start()
        st.success("AutoTrain vision process started.")
    except FileNotFoundError:
        logger.error("AutoTrain command not found. Please ensure it is installed and in PATH.")
        st.error("AutoTrain command not found. Please ensure it is installed and in PATH.")
        log_queue.put("AutoTrain command not found. Please ensure it is installed and in PATH.")
    except Exception as e:
        logger.error(f"Error starting AutoTrain vision process: {e}")
        st.error(f"Error starting AutoTrain vision process: {e}")
        log_queue.put(f"Error starting AutoTrain vision process: {e}")

# 5.15. Pause Training
def pause_training(task_type="text"):
    try:
        process = st.session_state.get('autotrain_process' if task_type == "text" else 'autotrain_vision_process')
        if process and process.poll() is None:
            if os.name == 'nt':
                # Windows-specific code remains unchanged
                pass
            else:
                import signal
                os.kill(process.pid, signal.SIGSTOP)
            logger.info("Training paused.")
            st.success("Training paused successfully.")
            return "Training paused successfully."
        else:
            logger.warning("No active training process to pause.")
            st.warning("No active training process to pause.")
            return "No active training process to pause."
    except Exception as e:
        logger.error(f"Error pausing training: {e}")
        st.error(f"Error pausing training: {e}")
        return f"Error pausing training: {e}"

# 5.16. Cancel Training
def cancel_training(task_type="text"):
    try:
        if task_type == "text":
            process = st.session_state.get('autotrain_process')
        else:
            process = st.session_state.get('autotrain_vision_process')
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
            logger.info("Training canceled.")
            st.success("Training canceled successfully.")
            return "Training canceled successfully."
        else:
            logger.warning("No active training process to cancel.")
            st.warning("No active training process to cancel.")
            return "No active training process to cancel."
    except Exception as e:
        logger.error(f"Error canceling training: {e}")
        st.error(f"Error canceling training: {e}")
        return f"Error canceling training: {e}"

# 5.17. Retrieve the latest AutoTrain logs
def get_logs():
    try:
        log_file = Path("logs/autotrain_logs.txt")
        if log_file.exists():
            with log_file.open("r", encoding='utf-8') as f:
                log_content = f.read()
            logger.info("Logs retrieved successfully.")
            return log_content[-5000:]
        else:
            logger.info("No logs available yet.")
            return "Logs not available yet."
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        st.error(f"Error retrieving logs: {e}")
        return f"Error retrieving logs: {e}"

# 5.18. Real-time Log Updates
def real_time_logs(log_placeholder):
    while True:
        try:
            log_message = log_queue.get(timeout=1)
            log_placeholder.text(log_message)
        except queue.Empty:
            pass
        time.sleep(1)  # Reduce update frequency

# 5.19. Clear Logs Function
def clear_logs():
    try:
        log_file = Path("logs/autotrain_logs.txt")
        if log_file.exists():
            with log_file.open("w", encoding='utf-8') as f:
                f.truncate(0)
            logger.info("Logs cleared successfully.")
            st.success("Logs cleared successfully.")
            return "Logs cleared successfully."
        else:
            logger.info("No logs to clear.")
            st.info("No logs to clear.")
            return "No logs to clear."
    except Exception as e:
        logger.error(f"Error clearing logs: {e}")
        st.error(f"Error clearing logs: {e}")
        return f"Error clearing logs: {e}"

# 5.20. Evaluate Text Model
def evaluate_text_model(input_text, model_name, save_directory='output'):
    model_path = Path(save_directory) / model_name
    if not model_path.exists():
        return f"Model {model_name} not found."
    
    try:
        logger.info(f"Loading text model from {model_path} for evaluation...")
        device = 0 if is_cuda_available() else -1
        generator = pipeline('text-generation', model=str(model_path), device=device)
        logger.info("Text model loaded successfully for evaluation.")

        result = generator(input_text, max_length=100, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
        generated_text = result[0]['generated_text']
        logger.info("Text model evaluation completed successfully.")
        st.success("Text model evaluation completed successfully.")
        return generated_text
    except Exception as e:
        logger.error(f"Error during text model evaluation: {e}")
        return f"Error during evaluation: {str(e)}"

# 5.21. Evaluate Vision Model
def evaluate_vision_model(image, model_name, save_directory='output'):
    try:
        model_path = Path(save_directory) / model_name
        if not model_path.exists():
            logger.error(f"Model {model_name} not found in {save_directory}.")
            st.error(f"Model {model_name} not found.")
            return f"Model {model_name} not found."

        logger.info(f"Loading vision model from {model_path} for evaluation...")
        device = 0 if is_cuda_available() else -1
        segmentation_pipeline = pipeline("image-segmentation", model=str(model_path), device=device)
        logger.info("Vision model loaded successfully for evaluation.")

        result = segmentation_pipeline(image)
        st.image(image, caption="Input Image", use_column_width=True)
        st.image(result[0]['mask'], caption="Predicted Mask", use_column_width=True)
        st.write(f"**Label:** {result[0]['label']}")
        st.write(f"**Score:** {result[0]['score']:.4f}")
        logger.info("Vision model evaluation completed successfully.")
        st.success("Vision model evaluation completed successfully.")
        return
    except Exception as e:
        logger.error(f"Error during vision model evaluation: {e}")
        st.error(f"Error during vision model evaluation: {e}")
        return f"Error during vision model evaluation: {e}"

# 5.22. Submit Feedback
def submit_feedback(feedback_text):
    if not feedback_text.strip():
        logger.warning("No feedback provided.")
        st.warning("No feedback provided.")
        return "No feedback provided."
    try:
        feedback_dir = Path('feedback')
        feedback_dir.mkdir(exist_ok=True)
        feedback_file = feedback_dir / f"feedback_{int(time.time())}.txt"
        with feedback_file.open("w", encoding='utf-8') as f:
            f.write(feedback_text)
        logger.info(f"Feedback submitted successfully and saved to {feedback_file}.")
        st.success("Feedback submitted successfully. Thank you!")
        return "Feedback submitted successfully. Thank you!"
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        st.error(f"Error submitting feedback: {e}")
        return f"Error submitting feedback: {e}"

# 5.23. Deploy Model
def deploy_model(action, model_name, save_directory='output', task_type="text"):
    try:
        model_path = Path(save_directory) / model_name
        if not model_path.exists():
            logger.error(f"Model {model_name} not found in {save_directory}.")
            st.error(f"Model {model_name} not found.")
            return f"Model {model_name} not found."

        if action == 'export':
            export_path = Path('exported_models') / model_name
            export_path.mkdir(parents=True, exist_ok=True)
            if task_type == "text":
                subprocess.check_call(["cp", "-r", str(model_path), str(export_path)])
            else:
                subprocess.check_call(["cp", "-r", str(model_path), str(export_path)])
            logger.info(f"Model exported successfully to {export_path}.")
            st.success(f"Model exported successfully to {export_path}.")
            return f"Model exported successfully to {export_path}."
        elif action == 'push':
            api = HfApi()
            token_decrypted = decrypt_data(os.environ.get("HF_TOKEN_ENCRYPTED", "").encode(), st.session_state.get("encryption_key")) if os.getenv("HF_TOKEN_ENCRYPTED") else ""
            if not token_decrypted:
                logger.error("Hugging Face token not found or decryption failed.")
                st.error("Hugging Face token not found or decryption failed.")
                return "Hugging Face token not found or decryption failed."
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=f"{os.environ.get('HF_USERNAME', '')}/{model_name}",
                repo_type="model",
                token=token_decrypted
            )
            logger.info(f"Model {model_name} pushed to Hugging Face Hub successfully.")
            st.success(f"Model {model_name} pushed to Hugging Face Hub successfully.")
            return f"Model {model_name} pushed to Hugging Face Hub successfully."
        else:
            logger.warning("Invalid deployment action specified.")
            st.warning("Invalid deployment action.")
            return "Invalid deployment action."
    except Exception as e:
        logger.error(f"Error during model deployment: {e}")
        st.error(f"Error during model deployment: {e}")
        return f"Error during model deployment: {e}"

# 5.24. Export Model Functionality
def export_model(model_name, save_directory='output', task_type="text"):
    return deploy_model('export', model_name, save_directory, task_type)

# 5.25. Push Model to Hugging Face Hub Functionality
def push_model(model_name, save_directory='output', task_type="text"):
    if not os.environ.get("HF_USERNAME") or not os.environ.get("HF_TOKEN_ENCRYPTED"):
        return "Hugging Face credentials not set. Please set them in the Credentials tab."
    
    try:
        model_path = Path(save_directory) / model_name
        if not model_path.exists():
            logger.error(f"Model {model_name} not found in {save_directory}.")
            st.error(f"Model {model_name} not found.")
            return f"Model {model_name} not found."

        if action == 'export':
            export_path = Path('exported_models') / model_name
            export_path.mkdir(parents=True, exist_ok=True)
            if task_type == "text":
                subprocess.check_call(["cp", "-r", str(model_path), str(export_path)])
            else:
                subprocess.check_call(["cp", "-r", str(model_path), str(export_path)])
            logger.info(f"Model exported successfully to {export_path}.")
            st.success(f"Model exported successfully to {export_path}.")
            return f"Model exported successfully to {export_path}."
        elif action == 'push':
            api = HfApi()
            token_decrypted = decrypt_data(os.environ.get("HF_TOKEN_ENCRYPTED", "").encode(), st.session_state.get("encryption_key")) if os.getenv("HF_TOKEN_ENCRYPTED") else ""
            if not token_decrypted:
                logger.error("Hugging Face token not found or decryption failed.")
                st.error("Hugging Face token not found or decryption failed.")
                return "Hugging Face token not found or decryption failed."
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=f"{os.environ.get('HF_USERNAME', '')}/{model_name}",
                repo_type="model",
                token=token_decrypted
            )
            logger.info(f"Model {model_name} pushed to Hugging Face Hub successfully.")
            st.success(f"Model {model_name} pushed to Hugging Face Hub successfully.")
            return f"Model {model_name} pushed to Hugging Face Hub successfully."
        else:
            logger.warning("Invalid deployment action specified.")
            st.warning("Invalid deployment action.")
            return "Invalid deployment action."
    except Exception as e:
        logger.error(f"Error pushing model to Hub: {e}")
        return f"Error pushing model to Hub: {str(e)}"

# 5.26. Monitor System Resources
def monitor_resources():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
        return f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%"
    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")
        st.error(f"Error monitoring resources: {e}")
        return f"Error monitoring resources: {e}"

# 1. Enhanced Model Selection
def get_latest_models(task="text-generation", limit=10):
    api = HfApi()
    models = api.list_models(filter=task, sort="downloads", direction=-1, limit=limit)
    return [model.modelId for model in models]

# 2. Data Visualization
def visualize_text_data(df):
    word_counts = df['text'].str.split().str.len()
    fig = px.histogram(word_counts, nbins=20, title="Word Count Distribution")
    st.plotly_chart(fig)

def visualize_image_dataset(images):
    cols = st.columns(5)
    for idx, img in enumerate(images):
        cols[idx % 5].image(img.permute(1, 2, 0).numpy(), use_column_width=True)

# 3. Training Progress Visualization
def visualize_training_progress(metrics):
    df = pd.DataFrame(metrics)
    fig = px.line(df, x="epoch", y="loss", title="Training Loss")
    st.plotly_chart(fig)

# 4. Model Comparison
@st.cache_data
def compare_models(models, eval_data):
    results = []
    for model_name in models:
        model = load_model(model_name, "Text Generation")  # Adjust task as needed
        score = evaluate_model(model, eval_data)
        results.append({"model": model_name, "score": score})
    return pd.DataFrame(results)

# 5. Hyperparameter Optimization
def optimize_hyperparameters(model, train_data, param_space):
    def objective(config):
        # Train model with config and return validation score
        pass
    
    analysis = tune.run(
        objective,
        config=param_space,
        num_samples=10,
        resources_per_trial={"cpu": 2, "gpu": 0.5}
    )
    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    return best_config

# 6. Enhanced Export Options
def export_model(model, format="onnx"):
    if format == "onnx":
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy_input, "model.onnx")
    elif format == "torchscript":
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, "model.pt")

# 7. Improved Error Handling
def safe_execute(func, error_message):
    try:
        return func()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        logger.error(f"Error in {func.__name__}: {str(e)}")
        return None

# 8. Enhanced Performance Metrics
def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted')
    }

# 9. Fine-tuning Presets
FINETUNING_PRESETS = {
    "fast": {"lr": 5e-5, "epochs": 3, "batch_size": 16},
    "balanced": {"lr": 2e-5, "epochs": 5, "batch_size": 32},
    "thorough": {"lr": 1e-5, "epochs": 10, "batch_size": 64}
}

# 10. Responsive UI Improvements
st.set_page_config(layout="wide", page_title="🤗 Enhanced AutoTrain LLM & Vision")

# Add this new function for MAE preprocessing
def mae_preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Add this new function for MAE model creation
def create_mae_model():
    class MAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
                nn.Sigmoid(),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    return MAE()

# Main application code
def main():
    st.title("🤗 Enhanced AutoTrain LLM & Vision")
    
    tabs = st.tabs(["Data", "Training", "Evaluation", "Deployment", "Utilities"])
    
    with tabs[0]:
        data_management()
    
    with tabs[1]:
        model_training()
    
    with tabs[2]:
        model_evaluation()
    
    with tabs[3]:
        model_deployment()
    
    with tabs[4]:
        utilities()
def data_management():
    st.header("Data Management")
    data_type = st.radio("Select Data Type", ["Text", "Image"])
    
    if data_type == "Text":
        text_data_management()
    else:
        image_data_management()

def text_data_management():
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        visualize_text_data(df)

def image_data_management():
    uploaded_files = st.file_uploader("Upload Images or Zip File", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing uploaded images..."):
            processed_images = [mae_preprocess(Image.open(file)) for file in uploaded_files if file.type.startswith('image')]
            if processed_images:
                st.session_state['processed_images'] = processed_images
                st.success(f"Processed {len(processed_images)} images.")
                visualize_image_dataset(processed_images[:5])
            else:
                st.warning("No valid images found in the uploaded files.")

def model_training():
    st.header("Model Training")
    task = st.selectbox("Select Task", ["Text Generation", "Image Segmentation", "MAE Pretraining"])
    
    if task == "MAE Pretraining":
        if st.button("Start MAE Pretraining"):
            safe_execute(start_mae_pretraining, "Error starting MAE pretraining")
    else:
        model_name = st.selectbox("Select Model", get_latest_models(task))
        preset = st.selectbox("Fine-tuning Preset", list(FINETUNING_PRESETS.keys()))
        
        advanced = st.expander("Advanced Settings")
        with advanced:
            lr = st.number_input("Learning Rate", value=FINETUNING_PRESETS[preset]["lr"])
            epochs = st.number_input("Epochs", value=FINETUNING_PRESETS[preset]["epochs"])
            batch_size = st.number_input("Batch Size", value=FINETUNING_PRESETS[preset]["batch_size"])
        
        if st.checkbox("Optimize Hyperparameters"):
            with st.spinner("Optimizing hyperparameters..."):
                best_config = optimize_hyperparameters(model, train_data, {
                    "lr": tune.loguniform(1e-5, 1e-2),
                    "batch_size": tune.choice([16, 32, 64]),
                    "epochs": tune.choice([3, 5, 10])
                })
                st.success("Hyperparameter optimization complete.")
                st.write("Best configuration:", best_config)
                lr, batch_size, epochs = best_config["lr"], best_config["batch_size"], best_config["epochs"]
        
        if st.button("Start Training"):
            safe_execute(lambda: start_training(task, model_name, lr, epochs, batch_size), "Error starting training")

def model_evaluation():
    st.header("Model Evaluation")
    model_name = st.text_input("Model Name")
    eval_data = st.file_uploader("Upload Evaluation Data", type="csv")
    
    if st.button("Evaluate Model"):
        results = safe_execute(lambda: evaluate_model(model_name, eval_data), "Error evaluating model")
        if results:
            st.write(results)
            st.plotly_chart(px.bar(results, x=results.index, y=results.values, title="Model Performance"))

    if st.button("Compare Models"):
        models_to_compare = st.multiselect("Select models to compare", get_latest_models())
        eval_data = load_eval_data()  # Implement this function to load evaluation data
        results_df = compare_models(models_to_compare, eval_data)
        st.write(results_df)
        st.plotly_chart(px.bar(results_df, x="model", y="score", title="Model Comparison"))

def model_deployment():
    st.header("Model Deployment")
    model_name = st.text_input("Model Name")
    export_format = st.selectbox("Export Format", ["ONNX", "TorchScript"])
    
    if st.button("Export Model"):
        safe_execute(lambda: export_model(model_name, export_format.lower()), "Error exporting model")

def utilities():
    st.header("Utilities")
    if st.button("Monitor Resources"):
        resources = safe_execute(monitor_resources, "Error monitoring resources")
        if resources:
            st.write(resources)
    
    if st.button("Clear Logs"):
        safe_execute(clear_logs, "Error clearing logs")

def start_mae_pretraining():
    model = create_mae_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    data_dir = Path('data/images/images')
    image_files = list(data_dir.glob('*.pt'))

    if not image_files:
        st.error("No preprocessed image data found. Please upload images first.")
        return

    num_epochs = 10
    batch_size = 32

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch = torch.stack([torch.load(f) for f in batch_files])
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        st.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(image_files):.4f}")

    torch.save(model.state_dict(), 'mae_pretrained.pth')
    st.success("MAE pretraining completed. Model saved as 'mae_pretrained.pth'")

def start_training(task, model_name, lr, epochs, batch_size):
    model = load_model(model_name, task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = get_criterion(task)
    dataloader = get_dataloader(task, batch_size)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            inputs, targets = process_batch(batch, task)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            
            progress = (epoch * len(dataloader) + i + 1) / (epochs * len(dataloader))
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}")
        
        avg_loss = running_loss / len(dataloader)
        st.session_state['training_progress'].append({"epoch": epoch+1, "loss": avg_loss})
        visualize_training_progress(st.session_state['training_progress'])
    
    save_model(model, f"{model_name}_finetuned")
    st.success(f"Training completed. Model saved as '{model_name}_finetuned'")

def validate_training_params(lr, epochs, batch_size):
    if lr <= 0 or lr > 1:
        st.error("Learning rate must be between 0 and 1.")
        return False
    if epochs <= 0 or epochs > 100:
        st.error("Number of epochs must be between 1 and 100.")
        return False
    if batch_size <= 0 or batch_size > 512:
        st.error("Batch size must be between 1 and 512.")
        return False
    return True

# Use this function before starting training
if validate_training_params(lr, epochs, batch_size):
    start_training(task, model_name, lr, epochs, batch_size)

def cleanup():
    # Remove temporary files
    for file in Path("temp").glob("*"):
        file.unlink()
    logger.info("Temporary files cleaned up.")

atexit.register(cleanup)

# Use temp directory for temporary files
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)

# Use temp_dir when saving temporary files

if __name__ == "__main__":
    main()

