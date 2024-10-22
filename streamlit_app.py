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
import asyncio

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
logger = logging.getLogger("LLM_Vision_AutoTrain")
logger.setLevel(logging.DEBUG)
Path("logs").mkdir(exist_ok=True)
file_handler = logging.FileHandler("logs/autotrain_logs.txt", mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_queue = queue.Queue()

# --------------------------- 5. Utility Functions ----------------------------------

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
        save_text_data(generated_data)
        st.success("Text data generation completed successfully.")
        return pd.DataFrame(generated_data)
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

# 5.9. Upload Image Dataset
def upload_image_data(images, masks):
    if images is None or masks is None:
        logger.warning("No images or masks uploaded.")
        st.warning("Both images and masks must be uploaded.")
        return "Both images and masks must be uploaded."
    try:
        logger.info("Uploading image data...")
        data_dir = Path('data/images')
        images_dir = data_dir / 'images'
        masks_dir = data_dir / 'masks'
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in images:
            img = Image.open(img_file)
            img.save(images_dir / img_file.name)
        
        for mask_file in masks:
            mask = Image.open(mask_file)
            mask.save(masks_dir / mask_file.name)
        
        logger.info("Image data uploaded successfully.")
        st.success("Image data uploaded and saved successfully.")
        return "Image data uploaded and saved successfully."
    except Exception as e:
        logger.error(f"Error uploading image data: {e}")
        st.error(f"Error uploading image data: {e}")
        return f"Error uploading image data: {e}"

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
            if not img.name.endswith(('.png', '.jpg', '.jpeg')):
                logger.warning(f"Image file {img.name} is not a supported format.")
                return False, f"Unsupported image format: {img.name}"
            if not mask.name.endswith(('.png', '.jpg', '.jpeg')):
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
        images = list(images_dir.glob('*'))
        masks = list(masks_dir.glob('*'))
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
def get_vision_model_list():
    try:
        logger.info("Fetching vision model list from Hugging Face...")
        api = HfApi()
        models = api.list_models(
            filter="segmentation",
            sort="downloads",
            direction=-1,
            limit=50
        )
        model_names = [model.modelId for model in models]
        recommended_models = [m for m in model_names if any(x in m.lower() for x in ["unet", "medsam", "maskrcnn"])]
        if not recommended_models:
            recommended_models = model_names[:10]
        logger.info("Vision model list fetched successfully.")
        return recommended_models
    except Exception as e:
        logger.error(f"Error fetching vision model list: {e}")
        return ["facebook/maskrcnn_resnet50_fpn"]

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
        start_autotrain_process(conf_file)
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
        if task_type == "text":
            process = st.session_state.get('autotrain_process')
        else:
            process = st.session_state.get('autotrain_vision_process')
        if process and process.poll() is None:
            if os.name == 'nt':
                # Windows does not support SIGSTOP, use ctypes to suspend the process
                import ctypes
                PROCESS_SUSPEND_RESUME = 0x0800
                handle = ctypes.windll.kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, process.pid)
                if handle:
                    ctypes.windll.kernel32.SuspendThread(handle)
                    ctypes.windll.kernel32.CloseHandle(handle)
                    logger.info("Training paused.")
                    st.success("Training paused successfully.")
                    return "Training paused successfully."
                else:
                    logger.error("Failed to acquire process handle for pausing.")
                    st.error("Failed to acquire process handle for pausing.")
                    return "Failed to acquire process handle for pausing."
            else:
                process.send_signal(subprocess.signal.SIGSTOP)
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
            current_logs = log_placeholder.text
            new_logs = f"{current_logs}\n{log_message}"
            log_placeholder.text = new_logs
        except queue.Empty:
            pass
        time.sleep(0.1)

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
    try:
        model_path = Path(save_directory) / model_name
        if not model_path.exists():
            logger.error(f"Model {model_name} not found in {save_directory}.")
            st.error(f"Model {model_name} not found.")
            return f"Model {model_name} not found."

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
        st.error(f"Error during text model evaluation: {e}")
        return f"Error during text model evaluation: {e}"

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
    return deploy_model('push', model_name, save_directory, task_type)

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

# --------------------------- 6. Streamlit Interface ----------------------------------

st.set_page_config(page_title="🤗 AutoTrain LLM & Vision - Enhanced Application", layout="wide")
st.title("🤗 AutoTrain LLM & Vision - Enhanced Application")
st.markdown("""
Easily generate, add, preview data, train, evaluate, and deploy LLMs and Vision Models with dynamic logging and comprehensive management tools.
""")

with st.expander("📝 Getting Started", expanded=True):
    st.markdown("""
    Welcome to the **AutoTrain LLM & Vision Application**! Here's a quick guide to help you get started:

    1. **Set Credentials**: Enter your Hugging Face token and username to enable model uploading.
    2. **Data Management**: Generate synthetic text data or upload your own text/image datasets.
    3. **Training**: Configure and start the training process for text or image models.
    4. **Evaluation**: Test your trained models with custom inputs.
    5. **Logs**: Monitor training progress and view logs in real-time.
    6. **Deployment**: Deploy your models to production environments.
    7. **Utility**: Manage logs, configurations, and provide feedback.
    """)
    
tabs = st.tabs(["📊 Data Management", "🔐 Credentials", "🎓 Training", "🧪 Evaluation", "📜 Logs", "🚀 Deployment", "🔧 Utility", "💬 Feedback", "❓ Help"])

# 📊 Data Management Tab
with tabs[0]:
    st.header("📊 Data Management")
    data_type = st.radio("Select Data Type", options=["Text", "Image"], horizontal=True)
    
    if data_type == "Text":
        st.subheader("### Text Data Management")
        text_management_col1, text_management_col2 = st.columns(2)
        
        with text_management_col1:
            st.markdown("#### Generate Synthetic Text Data")
            prompt = st.text_input("Data Generation Prompt", placeholder="Enter a prompt for data generation...")
            num_rows = st.number_input("Number of Rows", min_value=1, max_value=10000, value=5, step=1)
            model_name = st.selectbox("Model Name", options=get_model_list(), index=get_model_list().index("gpt2") if "gpt2" in get_model_list() else 0)
            if st.button("Generate Text Data"):
                generate_text_data(prompt, num_rows, model_name)
            st.markdown("---")
        
        with text_management_col2:
            st.markdown("#### Manual Text Data Entry")
            input_text = st.text_area("Manual Data Entry", placeholder="Enter text to add...")
            if st.button("Add Text Data"):
                add_text_data(input_text)
            st.markdown("---")
        
        st.markdown("#### Upload Text Data")
        uploaded_text_file = st.file_uploader("Upload Text CSV File", type=["csv"])
        if st.button("Upload Text Data"):
            upload_text_data(uploaded_text_file)
        st.markdown("---")
        
        st.markdown("#### Preview Current Text Data")
        if st.button("Preview Text Data"):
            df_preview = pd.read_csv(Path('data/text/train_text.csv')) if Path('data/text/train_text.csv').exists() else pd.DataFrame(columns=["text"])
            st.dataframe(df_preview.head(10))
    
    elif data_type == "Image":
        st.subheader("### Image Data Management")
        image_management_col1, image_management_col2 = st.columns(2)
        
        with image_management_col1:
            st.markdown("#### Upload Image Dataset")
            uploaded_images = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            uploaded_masks = st.file_uploader("Upload Corresponding Masks", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if st.button("Upload Image Data"):
                valid, message = validate_image_uploaded_files(uploaded_images, uploaded_masks)
                if valid:
                    upload_image_data(uploaded_images, uploaded_masks)
        with image_management_col2:
            st.markdown("#### Preview Image Dataset")
            if st.button("Preview Image Data"):
                preview_image_data()
        st.markdown("---")

# 🔐 Credentials Tab
with tabs[1]:
    st.header("🔐 Hugging Face Credentials")
    st.markdown("[Get your Hugging Face token here](https://huggingface.co/settings/tokens)")
    token_input = st.text_input("Token", type="password", placeholder="Enter your Hugging Face Token...")
    username_input = st.text_input("Username", placeholder="Enter your Hugging Face Username...")
    if st.button("Set Credentials"):
        set_hf_credentials(token_input, username_input)

# 🎓 Training Tab
with tabs[2]:
    st.header("🎓 Training Configuration")
    task_type = st.radio("Select Task Type", options=["Text Generation", "Image Segmentation"], horizontal=True)
    
    if task_type == "Text Generation":
        st.subheader("### Text Model Training Configuration")
        model_name_train = st.text_input("Base Model Name", value="gpt2", placeholder="Enter the base model name...")
        project_name = st.text_input("Project Name", value="MyTextProject", placeholder="Enter your project name...")
        push_to_hub = st.checkbox("Push Model to Hugging Face Hub", value=True)
        with st.expander("🔧 Advanced Settings", expanded=False):
            lr = st.number_input("Learning Rate", value=2e-4, format="%.5f", step=1e-5, help="Learning rate for the optimizer.")
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=1, step=1, help="Number of training epochs.")
            batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=2, step=1, help="Batch size per training step.")
            gradient_accumulation = st.number_input("Gradient Accumulation", min_value=1, max_value=100, value=4, step=1, help="Number of steps to accumulate gradients.")
            mixed_precision = st.selectbox("Mixed Precision", options=["fp16", "bf16", "none"], index=0, help="Enable mixed precision training for faster computation.")
            peft = st.checkbox("Use PEFT", value=True, help="Enable Parameter-Efficient Fine-Tuning techniques.")
            quantization = st.selectbox("Quantization", options=["int4", "int8", "none"], index=0, help="Apply quantization to reduce model size.")
            weight_decay = st.number_input("Weight Decay", value=0.01, format="%.4f", step=0.001, help="Weight decay for regularization.")
            warmup_steps = st.number_input("Warmup Steps", min_value=0, max_value=100000, value=500, step=100, help="Number of warmup steps for the learning rate scheduler.")
            logging_steps = st.number_input("Logging Steps", min_value=0, max_value=100000, value=100, step=100, help="Frequency of logging during training.")
        if st.button("Create and Start AutoTrain for Text"):
            training_params = aggregate_training_params(lr, epochs, batch_size, gradient_accumulation, mixed_precision, peft, quantization, weight_decay, warmup_steps, logging_steps)
            create_autotrain_configuration(model_name_train, project_name, push_to_hub, training_params)
    
    elif task_type == "Image Segmentation":
        st.subheader("### Vision Model Training Configuration")
        model_name_train = st.text_input("Base Model Name", value="facebook/maskrcnn_resnet50_fpn", placeholder="Enter the base model name...")
        project_name = st.text_input("Project Name", value="MyImageProject", placeholder="Enter your project name...")
        push_to_hub = st.checkbox("Push Model to Hugging Face Hub", value=True)
        with st.expander("🔧 Advanced Settings", expanded=False):
            lr = st.number_input("Learning Rate", value=2e-4, format="%.5f", step=1e-5, help="Learning rate for the optimizer.")
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1, help="Number of training epochs.")
            batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=4, step=1, help="Batch size per training step.")
            gradient_accumulation = st.number_input("Gradient Accumulation", min_value=1, max_value=100, value=2, step=1, help="Number of steps to accumulate gradients.")
            mixed_precision = st.selectbox("Mixed Precision", options=["fp16", "bf16", "none"], index=0, help="Enable mixed precision training for faster computation.")
            weight_decay = st.number_input("Weight Decay", value=0.01, format="%.4f", step=0.001, help="Weight decay for regularization.")
            warmup_steps = st.number_input("Warmup Steps", min_value=0, max_value=100000, value=500, step=100, help="Number of warmup steps for the learning rate scheduler.")
            logging_steps = st.number_input("Logging Steps", min_value=0, max_value=100000, value=100, step=100, help="Frequency of logging during training.")
        if st.button("Create and Start AutoTrain for Vision"):
            training_params = aggregate_training_params(lr, epochs, batch_size, gradient_accumulation, mixed_precision, False, "none", weight_decay, warmup_steps, logging_steps)
            create_autotrain_vision_configuration(model_name_train, project_name, push_to_hub, training_params)

# 🧪 Evaluation Tab
with tabs[3]:
    st.header("🧪 Model Evaluation")
    task_type = st.radio("Select Task Type for Evaluation", options=["Text Generation", "Image Segmentation"], horizontal=True)
    
    if task_type == "Text Generation":
        st.subheader("### Text Model Evaluation")
        eval_input_text = st.text_area("Input Text", placeholder="Enter text to evaluate the model...", height=150)
        eval_model_name = st.text_input("Model Name", value="gpt2", placeholder="Enter the model name to evaluate...")
        if st.button("Evaluate Text Model"):
            evaluate_text_model(eval_input_text, eval_model_name)
    
    elif task_type == "Image Segmentation":
        st.subheader("### Vision Model Evaluation")
        uploaded_eval_image = st.file_uploader("Upload Image for Evaluation", type=["png", "jpg", "jpeg"])
        eval_model_name = st.text_input("Model Name", value="facebook/maskrcnn_resnet50_fpn", placeholder="Enter the model name to evaluate...")
        if st.button("Evaluate Vision Model"):
            if uploaded_eval_image is not None:
                image = Image.open(uploaded_eval_image)
                evaluate_vision_model(image, eval_model_name)
            else:
                st.warning("Please upload an image for evaluation.")

# 📜 Logs Tab
with tabs[4]:
    st.header("📜 Training Logs")
    log_placeholder = st.empty()
    if 'log_thread_started' not in st.session_state:
        threading.Thread(target=real_time_logs, args=(log_placeholder,), daemon=True).start()
        st.session_state.log_thread_started = True
    st.text_area("AutoTrain Logs", value="", height=400, key="logs_output")
    if st.button("Refresh Logs"):
        logs = get_logs()
        st.session_state.logs_output = logs

# 🚀 Deployment Tab
with tabs[5]:
    st.header("🚀 Model Deployment Options")
    st.markdown("""
    After training, you can deploy your model in various ways:

    - **Export Model**: Download the trained model for local use.
    - **Push to Hugging Face Hub**: Automatically push your model to the Hugging Face repository.
    """)
    deployment_col1, deployment_col2 = st.columns(2)
    
    with deployment_col1:
        st.subheader("### Export Model")
        task_type_export = st.radio("Select Task Type for Export", options=["Text Generation", "Image Segmentation"], horizontal=True)
        model_name_export = st.text_input("Model Name for Export", value="gpt2", placeholder="Enter the model name to export...")
        if st.button("Export Model"):
            export_model(model_name_export, task_type=task_type_export.lower().replace(" ", "_"))
    
    with deployment_col2:
        st.subheader("### Push Model to Hugging Face Hub")
        task_type_push = st.radio("Select Task Type for Push", options=["Text Generation", "Image Segmentation"], horizontal=True)
        model_name_push = st.text_input("Model Name to Push", value="gpt2", placeholder="Enter the model name to push...")
        if st.button("Push to Hugging Face Hub"):
            push_model(model_name_push, task_type=task_type_push.lower().replace(" ", "_"))

# 🔧 Utility Tab
with tabs[6]:
    st.header("🔧 Log Management and Application Settings")
    if st.button("Clear Logs"):
        clear_logs()
    pause_col, cancel_col = st.columns(2)
    with pause_col:
        if st.button("Pause Text Training"):
            pause_training(task_type="text")
    with cancel_col:
        if st.button("Cancel Text Training"):
            cancel_training(task_type="text")
    pause_col_v, cancel_col_v = st.columns(2)
    with pause_col_v:
        if st.button("Pause Vision Training"):
            pause_training(task_type="vision")
    with cancel_col_v:
        if st.button("Cancel Vision Training"):
            cancel_training(task_type="vision")
    st.markdown("---")
    st.subheader("### Configuration Management")
    backup_conf = st.text_input("Backup Config Filename", placeholder="Enter backup config filename...")
    if st.button("Backup Configuration"):
        backup_configuration()
    restore_conf = st.text_input("Restore Config Filename", placeholder="Enter backup config filename to restore...")
    if st.button("Restore Configuration"):
        restore_configuration(restore_conf)
    st.markdown("---")
    st.subheader("### Resource Monitoring")
    if st.button("Monitor System Resources"):
        resources = monitor_resources()
        st.text(f"System Resources:\n{resources}")

# 💬 Feedback Tab
with tabs[7]:
    st.header("💬 Submit Feedback or Report Issues")
    feedback_text = st.text_area("Feedback", placeholder="Enter your feedback or issue report here...", height=200)
    if st.button("Submit Feedback"):
        submit_feedback(feedback_text)

# ❓ Help Tab
with tabs[8]:
    st.header("❓ Help and Documentation")
    st.markdown("""
    **Frequently Asked Questions (FAQs):**

    1. **How do I obtain a Hugging Face token?**
       - Visit [Hugging Face Tokens](https://huggingface.co/settings/tokens) to generate a new token.

    2. **Can I use my own dataset?**
       - Yes, you can generate synthetic text data or upload your own text/image datasets under the Data Management tab.

    3. **How do I deploy my trained model?**
       - After training, use the Deployment tab to export or push your model to Hugging Face Hub.

    **Documentation:**
    - Refer to the [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) for more details on model usage and customization.
    - Refer to the [Hugging Face AutoTrain Documentation](https://huggingface.co/docs/autotrain/) for managing training tasks.

    **Tutorials:**
    - [Getting Started with AutoTrain](https://huggingface.co/docs/autotrain/)
    - [Fine-Tuning Models](https://huggingface.co/docs/transformers/training)
    - [Image Segmentation with Hugging Face](https://huggingface.co/docs/transformers/main/en/task_summary#image-segmentation)
    """)

st.markdown("""
---
**Version:** 2.0.0 | **Credits:** [Transformers](https://github.com/huggingface/transformers), [Streamlit](https://streamlit.io/), [Hugging Face Hub](https://huggingface.co/)
""")

# --------------------------- 7. Launch Application ---------------------------
# Streamlit apps run automatically, so no need for a launch command.
