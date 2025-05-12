
# /download_models.py
import os
import urllib.request
import zipfile
import tarfile
import gzip
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_extract_model(url, output_dir, model_name=None):
    """Download and extract a model file"""
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name is None:
        model_name = url.split('/')[-1]
    
    output_path = os.path.join(output_dir, model_name)
    
    # Download the file
    print(f"Downloading {model_name}...")
    download_url(url, output_path)
    
    # Extract if it's a compressed file
    if model_name.endswith('.zip'):
        print(f"Extracting {model_name}...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(output_path)
    elif model_name.endswith('.tar.gz') or model_name.endswith('.tgz'):
        print(f"Extracting {model_name}...")
        with tarfile.open(output_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
        os.remove(output_path)
    elif model_name.endswith('.gz') and not model_name.endswith('.tar.gz'):
        print(f"Extracting {model_name}...")
        with gzip.open(output_path, 'rb') as f_in:
            with open(output_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(output_path)
    
    print(f"Downloaded and processed {model_name}")
    return output_path

def download_mediapipe_models():
    """Download MediaPipe models for various tasks"""
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # MediaPipe vision models
    vision_models = {
        "image_embedder.tflite": "https://storage.googleapis.com/mediapipe-assets/mobilenet_v3_small_075_224_embedder.tflite",
        "face_detector.tflite": "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite",
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task",
    }
    
    # MediaPipe text models
    text_models = {
        "text_classifier.tflite": "https://storage.googleapis.com/mediapipe-assets/text_classifier_q.tflite",
        "language_detector.tflite": "https://storage.googleapis.com/mediapipe-assets/language_detector.tflite",
    }
    
    # Download vision models
    for model_name, url in vision_models.items():
        output_path = os.path.join(assets_dir, model_name)
        if not os.path.exists(output_path):
            download_url(url, output_path)
            print(f"Downloaded {model_name}")
        else:
            print(f"{model_name} already exists, skipping download")
    
    # Download text models
    for model_name, url in text_models.items():
        output_path = os.path.join(assets_dir, model_name)
        if not os.path.exists(output_path):
            download_url(url, output_path)
            print(f"Downloaded {model_name}")
        else:
            print(f"{model_name} already exists, skipping download")
    
    print("All MediaPipe models downloaded successfully")

def download_diffusion_model():
    """Download diffusion model for image generation"""
    models_dir = "models/diffusion"
    os.makedirs(models_dir, exist_ok=True)
    
    # We'll just create a placeholder file with instructions since these models are large
    instructions = """
    Diffusion models are too large to download directly in this script.
    
    To use the NSFW-Uncensored model, you would run:
    
    ```python
    from diffusers import DiffusionPipeline
    
    pipe = DiffusionPipeline.from_pretrained("Heartsync/NSFW-Uncensored")
    ```
    
    For FLUX with uncensored LoRA:
    
    ```python
    from diffusers import DiffusionPipeline
    
    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    pipe.load_lora_weights("enhanceaiteam/Flux-Uncensored-V2")
    ```
    
    These will be automatically downloaded when used for the first time.
    """
    
    with open(os.path.join(models_dir, "README.md"), "w") as f:
        f.write(instructions)
    
    print("Created instructions for downloading diffusion models")

def download_llama_models():
    """Create instructions for downloading Llama models"""
    models_dir = "models/llama"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a placeholder file with instructions
    instructions = """
    Llama models are too large to download directly in this script.
    
    To use phi-2-uncensored, you would run:
    
    ```python
    from llama_cpp import Llama
    
    llm = Llama.from_pretrained(
        repo_id="afrideva/phi-2-uncensored-GGUF",
        filename="phi-2-uncensored.fp16.gguf",
    )
    ```
    
    For Aurora-12B:
    
    ```python
    from llama_cpp import Llama
    
    llm = Llama.from_pretrained(
        repo_id="XeroCodes/aurora-12b-gguf",
        filename="aurora-12b-f16.gguf",
    )
    ```
    
    For UIGEN-T2-7B:
    
    ```python
    from llama_cpp import Llama
    
    llm = Llama.from_pretrained(
        repo_id="Tesslate/UIGEN-T2-7B-Q8_0-GGUF",
        filename="uigen-t2-7b-3600-q8_0.gguf",
    )
    ```
    
    For Llama-3.2-8X3B-MOE:
    
    ```python
    from llama_cpp import Llama
    
    llm = Llama.from_pretrained(
        repo_id="DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B-GGUF",
        filename="L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-IQ4_XS.gguf",
    )
    ```
    
    These will be downloaded when used for the first time.
    """
    
    with open(os.path.join(models_dir, "README.md"), "w") as f:
        f.write(instructions)
    
    print("Created instructions for downloading Llama models")

if __name__ == "__main__":
    download_mediapipe_models()
    download_diffusion_model()
    download_llama_models()
    print("All downloads completed!")
