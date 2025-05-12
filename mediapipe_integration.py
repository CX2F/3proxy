
# /mediapipe_integration.py
import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, text
from mediapipe_model_maker import image_classifier
from mediapipe_model_maker.image_classifier import DataLoader, ModelSpec

class MediaPipeDatasetCreator:
    """Class to create and process datasets using MediaPipe tools"""
    
    def __init__(self, base_dir="data"):
        """Initialize the dataset creator with a base directory"""
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize MediaPipe tools
        self.image_embedder = None
        self.face_detector = None
        self.face_landmarker = None
        self.text_classifier = None
        self.language_detector = None
        
    def initialize_vision_tasks(self):
        """Initialize vision-related MediaPipe tasks"""
        # Set up image embedder
        base_options = python.BaseOptions(
            model_asset_path="assets/image_embedder.tflite")
        options = vision.ImageEmbedderOptions(base_options=base_options)
        self.image_embedder = vision.ImageEmbedder.create_from_options(options)
        
        # Set up face detector
        base_options = python.BaseOptions(
            model_asset_path="assets/face_detector.tflite")
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.face_detector = vision.FaceDetector.create_from_options(options)
        
        # Set up face landmarker
        base_options = python.BaseOptions(
            model_asset_path="assets/face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        print("Vision tasks initialized successfully")

    def initialize_text_tasks(self):
        """Initialize text-related MediaPipe tasks"""
        # Set up text classifier
        base_options = python.BaseOptions(
            model_asset_path="assets/text_classifier.tflite")
        options = text.TextClassifierOptions(base_options=base_options)
        self.text_classifier = text.TextClassifier.create_from_options(options)
        
        # Set up language detector
        base_options = python.BaseOptions(
            model_asset_path="assets/language_detector.tflite")
        options = text.LanguageDetectorOptions(base_options=base_options)
        self.language_detector = text.LanguageDetector.create_from_options(options)
        
        print("Text tasks initialized successfully")
    
    def process_image(self, image_path, save_features=True):
        """Process an image with all available vision tools"""
        # Load the image
        mp_image = mp.Image.create_from_file(image_path)
        results = {}
        
        # Extract embeddings
        if self.image_embedder:
            embedding_result = self.image_embedder.embed(mp_image)
            results['embedding'] = embedding_result.embeddings[0].float_embedding
        
        # Detect faces
        if self.face_detector:
            detection_result = self.face_detector.detect(mp_image)
            results['face_detection'] = detection_result
        
        # Get face landmarks
        if self.face_landmarker:
            landmarker_result = self.face_landmarker.detect(mp_image)
            results['face_landmarks'] = landmarker_result
        
        # Save features if requested
        if save_features:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            feature_path = os.path.join(self.base_dir, f"{base_filename}_features.npz")
            
            # Extract numpy arrays from results where possible
            saveable_results = {}
            if 'embedding' in results:
                saveable_results['embedding'] = np.array(results['embedding'])
            
            if saveable_results:
                np.savez(feature_path, **saveable_results)
                print(f"Saved features to {feature_path}")
                
        return results
    
    def process_text(self, text_content, save_features=True):
        """Process text with all available text tools"""
        results = {}
        
        # Classify text
        if self.text_classifier:
            classification_result = self.text_classifier.classify(text_content)
            results['classification'] = classification_result
        
        # Detect language
        if self.language_detector:
            language_result = self.language_detector.detect(text_content)
            results['language'] = language_result
        
        # Save features if requested
        if save_features:
            # Create a hash of the text to use as filename
            import hashlib
            text_hash = hashlib.md5(text_content.encode()).hexdigest()
            feature_path = os.path.join(self.base_dir, f"text_{text_hash}_features.json")
            
            import json
            # Convert results to JSON serializable format
            json_results = {}
            
            if 'classification' in results:
                categories = []
                for category in results['classification'].classifications[0].categories:
                    categories.append({
                        'index': category.index,
                        'score': category.score,
                        'category_name': category.category_name
                    })
                json_results['classification'] = categories
            
            if 'language' in results:
                languages = []
                for prediction in results['language'].predictions:
                    languages.append({
                        'language_code': prediction.language_code,
                        'probability': prediction.probability
                    })
                json_results['language'] = languages
                
            if json_results:
                with open(feature_path, 'w') as f:
                    json.dump(json_results, f, indent=2)
                print(f"Saved text features to {feature_path}")
                
        return results
    
    def create_image_classifier_dataset(self, data_dir, validation_split=0.2):
        """Create a dataset for MediaPipe image classifier training"""
        data = image_classifier.Dataset.from_folder(data_dir)
        train_data, validation_data = data.split(1 - validation_split)
        
        print(f"Created dataset with {len(train_data)} training and {len(validation_data)} validation samples")
        return train_data, validation_data
    
    def train_image_classifier(self, train_data, validation_data, model_name="MobileNetV2", epochs=10, export_dir="exported_model"):
        """Train an image classifier using MediaPipe Model Maker"""
        # Set up the model specification
        if model_name == "MobileNetV2":
            spec = image_classifier.SupportedModels.MOBILENET_V2
        elif model_name == "EfficientNetLite0":
            spec = image_classifier.SupportedModels.EFFICIENTNET_LITE0
        elif model_name == "EfficientNetLite2":
            spec = image_classifier.SupportedModels.EFFICIENTNET_LITE2
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Set hyperparameters
        hparams = image_classifier.HParams(
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
            export_dir=export_dir
        )
        
        # Configure training options
        options = image_classifier.ImageClassifierOptions(
            supported_model=spec,
            hparams=hparams
        )
        
        # Create and train the model
        model = image_classifier.ImageClassifier.create(
            train_data=train_data,
            validation_data=validation_data,
            options=options
        )
        
        # Evaluate the model
        eval_result = model.evaluate(validation_data)
        print(f"Model evaluation - Loss: {eval_result[0]}, Accuracy: {eval_result[1]}")
        
        # Export the model
        model.export_model()
        print(f"Model exported to {export_dir}")
        
        return model

# Diffusion models integration
class DiffusionModelGenerator:
    """Class to generate images using diffusion models"""
    
    def __init__(self, model_path=None):
        """Initialize the diffusion model generator"""
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path, uncensored_lora=None):
        """Load a diffusion model"""
        try:
            from diffusers import DiffusionPipeline
            import torch
            
            # Load the base model
            self.model = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Load uncensored LoRA weights if specified
            if uncensored_lora:
                self.model.load_lora_weights(uncensored_lora)
                print(f"Loaded uncensored LoRA weights from {uncensored_lora}")
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            
            print(f"Model loaded successfully on {device}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def generate_image(self, prompt, output_path="generated_images", seed=None, save_image=True):
        """Generate an image using the loaded diffusion model"""
        if not self.model:
            print("No model loaded. Please load a model first.")
            return None
        
        try:
            import torch
            
            # Create output directory if it doesn't exist
            if save_image:
                os.makedirs(output_path, exist_ok=True)
            
            # Set seed for reproducibility if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate image
            print(f"Generating image with prompt: {prompt}")
            image = self.model(prompt).images[0]
            
            # Save image
            if save_image:
                timestamp = int(time.time())
                filename = f"{output_path}/generated_{timestamp}.png"
                image.save(filename)
                print(f"Image saved to {filename}")
            
            return image
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

# LLama integration for chat completions
class LlamaInference:
    """Class for running inference with Llama models"""
    
    def __init__(self):
        """Initialize the Llama inference engine"""
        self.model = None
    
    def load_model(self, repo_id, filename):
        """Load a Llama model from Hugging Face"""
        try:
            from llama_cpp import Llama
            
            self.model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
            )
            print(f"Loaded model {repo_id}/{filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def generate_completion(self, prompt, max_tokens=256, temperature=0.7):
        """Generate a completion using the loaded model"""
        if not self.model:
            print("No model loaded. Please load a model first.")
            return None
        
        try:
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            print(f"Error generating completion: {str(e)}")
            return None
    
    def create_chat_completion(self, messages):
        """Generate a chat completion using the loaded model"""
        if not self.model:
            print("No model loaded. Please load a model first.")
            return None
        
        try:
            response = self.model.create_chat_completion(messages=messages)
            return response
        except Exception as e:
            print(f"Error generating chat completion: {str(e)}")
            return None

# Example usage:
# dataset_creator = MediaPipeDatasetCreator()
# dataset_creator.initialize_vision_tasks()
# dataset_creator.initialize_text_tasks()
# 
# image_generator = DiffusionModelGenerator()
# image_generator.load_model("Heartsync/NSFW-Uncensored")
# image_generator.generate_image("A beautiful sunset over a beach")
# 
# llm = LlamaInference()
# llm.load_model("afrideva/phi-2-uncensored-GGUF", "phi-2-uncensored.fp16.gguf")
# response = llm.create_chat_completion([{"role": "user", "content": "What is the capital of France?"}])
