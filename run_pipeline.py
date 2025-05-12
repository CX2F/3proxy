
#!/usr/bin/env python3
# /run_pipeline.py
import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, capture_output=False):
    """Run a shell command and print output in real-time"""
    print(f"Running: {command}")
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=True, 
                                   capture_output=True, text=True)
            return result.stdout
        else:
            subprocess.run(command, shell=True, check=True)
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    dependencies = [
        "mediapipe",
        "mediapipe-model-maker",
        "torch",
        "transformers",
        "llama-cpp-python",
        "diffusers",
        "tiktoken",
        "pandas",
        "tqdm",
        "cryptography",
        "pillow"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        install = input("Install missing dependencies? (y/n): ")
        if install.lower() == 'y':
            for dep in missing:
                run_command(f"pip install {dep}")
            return True
        else:
            return False
    
    return True

def setup_environment():
    """Set up the environment for dataset processing and model training"""
    # Create necessary directories
    dirs = ["data", "output", "assets", "models", "generated_images", "trained/model"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Download MediaPipe models
    print("Downloading MediaPipe models...")
    run_command("python download_models.py")
    
    return True

def create_sample_data():
    """Create sample data files if the data directory is empty"""
    if not os.listdir("data"):
        print("Creating sample data files...")
        
        # Create sample text files
        sample_text1 = """
        This is a sample text file for keyword extraction and classification.
        It contains various terms related to cyber security and data analysis.
        Keywords: encryption, decryption, data breach, firewall, network security.
        This text should be processed, categorized, and labeled appropriately.
        """
        
        sample_text2 = """
        Machine learning algorithms can be categorized as supervised, unsupervised, or reinforcement learning.
        Deep learning, a subset of machine learning, uses neural networks with many layers.
        NLP techniques can analyze text data to extract information and insights.
        Keywords: machine learning, deep learning, neural networks, NLP, algorithms.
        """
        
        with open("data/sample1.txt", "w") as f:
            f.write(sample_text1)
        
        with open("data/sample2.txt", "w") as f:
            f.write(sample_text2)
        
        # Create sample CSV files
        import pandas as pd
        
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'text': [
                'This is text about cybersecurity vulnerabilities.',
                'Machine learning can detect anomalies in network traffic.',
                'Encryption algorithms protect sensitive data from unauthorized access.'
            ],
            'size': [100, 120, 150]
        })
        
        df2 = pd.DataFrame({
            'id': [4, 5, 6],
            'text': [
                'Natural language processing enables sentiment analysis of text data.',
                'Cloud computing provides scalable resources for data processing.',
                'Data visualization tools help identify patterns in complex datasets.'
            ],
            'url': [
                'https://example.com/nlp',
                'https://example.com/cloud',
                'https://example.com/dataviz'
            ]
        })
        
        df3 = pd.DataFrame({
            'id': [7, 8, 9],
            'label': ['security', 'ai', 'database'],
            'text': [
                'Firewalls prevent unauthorized access to private networks.',
                'Artificial intelligence is transforming how we analyze data.',
                'Database management systems organize and store structured data.'
            ]
        })
        
        df1.to_csv("data/sample1.csv", index=False)
        df2.to_csv("data/sample2.csv", index=False)
        df3.to_csv("data/sample3.csv", index=False)
        
        # Create a sample SQLite database
        import sqlite3
        
        conn = sqlite3.connect("data/dataset.db")
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY,
            category TEXT,
            text TEXT,
            keyword TEXT
        )
        ''')
        
        sample_data = [
            (1, 'security', 'Penetration testing identifies vulnerabilities in systems.', 'pentest'),
            (2, 'networking', 'Routers direct traffic between different network segments.', 'routing'),
            (3, 'programming', 'Python is widely used for data science applications.', 'python')
        ]
        
        cursor.executemany('INSERT INTO samples VALUES (?, ?, ?, ?)', sample_data)
        conn.commit()
        conn.close()
        
        print("Sample data created successfully")
        return True
    
    return False

def process_data():
    """Process data using the dataset processor"""
    print("Processing data...")
    run_command("python dataset_processor.py")
    return True

def run_model_training():
    """Run model training using gpt2-llm.c"""
    print("Training model...")
    
    # Check if gpt2-llm.c directory exists
    if not os.path.exists("gpt2-llm.c"):
        print("gpt2-llm.c directory not found. Cloning repository...")
        run_command("git clone https://github.com/karpathy/llm.c.git gpt2-llm.c")
    
    # Create or update token file for gpt2-llm.c using the structured data
    print("Preparing tokenized data for training...")
    
    # Check if structured_data.csv exists
    if os.path.exists("output/structured_data.csv"):
        # Import the right modules
        import pandas as pd
        import numpy as np
        import hashlib
        
        # Try to load tiktoken if available
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            has_tiktoken = True
        except ImportError:
            has_tiktoken = False
        
        # Load the structured data
        df = pd.read_csv("output/structured_data.csv")
        
        # Combine all text into a single string
        all_text = " ".join(df['text'].astype(str).tolist())
        
        # Create tokens directory
        tokens_dir = "gpt2-llm.c/dev/data/custom_dataset"
        os.makedirs(tokens_dir, exist_ok=True)
        
        # Tokenize the text
        if has_tiktoken:
            tokens = enc.encode(all_text)
            # Save tokens to the file
            token_file = os.path.join(tokens_dir, "custom_dataset.bin")
            
            # Use the train_gpt2.py script to convert tokens to binary
            from gpt2_llm_c.dev.data.data_common import write_datafile
            write_datafile(token_file, tokens)
            
            print(f"Tokenized {len(tokens)} tokens to {token_file}")
        else:
            print("tiktoken not available, attempting simple whitespace tokenization")
            # Simple whitespace tokenization (much less effective but works as fallback)
            tokens = all_text.split()
            
            # Create a simple binary file with tokens
            token_file = os.path.join(tokens_dir, "custom_dataset.bin")
            with open(token_file, 'wb') as f:
                # Simple formatting - not ideal but works as fallback
                token_bytes = bytes(str(tokens), 'utf-8')
                f.write(token_bytes)
            
            print(f"Tokenized {len(tokens)} tokens using simple method to {token_file}")
    else:
        print("output/structured_data.csv not found. Cannot create tokenized data.")
        return False
    
    # Choose between CPU and GPU training
    use_gpu = input("Use GPU for training? (y/n): ").lower() == 'y'
    
    # CD into gpt2-llm.c directory
    os.chdir("gpt2-llm.c")
    
    # Run the appropriate training command
    if use_gpu:
        # First compile the GPU training binary
        run_command("make train_gpt2fp32cu")
        
        # Run training
        run_command("./train_gpt2fp32cu -i dev/data/custom_dataset/custom_dataset.bin -n 1000 -s 512 -b 4")
    else:
        # First compile the CPU training binary
        run_command("make train_gpt2")
        
        # Set OMP_NUM_THREADS for CPU training
        os.environ["OMP_NUM_THREADS"] = "4"
        
        # Run training
        run_command("OMP_NUM_THREADS=4 ./train_gpt2 -i dev/data/custom_dataset/custom_dataset.bin -n 1000 -s 512 -b 4")
    
    # Return to original directory
    os.chdir("..")
    
    # Copy the trained model to our trained/model directory
    os.makedirs("trained/model/Deceptiv2S7", exist_ok=True)
    run_command("cp gpt2-llm.c/gpt2_124M.bin trained/model/Deceptiv2S7/")
    
    print("Model training completed!")
    return True

def generate_images():
    """Generate images using diffusion models"""
    print("Generating images with diffusion models...")
    
    # Import required modules
    from mediapipe_integration import DiffusionModelGenerator
    
    # Create generator
    generator = DiffusionModelGenerator()
    
    # Try to load one of the models
    models_to_try = [
        "black-forest-labs/FLUX.1-dev",
        "Heartsync/NSFW-Uncensored"
    ]
    
    model_loaded = False
    for model in models_to_try:
        print(f"Trying to load model: {model}")
        if generator.load_model(model):
            model_loaded = True
            print(f"Successfully loaded {model}")
            break
    
    if not model_loaded:
        print("Could not load any diffusion model. Please check your internet connection and dependencies.")
        return False
    
    # Generate some sample images
    prompts = [
        "A beautiful sunset over a mountain landscape",
        "A futuristic cityscape with flying cars",
        "A magical forest with glowing plants and creatures"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
        generator.generate_image(prompt, output_path="generated_images", seed=i+42)
    
    print("Image generation complete!")
    return True

def chat_with_llm():
    """Interactive chat with an LLM"""
    print("Starting chat with LLM...")
    
    # Import required modules
    from mediapipe_integration import LlamaInference
    
    # Create LLM instance
    llm = LlamaInference()
    
    # Models to try
    models = [
        ("afrideva/phi-2-uncensored-GGUF", "phi-2-uncensored.fp16.gguf"),
        ("XeroCodes/aurora-12b-gguf", "aurora-12b-f16.gguf"),
        ("Tesslate/UIGEN-T2-7B-Q8_0-GGUF", "uigen-t2-7b-3600-q8_0.gguf")
    ]
    
    # Try to load models
    model_loaded = False
    for repo_id, filename in models:
        print(f"Trying to load model: {repo_id}/{filename}")
        if llm.load_model(repo_id, filename):
            model_loaded = True
            print(f"Successfully loaded {repo_id}/{filename}")
            break
    
    if not model_loaded:
        print("Could not load any LLM model. Please check your internet connection and dependencies.")
        return False
    
    # Start chat loop
    print("\nInteractive chat session started. Type 'exit' to quit.")
    messages = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        response = llm.create_chat_completion(messages)
        
        if response and 'choices' in response and response['choices']:
            assistant_message = response['choices'][0]['message']['content']
            print(f"\nAssistant: {assistant_message}")
            
            # Add assistant message to history
            messages.append({"role": "assistant", "content": assistant_message})
        else:
            print("\nAssistant: Sorry, I couldn't generate a response.")
    
    print("Chat session ended.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run dataset creation and model training pipeline")
    parser.add_argument("--setup", action="store_true", help="Set up environment and download models")
    parser.add_argument("--sample-data", action="store_true", help="Create sample data files")
    parser.add_argument("--process", action="store_true", help="Process data files")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--generate-images", action="store_true", help="Generate images with diffusion models")
    parser.add_argument("--chat", action="store_true", help="Interactive chat with LLM")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    # Check dependencies
    if not check_dependencies():
        print("Missing dependencies. Please install them and try again.")
        return 1
    
    # Run all steps if --all is specified
    if args.all:
        args.setup = True
        args.sample_data = True
        args.process = True
        args.train = True
        args.generate_images = True
    
    # Set up environment
    if args.setup:
        if not setup_environment():
            print("Environment setup failed.")
            return 1
    
    # Create sample data if requested
    if args.sample_data:
        create_sample_data()
    
    # Process data
    if args.process:
        if not process_data():
            print("Data processing failed.")
            return 1
    
    # Train model
    if args.train:
        if not run_model_training():
            print("Model training failed.")
            return 1
    
    # Generate images
    if args.generate_images:
        if not generate_images():
            print("Image generation failed.")
            return 1
    
    # Interactive chat
    if args.chat:
        if not chat_with_llm():
            print("Chat session failed.")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
