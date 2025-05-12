
# /train_processor.py
import os
import argparse
import subprocess
import sys
import shutil

def setup_gpt2_training():
    """Setup and prepare GPT-2 training with our structured data"""
    # Ensure we're in the right directory
    if not os.path.exists("gpt2-llm.c"):
        print("Error: gpt2-llm.c directory not found")
        return False
    
    # Create data directory inside gpt2-llm.c/dev/data if it doesn't exist
    data_dir = os.path.join("gpt2-llm.c", "dev", "data", "custom_dataset")
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy our tokenized data if it exists
    tokens_path = "trained/model/tokens.bin"
    if os.path.exists(tokens_path):
        shutil.copy(tokens_path, os.path.join(data_dir, "custom_dataset.bin"))
        print(f"Copied tokenized data to {data_dir}")
    else:
        print("Warning: No tokenized data found at trained/model/tokens.bin")
    
    return True

def run_training(use_gpu=False, iterations=1000, sequence_length=512, batch_size=4):
    """Run GPT-2 training using our data"""
    if not setup_gpt2_training():
        return False
    
    # Change to gpt2-llm.c directory
    os.chdir("gpt2-llm.c")
    
    # First, download the pretrained GPT-2 weights and prepare them for C
    print("Preparing model weights...")
    subprocess.run([sys.executable, "train_gpt2.py"])
    
    # Compile the appropriate training binary
    print("Compiling training binary...")
    if use_gpu:
        subprocess.run(["make", "train_gpt2fp32cu"])
        training_binary = "./train_gpt2fp32cu"
    else:
        subprocess.run(["make", "train_gpt2"])
        training_binary = "./train_gpt2"
    
    # Run the training
    print(f"Starting training with {'GPU' if use_gpu else 'CPU'}...")
    training_cmd = [
        training_binary,
        f"-i dev/data/custom_dataset/custom_dataset.bin",
        f"-n {iterations}",
        f"-s {sequence_length}",
        f"-b {batch_size}"
    ]
    
    if not use_gpu:
        # Set OMP_NUM_THREADS for CPU training
        os.environ["OMP_NUM_THREADS"] = "4"
    
    result = subprocess.run(' '.join(training_cmd), shell=True)
    
    if result.returncode == 0:
        print("Training completed successfully")
        # Copy the trained model from gpt2-llm.c to our trained/model directory
        model_dir = "../trained/model/Deceptiv2S7"
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy("gpt2_124M.bin", os.path.join(model_dir, "gpt2_124M.bin"))
        print(f"Trained model saved to {model_dir}")
        return True
    else:
        print("Training failed with error code", result.returncode)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 model with structured data")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training (if available)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--sequence_length", type=int, default=512, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    
    args = parser.parse_args()
    
    run_training(args.gpu, args.iterations, args.sequence_length, args.batch_size)
