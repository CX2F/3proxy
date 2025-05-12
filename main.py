
#!/usr/bin/env python3
# /main.py
import os
import argparse
import subprocess
import time
import shutil
from data_processor import main as process_data
from train_processor import run_training
from gpt4_classifier import classify_dataframe
import pandas as pd

def ensure_directory_structure():
    """Ensure all required directories exist"""
    directories = [
        "data",
        "trained/model",
        "generated_images",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory: {directory}")

def create_sample_data_if_needed():
    """Create sample data files if data directory is empty"""
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

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Training and Data Processing Pipeline")
    
    # Main operation groups
    parser.add_argument("--setup", action="store_true", help="Setup environment and directories")
    parser.add_argument("--process", action="store_true", help="Process and structure data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--classify", action="store_true", help="Classify data using GPT-4o")
    
    # Options for training
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training (if available)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    
    # Options for data processing
    parser.add_argument("--sample-data", action="store_true", help="Create sample data files")
    
    args = parser.parse_args()
    
    # Default behavior if no args specified
    if not any(vars(args).values()):
        args.setup = True
        args.process = True
        args.train = True
    
    # Setup environment
    if args.setup:
        print("Setting up environment...")
        ensure_directory_structure()
        
        # Clone gpt2-llm.c repository if not already present
        if not os.path.exists("gpt2-llm.c"):
            print("Cloning gpt2-llm.c repository...")
            subprocess.run(["git", "clone", "https://github.com/karpathy/llm.c.git", "gpt2-llm.c"])
        
        # Install requirements
        print("Installing requirements...")
        subprocess.run(["pip", "install", "-r", "requirements.txt"])
        
        # Install gpt2-llm.c requirements
        if os.path.exists("gpt2-llm.c/requirements.txt"):
            print("Installing gpt2-llm.c requirements...")
            subprocess.run(["pip", "install", "-r", "gpt2-llm.c/requirements.txt"])
    
    # Create sample data if requested
    if args.sample_data:
        create_sample_data_if_needed()
    
    # Process data
    if args.process:
        print("\n=== Processing Data ===")
        process_data()
    
    # Classify data with GPT-4o
    if args.classify:
        print("\n=== Classifying Data with GPT-4o ===")
        if os.path.exists("structured_data.csv"):
            df = pd.read_csv("structured_data.csv")
            print(f"Loaded {len(df)} records from structured_data.csv")
            
            # Check if we have an OpenAI API key
            if os.getenv("OPENAI_API_KEY"):
                classified_df = classify_dataframe(df)
                classified_df.to_csv("structured_data_classified.csv", index=False)
                print("Data classified and saved to structured_data_classified.csv")
            else:
                print("Warning: OpenAI API key not found. Set OPENAI_API_KEY to use classification.")
        else:
            print("Error: structured_data.csv not found. Run processing first.")
    
    # Train model
    if args.train:
        print("\n=== Training Model ===")
        run_training(args.gpu, args.iterations)
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()
