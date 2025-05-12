
# /data_processor.py
import os
import json
import sqlite3
import pandas as pd
import numpy as np
import hashlib
import base64
from cryptography.fernet import Fernet
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import requests
from datetime import datetime
import glob

# Ensure required directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("trained/model", exist_ok=True)

# Initialize GPT-2 tokenizer for token counts
enc = tiktoken.get_encoding("gpt2")

# Initialize encryption
def generate_key():
    """Generate a key for encryption/decryption"""
    try:
        with open("crypto.key", "rb") as key_file:
            key = key_file.read()
    except FileNotFoundError:
        key = Fernet.generate_key()
        with open("crypto.key", "wb") as key_file:
            key_file.write(key)
    return key

crypto_key = generate_key()
cipher_suite = Fernet(crypto_key)

def encrypt_text(text):
    """Encrypt text using Fernet"""
    if isinstance(text, str):
        encrypted = cipher_suite.encrypt(text.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    return ""

def decrypt_text(encrypted_text):
    """Decrypt text using Fernet"""
    try:
        if isinstance(encrypted_text, str):
            decoded = base64.urlsafe_b64decode(encrypted_text)
            decrypted = cipher_suite.decrypt(decoded)
            return decrypted.decode()
    except Exception as e:
        print(f"Decryption error: {e}")
    return ""

# Function to load and merge text files
def load_text_files(directory="data"):
    """Load and merge all text files in the directory"""
    content = []
    files = glob.glob(f"{directory}/*.txt")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                content.append({
                    'filename': os.path.basename(file_path),
                    'content': file_content,
                    'size': os.path.getsize(file_path)
                })
                print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return pd.DataFrame(content)

# Function to load CSV files
def load_csv_files(directory="data"):
    """Load and merge all CSV files in the directory"""
    dataframes = []
    files = glob.glob(f"{directory}/*.csv")
    
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            # Add filename column
            df['filename'] = os.path.basename(file_path)
            dataframes.append(df)
            print(f"Loaded CSV: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return pd.DataFrame()

# Load SQLite database
def load_sqlite_db(db_path):
    """Load all tables from SQLite database into a dictionary of dataframes"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    dataframes = {}
    for table in tables:
        table_name = table[0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        dataframes[table_name] = df
        print(f"Loaded table {table_name} from {db_path}")
    
    conn.close()
    return dataframes

# Function to process and structure data
def process_data(df, data_type='text'):
    """Process and structure data with required labels"""
    # Create structured DataFrame with all required columns
    structured_df = pd.DataFrame()
    
    # Generate IDs if not present
    if 'id' not in df.columns:
        structured_df['id'] = [str(i) for i in range(len(df))]
    else:
        structured_df['id'] = df['id']
    
    # Map existing columns or create defaults
    columns_map = {
        'category': lambda x: x.get('category', 'general'),
        'genre': lambda x: x.get('genre', 'unknown'),
        'label': lambda x: x.get('label', 'unclassified'),
        'filename': lambda x: x.get('filename', 'unknown'),
        'text': lambda x: x.get('text', x.get('content', '')),
        'url': lambda x: x.get('url', ''),
        'content': lambda x: x.get('content', x.get('text', '')),
        'vocabulary': lambda x: [],
        'keyword': lambda x: extract_keywords(x.get('content', x.get('text', ''))),
        'definition': lambda x: '',
        'time/date': lambda x: datetime.now().isoformat(),
        'size': lambda x: x.get('size', 0),
        'enc': lambda x: encrypt_text(str(x.get('content', x.get('text', '')))),
        'gpt2token': lambda x: len(enc.encode(str(x.get('content', x.get('text', '')))))
    }
    
    # Fill in all required columns
    for col, func in columns_map.items():
        structured_df[col] = df.apply(lambda row: func(row), axis=1)
    
    # Classify content using DarkBERT
    try:
        classify_with_darkbert(structured_df)
        print("Classification with DarkBERT completed")
    except Exception as e:
        print(f"Error in DarkBERT classification: {e}")
    
    return structured_df

# Extract keywords using simple frequency analysis
def extract_keywords(text, top_n=5):
    """Extract keywords from text using simple frequency analysis"""
    if not isinstance(text, str) or not text:
        return []
    
    # Simple tokenization and cleaning
    words = text.lower().split()
    words = [word.strip('.,!?;:()[]{}""\'') for word in words]
    
    # Filter stop words (a very basic list)
    stop_words = set(['the', 'a', 'an', 'and', 'is', 'in', 'of', 'to', 'for', 'with', 'on', 'at', 'by', 'from'])
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequencies
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top N keywords
    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [keyword for keyword, _ in keywords]

# Classify text with DarkBERT
def classify_with_darkbert(df):
    """Classify text using DarkBERT model"""
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("guidobenb/DarkBERT-finetuned-ner")
        model = AutoModelForTokenClassification.from_pretrained("guidobenb/DarkBERT-finetuned-ner")
        
        # Create NER pipeline
        ner = pipeline("ner", model=model, tokenizer=tokenizer)
        
        def classify_text(text):
            if not isinstance(text, str) or not text:
                return {'category': 'unknown', 'label': 'unclassified'}
            
            # Truncate text to avoid token length issues
            text = text[:512]
            
            try:
                # Get NER results
                ner_results = ner(text)
                
                # Extract categories
                categories = set()
                for entity in ner_results:
                    if entity['entity'].startswith('B-') or entity['entity'].startswith('I-'):
                        category = entity['entity'][2:]  # Remove B- or I- prefix
                        categories.add(category)
                
                # Determine primary category and label
                if not categories:
                    return {'category': 'general', 'label': 'unclassified'}
                
                primary_category = list(categories)[0]
                
                # Special handling for potentially sensitive content
                sensitive_categories = {'THREAT', 'ATTACK', 'MALWARE', 'VULNERABILITY'}
                for cat in sensitive_categories:
                    if cat in categories:
                        # Ensure this content is encrypted
                        return {'category': 'sensitive', 'label': cat.lower()}
                
                return {'category': primary_category.lower(), 'label': primary_category.lower()}
            
            except Exception as e:
                print(f"Error in text classification: {e}")
                return {'category': 'error', 'label': 'classification_error'}
        
        # Apply classification to each row
        classification_results = df['text'].apply(classify_text)
        
        # Update category and label columns
        df['category'] = classification_results.apply(lambda x: x['category'])
        df['label'] = classification_results.apply(lambda x: x['label'])
        
    except Exception as e:
        print(f"DarkBERT classification error: {e}")

# Save dataframe to SQLite
def save_to_sqlite(df, db_path, table_name='data'):
    """Save dataframe to SQLite database"""
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved to SQLite: {db_path}, table: {table_name}")

# Export SQLite to CSV or JSON
def export_sqlite_to_file(db_path, output_format='csv'):
    """Export SQLite database to CSV or JSON"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    exported_files = []
    for table in tables:
        table_name = table[0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Generate output file name
        output_file = f"{os.path.splitext(db_path)[0]}_{table_name}.{output_format}"
        
        # Export based on format
        if output_format == 'csv':
            df.to_csv(output_file, index=False)
        elif output_format == 'json':
            df.to_json(output_file, orient='records', indent=2)
        
        exported_files.append(output_file)
        print(f"Exported {table_name} to {output_file}")
    
    conn.close()
    return exported_files

# Tokenize dataset and save tokens
def tokenize_dataset(df, output_file="trained/model/tokens.bin"):
    """Tokenize text data and save tokens to file"""
    all_text = " ".join(df['text'].astype(str).tolist())
    tokens = enc.encode(all_text)
    
    # Write tokens to file in format compatible with gpt2-llm.c
    from gpt2_llm_c.dev.data.data_common import write_datafile
    write_datafile(output_file, tokens)
    print(f"Tokenized {len(tokens)} tokens to {output_file}")
    return tokens

# Save the model
def save_model(model_name="Deceptiv2S7"):
    """Save model configuration and necessary files"""
    model_dir = f"trained/model/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model configuration
    config = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "tokenizer": "gpt2",
        "vocab_size": 50257,  # GPT-2 vocab size
        "has_encrypted_data": True,
        "encryption_key_path": "crypto.key",
        "data_files": {
            "tokens": "tokens.bin",
            "structured_data": "structured_data.csv",
            "keywords": "keywords.csv"
        }
    }
    
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved at {model_dir}")
    return model_dir

# Check database for errors
def check_database(db_path):
    """Check SQLite database for issues"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check integrity
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        table_stats = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table_stats[table_name] = count
        
        conn.close()
        
        return {
            "integrity": integrity[0],
            "tables": table_stats
        }
    except Exception as e:
        return {"error": str(e)}

# Main execution function
def main():
    print("Starting data processing pipeline...")
    
    # Step 1: Load and process text files
    print("\n=== Processing Text Files ===")
    text_df = load_text_files()
    if not text_df.empty:
        processed_text_df = process_data(text_df, 'text')
        save_to_sqlite(processed_text_df, "keywords.db", "keywords")
        print("Text files processed and saved to keywords.db")
    
    # Step 2: Load and process CSV files
    print("\n=== Processing CSV Files ===")
    csv_df = load_csv_files()
    if not csv_df.empty:
        processed_csv_df = process_data(csv_df, 'csv')
        save_to_sqlite(processed_csv_df, "labeled_dataset_ext.db", "data")
        print("CSV files processed and saved to labeled_dataset_ext.db")
    
    # Step 3: Process dataset.db if it exists
    print("\n=== Processing dataset.db ===")
    if os.path.exists("data/dataset.db"):
        dataset_dfs = load_sqlite_db("data/dataset.db")
        if dataset_dfs:
            # Combine all tables
            combined_df = pd.concat([df for df in dataset_dfs.values()], ignore_index=True)
            processed_db_df = process_data(combined_df, 'database')
            save_to_sqlite(processed_db_df, "labeled_dataset.db", "data")
            print("dataset.db processed and saved to labeled_dataset.db")
    
    # Step 4: Verify databases
    print("\n=== Verifying Databases ===")
    db_paths = ["keywords.db", "labeled_dataset.db", "labeled_dataset_ext.db"]
    for db_path in db_paths:
        if os.path.exists(db_path):
            check_result = check_database(db_path)
            print(f"Check result for {db_path}: {check_result}")
    
    # Step 5: Merge labeled databases
    print("\n=== Merging Labeled Databases ===")
    merged_df = pd.DataFrame()
    
    # Merge labeled_dataset.db and labeled_dataset_ext.db
    if os.path.exists("labeled_dataset.db") and os.path.exists("labeled_dataset_ext.db"):
        db1 = load_sqlite_db("labeled_dataset.db")
        db2 = load_sqlite_db("labeled_dataset_ext.db")
        
        dfs_to_merge = []
        for df in db1.values():
            dfs_to_merge.append(df)
        for df in db2.values():
            dfs_to_merge.append(df)
        
        if dfs_to_merge:
            merged_df = pd.concat(dfs_to_merge, ignore_index=True)
            
            # Export to CSV and JSON
            merged_df.to_csv("structured_data.csv", index=False)
            merged_df.to_json("structured_data.json", orient='records', indent=2)
            print("Merged databases exported to structured_data.csv and structured_data.json")
    
    # Step 6: Export keywords.db to CSV/JSON
    print("\n=== Exporting keywords.db ===")
    if os.path.exists("keywords.db"):
        export_sqlite_to_file("keywords.db", 'csv')
        export_sqlite_to_file("keywords.db", 'json')
    
    # Step 7: Tokenize dataset
    print("\n=== Tokenizing Dataset ===")
    if not merged_df.empty:
        tokenize_dataset(merged_df)
    
    # Step 8: Save model
    print("\n=== Saving Model ===")
    save_model()
    
    print("\nData processing pipeline completed successfully!")

if __name__ == "__main__":
    main()
