
# /dataset_processor.py
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
from crypto_utils import CryptoUtils
from mediapipe_integration import MediaPipeDatasetCreator, DiffusionModelGenerator, LlamaInference

class DatasetProcessor:
    """Process and structure datasets using various AI tools"""
    
    def __init__(self, data_dir="data", output_dir="output"):
        """Initialize the dataset processor"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.mediapipe = MediaPipeDatasetCreator(base_dir=data_dir)
        self.image_generator = DiffusionModelGenerator()
        self.llm = LlamaInference()
        self.crypto = CryptoUtils()
        
        # Databases
        self.db_keywords = os.path.join(output_dir, "keywords.db")
        self.db_labeled_dataset = os.path.join(output_dir, "labeled_dataset.db")
        self.db_labeled_dataset_ext = os.path.join(output_dir, "labeled_dataset_ext.db")
        self.db_structured_data = os.path.join(output_dir, "structured_data.db")
    
    def initialize_tools(self):
        """Initialize all AI tools"""
        print("Initializing MediaPipe vision tasks...")
        self.mediapipe.initialize_vision_tasks()
        
        print("Initializing MediaPipe text tasks...")
        self.mediapipe.initialize_text_tasks()
        
        print("Initializing image generation model...")
        self.image_generator.load_model("black-forest-labs/FLUX.1-dev")
        self.image_generator.load_model("Heartsync/NSFW-Uncensored")
        
        print("Initializing LLM for text processing...")
        self.llm.load_model("afrideva/phi-2-uncensored-GGUF", "phi-2-uncensored.fp16.gguf")
        
        print("All tools initialized successfully")
    
    def process_text_files(self):
        """Process text files in the data directory and create keywords database"""
        print("Processing text files...")
        text_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        if not text_files:
            print("No text files found in data directory")
            return False
        
        # Create connection to keywords database
        conn = sqlite3.connect(self.db_keywords)
        cursor = conn.cursor()
        
        # Create keywords table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            genre TEXT,
            label TEXT,
            filename TEXT,
            text TEXT,
            url TEXT,
            content TEXT,
            vocabulary TEXT,
            keyword TEXT,
            definition TEXT,
            timestamp TEXT,
            size INTEGER,
            enc TEXT,
            gpt2token INTEGER
        )
        ''')
        
        # Process each text file
        all_data = []
        
        for filename in tqdm(text_files, desc="Processing text files"):
            file_path = os.path.join(self.data_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process text with MediaPipe
            text_results = self.mediapipe.process_text(content)
            
            # Get language detection
            language = "unknown"
            if 'language' in text_results and text_results['language'].predictions:
                language = text_results['language'].predictions[0].language_code
            
            # Get classification
            category = "general"
            if 'classification' in text_results and text_results['classification'].classifications:
                if text_results['classification'].classifications[0].categories:
                    category = text_results['classification'].classifications[0].categories[0].category_name
            
            # Extract keywords using LLM
            prompt = f"Extract 5-10 keywords from this text. Return only the keywords as a comma-separated list:\n\n{content[:1000]}"
            
            keywords = []
            try:
                llm_response = self.llm.generate_completion(prompt, max_tokens=100)
                if llm_response and 'text' in llm_response:
                    keywords = [k.strip() for k in llm_response['text'].split(',')]
            except Exception as e:
                print(f"Error extracting keywords: {str(e)}")
            
            # Encrypt sensitive content
            encrypted_content = self.crypto.encrypt_text(content)
            
            # Create dataset entry
            entry = {
                'category': category,
                'genre': language,
                'label': 'text',
                'filename': filename,
                'text': content[:1000] if len(content) > 1000 else content,  # Truncate long texts
                'url': '',
                'content': content[:1000] if len(content) > 1000 else content,
                'vocabulary': json.dumps([]),
                'keyword': json.dumps(keywords),
                'definition': '',
                'timestamp': pd.Timestamp.now().isoformat(),
                'size': os.path.getsize(file_path),
                'enc': encrypted_content,
                'gpt2token': len(content.split())  # Simple approximation
            }
            
            all_data.append(entry)
            
            # Insert into database
            cursor.execute('''
            INSERT INTO keywords 
            (category, genre, label, filename, text, url, content, vocabulary, keyword, definition, timestamp, size, enc, gpt2token)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry['category'], entry['genre'], entry['label'], entry['filename'],
                entry['text'], entry['url'], entry['content'], entry['vocabulary'],
                entry['keyword'], entry['definition'], entry['timestamp'], entry['size'],
                entry['enc'], entry['gpt2token']
            ))
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Processed {len(all_data)} text files and saved to {self.db_keywords}")
        return True
    
    def process_csv_files(self):
        """Process CSV files in the data directory and create labeled_dataset_ext database"""
        print("Processing CSV files...")
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found in data directory")
            return False
        
        # Create connection to database
        conn = sqlite3.connect(self.db_labeled_dataset_ext)
        cursor = conn.cursor()
        
        # Create data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            genre TEXT,
            label TEXT,
            filename TEXT,
            text TEXT,
            url TEXT,
            content TEXT,
            vocabulary TEXT,
            keyword TEXT,
            definition TEXT,
            timestamp TEXT,
            size INTEGER,
            enc TEXT,
            gpt2token INTEGER
        )
        ''')
        
        # Process each CSV file
        all_data = []
        
        for filename in tqdm(csv_files, desc="Processing CSV files"):
            file_path = os.path.join(self.data_dir, filename)
            
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Process each row
            for _, row in df.iterrows():
                # Extract text content
                text = ""
                if 'text' in row:
                    text = row['text']
                elif 'content' in row:
                    text = row['content']
                
                if not isinstance(text, str) or not text.strip():
                    continue
                
                # Process text with MediaPipe if available
                category = row.get('category', 'general')
                label = row.get('label', 'unlabeled')
                
                # Extract keywords
                keywords = []
                if 'keyword' in row:
                    if isinstance(row['keyword'], str):
                        try:
                            keywords = json.loads(row['keyword'])
                        except:
                            keywords = [k.strip() for k in row['keyword'].split(',')]
                
                # If no keywords, try to extract them using LLM
                if not keywords and self.llm.model:
                    prompt = f"Extract 5-10 keywords from this text. Return only the keywords as a comma-separated list:\n\n{text[:1000]}"
                    
                    try:
                        llm_response = self.llm.generate_completion(prompt, max_tokens=100)
                        if llm_response and 'text' in llm_response:
                            keywords = [k.strip() for k in llm_response['text'].split(',')]
                    except Exception as e:
                        print(f"Error extracting keywords: {str(e)}")
                
                # Encrypt sensitive content
                encrypted_content = self.crypto.encrypt_text(text)
                
                # Create dataset entry
                entry = {
                    'category': category,
                    'genre': row.get('genre', ''),
                    'label': label,
                    'filename': filename,
                    'text': text[:1000] if len(text) > 1000 else text,  # Truncate long texts
                    'url': row.get('url', ''),
                    'content': text[:1000] if len(text) > 1000 else text,
                    'vocabulary': json.dumps(row.get('vocabulary', [])),
                    'keyword': json.dumps(keywords),
                    'definition': row.get('definition', ''),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'size': len(text),
                    'enc': encrypted_content,
                    'gpt2token': len(text.split())  # Simple approximation
                }
                
                all_data.append(entry)
                
                # Insert into database
                cursor.execute('''
                INSERT INTO data 
                (category, genre, label, filename, text, url, content, vocabulary, keyword, definition, timestamp, size, enc, gpt2token)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry['category'], entry['genre'], entry['label'], entry['filename'],
                    entry['text'], entry['url'], entry['content'], entry['vocabulary'],
                    entry['keyword'], entry['definition'], entry['timestamp'], entry['size'],
                    entry['enc'], entry['gpt2token']
                ))
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Processed {len(all_data)} rows from CSV files and saved to {self.db_labeled_dataset_ext}")
        return True
    
    def process_sqlite_database(self):
        """Process dataset.db and create labeled_dataset database"""
        print("Processing dataset.db...")
        db_path = os.path.join(self.data_dir, "dataset.db")
        
        if not os.path.exists(db_path):
            print("dataset.db not found in data directory")
            return False
        
        # Connect to source database
        source_conn = sqlite3.connect(db_path)
        source_cursor = source_conn.cursor()
        
        # Get list of tables
        source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = source_cursor.fetchall()
        
        if not tables:
            print("No tables found in dataset.db")
            source_conn.close()
            return False
        
        # Create connection to output database
        conn = sqlite3.connect(self.db_labeled_dataset)
        cursor = conn.cursor()
        
        # Create data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            genre TEXT,
            label TEXT,
            filename TEXT,
            text TEXT,
            url TEXT,
            content TEXT,
            vocabulary TEXT,
            keyword TEXT,
            definition TEXT,
            timestamp TEXT,
            size INTEGER,
            enc TEXT,
            gpt2token INTEGER
        )
        ''')
        
        # Process each table
        all_data = []
        
        for table in tables:
            table_name = table[0]
            print(f"Processing table: {table_name}")
            
            # Get all rows from the table
            source_cursor.execute(f"SELECT * FROM {table_name}")
            columns = [desc[0] for desc in source_cursor.description]
            rows = source_cursor.fetchall()
            
            for row in tqdm(rows, desc=f"Processing {table_name}"):
                # Convert to dictionary
                row_dict = dict(zip(columns, row))
                
                # Extract text content
                text = ""
                if 'text' in row_dict:
                    text = row_dict['text']
                elif 'content' in row_dict:
                    text = row_dict['content']
                
                if not isinstance(text, str) or not text.strip():
                    continue
                
                # Get category and label
                category = row_dict.get('category', 'general')
                label = row_dict.get('label', 'unlabeled')
                
                # Extract keywords
                keywords = []
                if 'keyword' in row_dict:
                    if isinstance(row_dict['keyword'], str):
                        try:
                            keywords = json.loads(row_dict['keyword'])
                        except:
                            keywords = [k.strip() for k in row_dict['keyword'].split(',')]
                
                # If no keywords, try to extract them using LLM
                if not keywords and self.llm.model:
                    prompt = f"Extract 5-10 keywords from this text. Return only the keywords as a comma-separated list:\n\n{text[:1000]}"
                    
                    try:
                        llm_response = self.llm.generate_completion(prompt, max_tokens=100)
                        if llm_response and 'text' in llm_response:
                            keywords = [k.strip() for k in llm_response['text'].split(',')]
                    except Exception as e:
                        print(f"Error extracting keywords: {str(e)}")
                
                # Encrypt sensitive content
                encrypted_content = self.crypto.encrypt_text(text)
                
                # Create dataset entry
                entry = {
                    'category': category,
                    'genre': row_dict.get('genre', ''),
                    'label': label,
                    'filename': f"{table_name}_{row_dict.get('id', len(all_data))}",
                    'text': text[:1000] if len(text) > 1000 else text,  # Truncate long texts
                    'url': row_dict.get('url', ''),
                    'content': text[:1000] if len(text) > 1000 else text,
                    'vocabulary': json.dumps(row_dict.get('vocabulary', [])),
                    'keyword': json.dumps(keywords),
                    'definition': row_dict.get('definition', ''),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'size': len(text),
                    'enc': encrypted_content,
                    'gpt2token': len(text.split())  # Simple approximation
                }
                
                all_data.append(entry)
                
                # Insert into database
                cursor.execute('''
                INSERT INTO data 
                (category, genre, label, filename, text, url, content, vocabulary, keyword, definition, timestamp, size, enc, gpt2token)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry['category'], entry['genre'], entry['label'], entry['filename'],
                    entry['text'], entry['url'], entry['content'], entry['vocabulary'],
                    entry['keyword'], entry['definition'], entry['timestamp'], entry['size'],
                    entry['enc'], entry['gpt2token']
                ))
        
        # Commit changes and close connections
        conn.commit()
        conn.close()
        source_conn.close()
        
        print(f"Processed {len(all_data)} rows from dataset.db and saved to {self.db_labeled_dataset}")
        return True
    
    def merge_databases(self):
        """Merge labeled_dataset.db and labeled_dataset_ext.db into structured_data.db"""
        print("Merging databases...")
        
        # Check if databases exist
        if not os.path.exists(self.db_labeled_dataset) or not os.path.exists(self.db_labeled_dataset_ext):
            print("One or both databases not found, cannot merge")
            return False
        
        # Connect to source databases
        conn1 = sqlite3.connect(self.db_labeled_dataset)
        cursor1 = conn1.cursor()
        
        conn2 = sqlite3.connect(self.db_labeled_dataset_ext)
        cursor2 = conn2.cursor()
        
        # Create connection to output database
        conn_out = sqlite3.connect(self.db_structured_data)
        cursor_out = conn_out.cursor()
        
        # Create structured_data table
        cursor_out.execute('''
        CREATE TABLE IF NOT EXISTS structured_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            genre TEXT,
            label TEXT,
            filename TEXT,
            text TEXT,
            url TEXT,
            content TEXT,
            vocabulary TEXT,
            keyword TEXT,
            definition TEXT,
            timestamp TEXT,
            size INTEGER,
            enc TEXT,
            gpt2token INTEGER,
            source TEXT
        )
        ''')
        
        # Get data from first database
        cursor1.execute("SELECT * FROM data")
        rows1 = cursor1.fetchall()
        
        # Get column names
        columns1 = [desc[0] for desc in cursor1.description]
        
        # Insert data from first database
        for row in tqdm(rows1, desc="Copying from labeled_dataset.db"):
            row_dict = dict(zip(columns1, row))
            
            # Add source field
            source = "labeled_dataset"
            
            cursor_out.execute('''
            INSERT INTO structured_data 
            (category, genre, label, filename, text, url, content, vocabulary, keyword, definition, timestamp, size, enc, gpt2token, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row_dict.get('category', ''), row_dict.get('genre', ''), row_dict.get('label', ''), 
                row_dict.get('filename', ''), row_dict.get('text', ''), row_dict.get('url', ''), 
                row_dict.get('content', ''), row_dict.get('vocabulary', ''), row_dict.get('keyword', ''), 
                row_dict.get('definition', ''), row_dict.get('timestamp', ''), row_dict.get('size', 0), 
                row_dict.get('enc', ''), row_dict.get('gpt2token', 0), source
            ))
        
        # Get data from second database
        cursor2.execute("SELECT * FROM data")
        rows2 = cursor2.fetchall()
        
        # Get column names
        columns2 = [desc[0] for desc in cursor2.description]
        
        # Insert data from second database
        for row in tqdm(rows2, desc="Copying from labeled_dataset_ext.db"):
            row_dict = dict(zip(columns2, row))
            
            # Add source field
            source = "labeled_dataset_ext"
            
            cursor_out.execute('''
            INSERT INTO structured_data 
            (category, genre, label, filename, text, url, content, vocabulary, keyword, definition, timestamp, size, enc, gpt2token, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row_dict.get('category', ''), row_dict.get('genre', ''), row_dict.get('label', ''), 
                row_dict.get('filename', ''), row_dict.get('text', ''), row_dict.get('url', ''), 
                row_dict.get('content', ''), row_dict.get('vocabulary', ''), row_dict.get('keyword', ''), 
                row_dict.get('definition', ''), row_dict.get('timestamp', ''), row_dict.get('size', 0), 
                row_dict.get('enc', ''), row_dict.get('gpt2token', 0), source
            ))
        
        # Commit changes and close connections
        conn_out.commit()
        
        # Export to CSV and JSON
        cursor_out.execute("SELECT * FROM structured_data")
        rows = cursor_out.fetchall()
        columns = [desc[0] for desc in cursor_out.description]
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "structured_data.csv")
        df.to_csv(csv_path, index=False)
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, "structured_data.json")
        df.to_json(json_path, orient='records', indent=2)
        
        # Close connections
        conn1.close()
        conn2.close()
        conn_out.close()
        
        print(f"Merged {len(rows1)} rows from labeled_dataset.db and {len(rows2)} rows from labeled_dataset_ext.db")
        print(f"Saved to {self.db_structured_data}, {csv_path}, and {json_path}")
        return True
    
    def export_keywords_db(self):
        """Export keywords.db to CSV and JSON formats"""
        print("Exporting keywords.db...")
        
        if not os.path.exists(self.db_keywords):
            print("keywords.db not found")
            return False
        
        # Connect to database
        conn = sqlite3.connect(self.db_keywords)
        
        # Create DataFrame
        df = pd.read_sql_query("SELECT * FROM keywords", conn)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "keywords.csv")
        df.to_csv(csv_path, index=False)
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, "keywords.json")
        df.to_json(json_path, orient='records', indent=2)
        
        # Close connection
        conn.close()
        
        print(f"Exported {len(df)} rows from keywords.db to {csv_path} and {json_path}")
        return True
    
    def generate_sample_images(self, num_images=5):
        """Generate sample images based on keywords from the database"""
        print("Generating sample images...")
        
        # Connect to structured data database
        if not os.path.exists(self.db_structured_data):
            print("structured_data.db not found")
            return False
        
        conn = sqlite3.connect(self.db_structured_data)
        cursor = conn.cursor()
        
        # Get keywords from the database
        cursor.execute("SELECT keyword FROM structured_data WHERE keyword IS NOT NULL AND keyword != ''")
        keyword_rows = cursor.fetchall()
        
        if not keyword_rows:
            print("No keywords found in database")
            conn.close()
            return False
        
        # Extract keywords
        all_keywords = []
        for row in keyword_rows:
            if row[0]:
                try:
                    keywords = json.loads(row[0])
                    if isinstance(keywords, list):
                        all_keywords.extend(keywords)
                except:
                    continue
        
        # Filter out empty keywords
        all_keywords = [k for k in all_keywords if k and isinstance(k, str)]
        
        if not all_keywords:
            print("No valid keywords found")
            conn.close()
            return False
        
        # Create prompts using extracted keywords
        prompts = []
        for i in range(min(num_images, len(all_keywords))):
            keyword = all_keywords[i % len(all_keywords)]
            prompt = f"A high-quality image of {keyword}, detailed, professional"
            prompts.append(prompt)
        
        # Generate images
        images_dir = os.path.join(self.output_dir, "generated_images")
        os.makedirs(images_dir, exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
            self.image_generator.generate_image(prompt, output_path=images_dir, seed=i+42)
        
        conn.close()
        print(f"Generated {len(prompts)} sample images in {images_dir}")
        return True
    
    def process_all(self):
        """Run the complete data processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Process text files and create keywords.db
        self.process_text_files()
        
        # Process CSV files and create labeled_dataset_ext.db
        self.process_csv_files()
        
        # Process dataset.db and create labeled_dataset.db
        self.process_sqlite_database()
        
        # Merge databases into structured_data.db
        self.merge_databases()
        
        # Export keywords.db to CSV and JSON
        self.export_keywords_db()
        
        # Generate sample images
        self.generate_sample_images()
        
        print("Data processing pipeline completed successfully!")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and structure datasets for AI training")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing input data files")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--init-tools", action="store_true", help="Initialize MediaPipe and AI tools")
    parser.add_argument("--generate-images", action="store_true", help="Generate sample images from keywords")
    parser.add_argument("--num-images", type=int, default=5, help="Number of sample images to generate")
    
    args = parser.parse_args()
    
    processor = DatasetProcessor(data_dir=args.data_dir, output_dir=args.output_dir)
    
    if args.init_tools:
        processor.initialize_tools()
    
    if args.generate_images:
        processor.generate_sample_images(num_images=args.num_images)
    else:
        processor.process_all()
