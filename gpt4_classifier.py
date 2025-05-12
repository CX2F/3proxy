
# /gpt4_classifier.py
import os
import json
import openai
import pandas as pd
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment or use a default one
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize the OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def classify_text_with_gpt4o(text, max_retries=3, backoff_factor=2):
    """
    Classifies a text sample using GPT-4o.
    Returns a JSON with classification information.
    """
    if not OPENAI_API_KEY:
        return {
            "error": "OpenAI API key not found. Set the OPENAI_API_KEY environment variable."
        }
    
    # Create a system prompt for classification
    system_prompt = """
    You are an AI text classifier. Analyze the provided text and classify it with the following information:
    - category: A broad category the text belongs to.
    - genre: The genre of the content.
    - label: A specific label that best describes the content.
    - vocabulary: List of 3-5 domain-specific vocabulary terms found in the text.
    - keywords: List of 3-7 important keywords that represent the main themes.
    - definition: A concise definition of what this text is about.
    
    Provide your response as a JSON object with these fields only. Do not include explanations outside the JSON.
    """
    
    # Prepare a concise version of the text to avoid token limits
    # Keep first 1000 chars to stay well under token limits
    truncated_text = text[:1000] + ("..." if len(text) > 1000 else "")
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": truncated_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=500
            )
            
            # Parse the JSON response
            try:
                classification = json.loads(response.choices[0].message.content)
                return classification
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return {"error": "Failed to parse response as JSON"}
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait with exponential backoff
                wait_time = backoff_factor ** attempt
                print(f"API error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached. API error: {e}")
                return {"error": str(e)}
    
    return {"error": "Failed to classify text after multiple attempts"}

def classify_dataframe(df, text_column='text', batch_size=10):
    """Classify text in a dataframe using GPT-4o"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Invalid or empty dataframe")
        return df
    
    if text_column not in df.columns:
        print(f"Error: Column {text_column} not found in dataframe")
        return df
    
    results = []
    
    # Process in batches to avoid rate limits
    for i in tqdm(range(0, len(df), batch_size), desc="Classifying with GPT-4o"):
        batch = df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            text = row[text_column]
            
            # Skip empty texts
            if not isinstance(text, str) or not text.strip():
                results.append({})
                continue
                
            # Get classification
            classification = classify_text_with_gpt4o(text)
            results.append(classification)
            
            # Small pause to avoid rate limits
            time.sleep(0.5)
    
    # Now update the dataframe with the classification results
    for i, classification in enumerate(results):
        if not classification or "error" in classification:
            continue
            
        # Update each field from the classification
        for key, value in classification.items():
            if key in df.columns:
                df.iloc[i, df.columns.get_loc(key)] = value
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify text data with GPT-4o")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, help="Output CSV file (default: input_classified.csv)")
    parser.add_argument("--column", type=str, default="text", help="Column containing text to classify")
    parser.add_argument("--batch", type=int, default=10, help="Batch size for API calls")
    
    args = parser.parse_args()
    
    # Generate default output name if not provided
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_classified{ext}"
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records from {args.input}")
    
    # Classify
    classified_df = classify_dataframe(df, args.column, args.batch)
    
    # Save results
    classified_df.to_csv(args.output, index=False)
    print(f"Saved classified data to {args.output}")
