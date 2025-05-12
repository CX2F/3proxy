
#!/bin/bash

# Setup script for the data processing and GPT-2 training environment

# Make the script executable
chmod +x *.py *.sh

# Create necessary directories
mkdir -p data
mkdir -p trained/model
mkdir -p generated_images
mkdir -p logs

# Check if gpt2-llm.c repo exists, clone if not
if [ ! -d "gpt2-llm.c" ]; then
    echo "Cloning gpt2-llm.c repository..."
    git clone https://github.com/karpathy/llm.c.git gpt2-llm.c
fi

# Install requirements
echo "Installing Python requirements..."
pip install -r requirements.txt

# Install tiktoken which is needed for tokenization
pip install tiktoken

# If gpt2-llm.c has requirements, install those too
if [ -f "gpt2-llm.c/requirements.txt" ]; then
    echo "Installing gpt2-llm.c requirements..."
    pip install -r gpt2-llm.c/requirements.txt
fi

# Create a sample .env file for OpenAI API key
if [ ! -f ".env" ]; then
    echo "Creating sample .env file..."
    echo "# Add your OpenAI API key here" > .env
    echo "OPENAI_API_KEY=\"\"" >> .env
    echo ".env file created. Please add your OpenAI API key."
fi

echo "Setup completed successfully!"
echo "You can now run:"
echo "  python main.py --sample-data  # To create sample data"
echo "  python main.py --process      # To process and structure data"
echo "  python main.py --train        # To train the model"
echo "  python main.py --classify     # To classify data using GPT-4o (requires API key)"
