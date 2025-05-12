
#!/bin/bash

# Helper script for llm.c operations
cd gpt2-llm.c

function show_help {
    echo "Usage: ./run_llmc.sh COMMAND"
    echo ""
    echo "Commands:"
    echo "  setup             - Install dependencies and prepare environment"
    echo "  data              - Download and preprocess the TinyShakespeare dataset"
    echo "  train-cpu         - Train the model on CPU"
    echo "  test-cpu          - Run unit tests on CPU"
    echo "  train-gpu         - Train the model on GPU (if available)"
    echo "  test-gpu          - Run unit tests on GPU (if available)"
    echo "  help              - Show this help message"
}

case "$1" in
    setup)
        pip install -r requirements.txt
        ;;
    data)
        python dev/data/tinyshakespeare.py
        python train_gpt2.py
        ;;
    train-cpu)
        make train_gpt2
        OMP_NUM_THREADS=4 ./train_gpt2
        ;;
    test-cpu)
        make test_gpt2
        ./test_gpt2
        ;;
    train-gpu)
        make train_gpt2fp32cu
        ./train_gpt2fp32cu
        ;;
    test-gpu)
        make test_gpt2fp32cu
        ./test_gpt2fp32cu
        ;;
    *)
        show_help
        ;;
esac
