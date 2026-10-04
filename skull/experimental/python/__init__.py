# ============================================================
#  SKULL PYTHON PACKAGE v1.0.0
#  Python API for Skull
# ============================================================

"""
Skull - A programming language for training language models.

This package provides Python bindings for the Skull language, allowing you to:
- Create and train neural network models
- Generate text using trained models
- Use various optimizers and schedulers
- Run on CPU (with AVX2/SSE optimization) or GPU (OpenCL/CUDA)

Example:
    >>> import skull
    >>> model = skull.Model(architecture="transformer", vocab_size=256, dim=512)
    >>> model.train("data.txt", epochs=10)
    >>> text = model.generate("Hello", max_tokens=100)
    >>> print(text)
"""

from ._skull import (
    Tensor,
    Model,
    Tokenizer,
    init,
    train,
    generate,
    info,
    version,
)

__version__ = version()
__all__ = [
    "Tensor",
    "Model",
    "Tokenizer",
    "init",
    "train",
    "generate",
    "info",
    "version",
    "__version__",
]

# ============================================================
#  UTILITY FUNCTIONS
# ============================================================

def get_build_info():
    """Get Skull build information"""
    return info()

def get_version():
    """Get Skull version"""
    return version()

# ============================================================
#  MODEL FACTORY FUNCTIONS
# ============================================================

def create_transformer(vocab_size=256, dim=512, num_layers=6, num_heads=8, 
                     d_ff=2048, dropout_p=0.1, name="Transformer"):
    """
    Create a Transformer model.
    
    Args:
        vocab_size: Size of vocabulary
        dim: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout_p: Dropout probability
        name: Model name
    
    Returns:
        Model: A Transformer model instance
    """
    return Model(name, "transformer", vocab_size, dim, num_layers, num_heads, 
                d_ff, 0, dropout_p)

def create_feedforward(vocab_size=256, dim=512, num_layers=2, 
                     hidden_dim=0, name="Feedforward"):
    """
    Create a Feedforward model.
    
    Args:
        vocab_size: Size of vocabulary
        dim: Model dimension
        num_layers: Number of hidden layers
        hidden_dim: Hidden layer dimension (0 = same as dim)
        name: Model name
    
    Returns:
        Model: A Feedforward model instance
    """
    return Model(name, "feedforward", vocab_size, dim, num_layers, 1, 
                0, hidden_dim)

def create_lstm(vocab_size=256, dim=512, num_layers=2, 
               hidden_size=0, name="LSTM"):
    """
    Create an LSTM model.
    
    Args:
        vocab_size: Size of vocabulary
        dim: Input dimension
        num_layers: Number of LSTM layers
        hidden_size: Hidden state size (0 = same as dim)
        name: Model name
    
    Returns:
        Model: An LSTM model instance
    """
    return Model(name, "lstm", vocab_size, dim, num_layers, 1, 
                0, hidden_size)

# ============================================================
#  TOKENIZER FACTORY FUNCTIONS
# ============================================================

def create_bpe_tokenizer(vocab_size=256, bpe_path=None):
    """
    Create a BPE (Byte Pair Encoding) tokenizer.
    
    Args:
        vocab_size: Size of vocabulary
        bpe_path: Path to BPE vocabulary file (optional)
    
    Returns:
        Tokenizer: A BPE tokenizer instance
    """
    if bpe_path:
        return Tokenizer(vocab_size, bpe_path)
    return Tokenizer(vocab_size)

# ============================================================
#  TRAINING UTILITIES
# ============================================================

def train_model(model, data_path, epochs=10, batch_size=32, learning_rate=0.001,
                optimizer="adam", scheduler="cosine"):
    """
    Train a model with the given parameters.
    
    Args:
        model: The model to train
        data_path: Path to training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ("sgd", "adam", "adamw", "rmsprop", "adagrad", "lion")
        scheduler: Learning rate scheduler ("steplr", "cosine", "exponential", "linear")
    """
    # Set optimizer
    model.set_optimizer(optimizer, learning_rate)
    
    # Set scheduler
    model.set_scheduler(scheduler, learning_rate)
    
    # Train
    model.train(data_path, epochs)

def evaluate_model(model, data_path, batch_size=32):
    """
    Evaluate a model on the given data.
    
    Args:
        model: The model to evaluate
        data_path: Path to evaluation data
        batch_size: Batch size for evaluation
    
    Returns:
        float: Evaluation loss
    """
    return model.evaluate(data_path, batch_size)

def generate_text(model, prompt="", max_tokens=100, temperature=0.8):
    """
    Generate text using a trained model.
    
    Args:
        model: The model to use for generation
        prompt: Starting text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more creative)
    
    Returns:
        str: Generated text
    """
    return model.generate(prompt, max_tokens, temperature)

# ============================================================
#  CONTEXT MANAGER FOR SKULL INITIALIZATION
# ============================================================

class SkullContext:
    """Context manager for Skull initialization"""
    
    def __init__(self, gpu_backend="auto", num_threads=0, precision="fp32"):
        """
        Initialize Skull context.
        
        Args:
            gpu_backend: GPU backend to use ("cpu", "opencl", "cuda", "metal", "auto")
            num_threads: Number of CPU threads to use (0 = auto-detect)
            precision: Precision mode ("fp32", "fp16", "bf16", "fp64")
        """
        self.gpu_backend = gpu_backend
        self.num_threads = num_threads
        self.precision = precision
    
    def __enter__(self):
        """Enter the context"""
        init()
        # Additional initialization would go here
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context"""
        pass  # Cleanup if needed

# ============================================================
#  DEPRECATED COMPATIBILITY
# ============================================================

# For backward compatibility
skull_info = get_build_info
skull_version = get_version
