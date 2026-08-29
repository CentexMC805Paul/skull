#!/usr/bin/env python3
# ============================================================
#  SKULL COMMAND LINE INTERFACE v1.0.0
# ============================================================

"""
Skull CLI - Command line interface for Skull

Usage:
    skull train config.yaml          # Train a model
    skull generate model.skull       # Generate text with a model
    skull info                      # Show Skull information
    skull convert data.txt           # Convert data to Skull format
    skull tokenize text.txt          # Tokenize text
"""

import argparse
import sys
import os
import yaml
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from . import skull
    from . import __version__
except ImportError:
    import skull
    from skull import __version__

# ============================================================
#  MAIN FUNCTION
# ============================================================

def main():
    """Main entry point for Skull CLI"""
    
    parser = argparse.ArgumentParser(
        prog="skull",
        description="Skull - A programming language for training language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  skull train config.yaml           Train a model
  skull generate model.skull        Generate text with a model
  skull info                       Show Skull information
  skull convert data.txt            Convert data to Skull format
  skull tokenize text.txt           Tokenize text
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", type=str, help="Path to training configuration file")
    train_parser.add_argument("--gpu", action="store_true", help="Use GPU")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    train_parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text")
    generate_parser.add_argument("model", type=str, help="Path to model file or model name")
    generate_parser.add_argument("--prompt", "-p", type=str, default="", help="Starting prompt")
    generate_parser.add_argument("--tokens", "-t", type=int, default=100, help="Number of tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    generate_parser.add_argument("--output", "-o", type=str, default=None, help="Output file")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show Skull information")
    info_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert data to Skull format")
    convert_parser.add_argument("input", type=str, help="Input file or directory")
    convert_parser.add_argument("--output", "-o", type=str, default=None, help="Output file or directory")
    convert_parser.add_argument("--format", "-f", type=str, default="jsonl", 
                                choices=["txt", "json", "jsonl", "csv", "md"],
                                help="Output format")
    convert_parser.add_argument("--bpe", action="store_true", help="Use BPE tokenization")
    convert_parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size for BPE")
    
    # Tokenize command
    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize text")
    tokenize_parser.add_argument("input", type=str, help="Input text file")
    tokenize_parser.add_argument("--output", "-o", type=str, default=None, help="Output file")
    tokenize_parser.add_argument("--bpe", action="store_true", help="Use BPE tokenization")
    tokenize_parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size for BPE")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    evaluate_parser.add_argument("model", type=str, help="Path to model file")
    evaluate_parser.add_argument("data", type=str, help="Path to evaluation data")
    evaluate_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize Skull
    skull.init()
    
    # Execute command
    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "tokenize":
        cmd_tokenize(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()

# ============================================================
#  COMMAND IMPLEMENTATIONS
# ============================================================

def cmd_train(args):
    """Train a model"""
    print(f"Skull v{__version__}")
    print(f"Training with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override from command line
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    
    # Set GPU if requested
    if args.gpu:
        config["gpu"] = True
    
    # Create model
    print("Creating model...")
    model = create_model_from_config(config)
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    try:
        model.train(
            config.get("data", ""),
            config.get("epochs", 10)
        )
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    
    # Save model
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        model_path = os.path.join(args.output, "model.skull")
        model.save(model_path)
        print(f"Model saved to: {model_path}")
    elif config.get("save_path"):
        model.save(config["save_path"])
        print(f"Model saved to: {config['save_path']}")

def cmd_generate(args):
    """Generate text"""
    print(f"Skull v{__version__}")
    print(f"Generating text with model: {args.model}")
    
    # Load or create model
    if os.path.exists(args.model):
        print("Loading model...")
        model = skull.Model()
        model.load(args.model)
    else:
        print(f"Model file not found: {args.model}")
        print("Creating a new model...")
        model = skull.create_transformer()
    
    # Generate text
    print(f"Prompt: {args.prompt[:50]}..." if len(args.prompt) > 50 else f"Prompt: {args.prompt}")
    print(f"Generating {args.tokens} tokens with temperature {args.temperature}...")
    
    start_time = time.time()
    text = model.generate(args.prompt, args.tokens, args.temperature)
    elapsed = time.time() - start_time
    
    print(f"\nGenerated text:")
    print("-" * 80)
    print(text)
    print("-" * 80)
    print(f"Generated in {elapsed:.2f} seconds")
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to: {args.output}")

def cmd_info(args):
    """Show Skull information"""
    print(f"Skull v{__version__}")
    print("=" * 80)
    
    info = skull.info()
    print(info)
    
    if args.verbose:
        print("\nBuild Information:")
        print("-" * 40)
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"Executor: {sys.executable}")
        
        # Check for GPU support
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"CUDA device count: {torch.cuda.device_count()}")
        except ImportError:
            print("PyTorch not installed")
        
        try:
            import pyopencl
            print(f"PyOpenCL version: {pyopencl.__version__}")
            platforms = pyopencl.get_platforms()
            print(f"OpenCL platforms: {len(platforms)}")
            for i, platform in enumerate(platforms):
                print(f"  Platform {i}: {platform.name}")
                devices = platform.get_devices()
                print(f"    Devices: {len(devices)}")
                for j, device in enumerate(devices):
                    print(f"      Device {j}: {device.name}")
        except ImportError:
            print("PyOpenCL not installed")

def cmd_convert(args):
    """Convert data to Skull format"""
    print(f"Skull v{__version__}")
    print(f"Converting: {args.input}")
    
    input_path = args.input
    output_path = args.output if args.output else f"{input_path}.converted"
    
    # Check if input is a directory
    if os.path.isdir(input_path):
        # Convert all files in directory
        os.makedirs(output_path, exist_ok=True)
        
        for filename in os.listdir(input_path):
            input_file = os.path.join(input_path, filename)
            if os.path.isfile(input_file):
                output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.{args.format}")
                convert_file(input_file, output_file, args.format, args.bpe, args.vocab_size)
    else:
        # Convert single file
        convert_file(input_path, output_path, args.format, args.bpe, args.vocab_size)
    
    print(f"Conversion complete. Output: {output_path}")

def cmd_tokenize(args):
    """Tokenize text"""
    print(f"Skull v{__version__}")
    print(f"Tokenizing: {args.input}")
    
    # Create tokenizer
    tokenizer = skull.create_bpe_tokenizer(args.vocab_size)
    
    # Read input
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Tokenize
    tokens = tokenizer.encode(text)
    
    print(f"Text length: {len(text)} characters")
    print(f"Token count: {len(tokens)} tokens")
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for token in tokens:
                f.write(f"{token}\n")
        print(f"Tokens saved to: {args.output}")
    else:
        print("Tokens:")
        print(tokens[:100])  # Print first 100 tokens
        if len(tokens) > 100:
            print(f"... ({len(tokens) - 100} more tokens)")

def cmd_evaluate(args):
    """Evaluate a model"""
    print(f"Skull v{__version__}")
    print(f"Evaluating model: {args.model}")
    print(f"Data: {args.data}")
    
    # Load model
    model = skull.Model()
    model.load(args.model)
    
    # Evaluate
    print("Evaluating...")
    start_time = time.time()
    
    try:
        loss = model.evaluate(args.data, args.batch_size)
        elapsed = time.time() - start_time
        print(f"Evaluation loss: {loss:.6f}")
        print(f"Evaluated in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def load_config(config_path):
    """Load configuration from file"""
    config = {}
    
    # Try YAML first
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except ImportError:
            print("PyYAML not installed. Trying JSON...")
            with open(config_path, "r") as f:
                config = json.load(f)
    
    # Try JSON
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = json.load(f)
    
    # Try Python
    elif config_path.endswith(".py"):
        # Execute Python file and get config dict
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = getattr(config_module, "config", {})
    
    else:
        # Try as YAML, then JSON
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except:
            with open(config_path, "r") as f:
                config = json.load(f)
    
    return config

def create_model_from_config(config):
    """Create a model from configuration"""
    arch = config.get("architecture", "transformer").lower()
    
    if arch in ["transformer", "transformer_model"]:
        return skull.create_transformer(
            vocab_size=config.get("vocab_size", 256),
            dim=config.get("dim", 512),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 8),
            d_ff=config.get("d_ff", 2048),
            dropout_p=config.get("dropout_p", 0.1),
            name=config.get("name", "Transformer")
        )
    elif arch in ["feedforward", "ff", "mlp"]:
        return skull.create_feedforward(
            vocab_size=config.get("vocab_size", 256),
            dim=config.get("dim", 512),
            num_layers=config.get("num_layers", 2),
            hidden_dim=config.get("hidden_dim", 0),
            name=config.get("name", "Feedforward")
        )
    elif arch in ["lstm", "rnn"]:
        return skull.create_lstm(
            vocab_size=config.get("vocab_size", 256),
            dim=config.get("dim", 512),
            num_layers=config.get("num_layers", 2),
            hidden_size=config.get("hidden_size", 0),
            name=config.get("name", "LSTM")
        )
    else:
        print(f"Unknown architecture: {arch}. Using transformer.")
        return skull.create_transformer(
            vocab_size=config.get("vocab_size", 256),
            dim=config.get("dim", 512),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 8),
            name=config.get("name", "Transformer")
        )

def convert_file(input_path, output_path, format, bpe, vocab_size):
    """Convert a single file to the specified format"""
    print(f"  Converting {input_path} to {output_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    elif format == "json":
        # Convert to JSON with one document
        import json
        data = {"text": content}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif format == "jsonl":
        # Convert to JSON Lines (one line per document)
        import json
        lines = content.split("\n\n")  # Split by empty lines
        with open(output_path, "w", encoding="utf-8") as f:
            for line in lines:
                if line.strip():
                    data = {"text": line.strip()}
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    elif format == "csv":
        # Convert to CSV
        import csv
        lines = content.split("\n")
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            for line in lines:
                if line.strip():
                    writer.writerow([line.strip()])
    
    elif format == "md":
        # Just copy as markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

# ============================================================
#  MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
