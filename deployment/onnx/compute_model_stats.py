'''
pip install prettytable pandas numpy torch transformers
'''


import torch
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
pd.set_option('display.max_colwidth', 1000)


# Load the model
model_name = "Helsinki-NLP/opus-mt-fr-en"
print(f"Loading model: {model_name}")
model = MarianMTModel.from_pretrained(model_name)

# Function to analyze layers in detail
def analyze_layers(model):
    layer_stats = []
    total_params = 0
    total_size_bytes = 0
    
    # Iterate through named parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Get parameter details
            param_count = param.numel()
            size_bytes = param_count * 4  # assuming float32 (4 bytes)
            size_mb = size_bytes / (1000 * 1000)
            
            # Update totals
            total_params += param_count
            total_size_bytes += size_bytes
            
            # Add to stats
            layer_stats.append({
                "Layer": name,
                "Shape": list(param.shape),
                "Parameters": param_count,
                "Size (MB)": size_mb,
                "% of Total": (param_count / total_params * 100) if total_params > 0 else 0
            })
    
    # Fix the percentage calculation after we know the total
    for stat in layer_stats:
        stat["% of Total"] = (stat["Parameters"] / total_params * 100)
    
    return pd.DataFrame(layer_stats), total_params, total_size_bytes

# Get detailed layer analysis
layer_df, total_params, total_size_bytes = analyze_layers(model)

# Add a total row
total_size_mb = total_size_bytes / (1000 * 1000)
layer_df = pd.concat([
    layer_df, 
    pd.DataFrame([{
        "Layer": "TOTAL",
        "Shape": None,
        "Parameters": total_params,
        "Size (MB)": total_size_mb,
        "% of Total": 100.0
    }])
])

# Print detailed layer information
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1500)
pd.set_option('display.float_format', '{:.4f}'.format)
print("\n🔍 DETAILED LAYER ANALYSIS:")
print(layer_df)

# Print summary statistics
print("\n📊 MODEL SUMMARY:")
print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Total size: {total_size_mb:.2f} MB ({total_size_mb/1000:.4f} GB)")

# Group by module types to see distribution
print("\n📈 PARAMETER DISTRIBUTION BY MODULE TYPE:")
layer_types = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        # Extract module type from name
        module_type = name.split('.')[0]
        if module_type not in layer_types:
            layer_types[module_type] = {"params": 0, "size": 0}
        
        params = param.numel()
        size = params * 4 / (1000 * 1000)  # MB
        
        layer_types[module_type]["params"] += params
        layer_types[module_type]["size"] += size

# Create and print summary DataFrame
summary_data = []
for module, stats in layer_types.items():
    summary_data.append({
        "Module": module,
        "Parameters": stats["params"],
        "Parameters (M)": stats["params"] / 1e6,
        "Size (MB)": stats["size"],
        "% of Total": stats["params"] / total_params * 100
    })

summary_df = pd.DataFrame(summary_data).sort_values("Parameters", ascending=False)
print(summary_df)

# Print model configuration
print("\n⚙️ MODEL ARCHITECTURE:")
print(f"Encoder-Decoder type: {model.__class__.__name__}")
print(f"Embedding dimension: {model.config.d_model}")
print(f"Encoder layers: {model.config.encoder_layers}")
print(f"Decoder layers: {model.config.decoder_layers}")
print(f"Attention heads: {model.config.encoder_attention_heads}")
print(f"Vocabulary size: {model.config.vocab_size}")


import torch
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import pandas as pd
from prettytable import PrettyTable

def count_parameters(model):
    """Count the parameters of a model and create a pretty table with details."""
    table = PrettyTable(["Module", "Parameters (M)", "Size (MB)", "% of Total"])
    total_params = 0
    total_size_bytes = 0
    module_details = []
    
    # Iterate through named parameters
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
            
        params = parameter.numel()
        total_params += params
        
        # Calculate size in bytes (assume float32 - 4 bytes per parameter)
        size_bytes = params * 4
        total_size_bytes += size_bytes
        
        module_name = name.split('.')[0] if '.' in name else name
        module_details.append((module_name, params, size_bytes))
    
    # Aggregate by module name
    module_stats = {}
    for name, params, size in module_details:
        if name not in module_stats:
            module_stats[name] = {"params": 0, "size": 0}
        module_stats[name]["params"] += params
        module_stats[name]["size"] += size
    
    # Add rows to the table
    for name, stats in module_stats.items():
        table.add_row([
            name, 
            f"{stats['params']/1e6:.2f}", 
            f"{stats['size']/1e6:.2f}", 
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    # Add a row for the total
    table.add_row(["Total", f"{total_params/1e6:.2f}", f"{total_size_bytes/1e6:.2f}", "100%"])
    
    return table, total_params, total_size_bytes

def analyze_model_architecture(model):
    """Analyze the model architecture and return detailed information."""
    print("\n📋 MODEL ARCHITECTURE SUMMARY")
    print("-" * 50)
    
    # Print model structure
    print(f"Model type: {model.__class__.__name__}")
    
    # Module summary
    module_types = {}
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if module_type not in module_types:
            module_types[module_type] = 0
        module_types[module_type] += 1
    
    print("\n📊 MODULE TYPES:")
    for module_type, count in module_types.items():
        print(f"  - {module_type}: {count}")

# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-fr-en"
print(f"Loading model: {model_name}")

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Detailed analysis
print("\n📏 MODEL SIZE ANALYSIS")
print("-" * 50)
table, total_params, total_size_bytes = count_parameters(model)
print(table)

# Calculate total size in different units
size_mb = total_size_bytes / (1000 * 1000)
size_gb = size_mb / 1000

print(f"\n📈 TOTAL MODEL STATISTICS:")
print(f"  - Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  - Size: {size_mb:.2f} MB ({size_gb:.4f} GB)")

# Get input/output embedding dimensions
print("\n🔍 EMBEDDING DIMENSIONS:")
print(f"  - Encoder embedding dim: {model.config.d_model}")
print(f"  - Decoder embedding dim: {model.config.d_model}")
print(f"  - Vocabulary size: {model.config.vocab_size}")

# Additional configuration details
print("\n⚙️ MODEL CONFIGURATION:")
for key, value in model.config.to_dict().items():
    if isinstance(value, (int, float, str, bool)) or value is None:
        print(f"  - {key}: {value}")

analyze_model_architecture(model)

# Example usage to show the model works
print("\n✅ MODEL TEST:")
inputs = tokenizer("Bonjour, comment ça va?", return_tensors="pt")
outputs = model.generate(**inputs)
print(f"Input: Bonjour, comment ça va?")
print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
