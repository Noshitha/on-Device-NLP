'''
conda create -n mt-312 python=3.12 anaconda
conda activate mt-312
pip install transformers onnx onnxruntime optimum[onnxruntime] torch sentencepiece
'''
import os
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoConfig # Added AutoConfig

# --- Configuration ---
hf_model_id = "Helsinki-NLP/opus-mt-de-en"
onnx_save_directory = "./onnx_model_de_en" # Directory to save the ONNX model

# --- Ensure output directory exists ---
os.makedirs(onnx_save_directory, exist_ok=True)

print(f"Starting conversion for model: {hf_model_id}")
print(f"ONNX model will be saved to: {onnx_save_directory}")

try:
    # --- Load tokenizer and config ---
    print("Loading tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    # Load the original config - needed by save_pretrained later
    config = AutoConfig.from_pretrained(hf_model_id)

    # --- Convert the model to ONNX ---
    # This step primarily generates the .onnx files but doesn't
    # fully save the wrapper object's state in the target directory yet.
    print("Converting model to ONNX format (creating .onnx files)...")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        hf_model_id,
        export=True,
        from_transformers=True,
        # Pass the loaded config explicitly during export
        config=config
    )

    # --- Save the ORTModel wrapper, tokenizer and config ---
    # This step is CRUCIAL: it saves the necessary configuration files
    # (like config.json adapted for ONNX) alongside the .onnx files
    # created in the previous step, allowing from_pretrained to load it back correctly.
    print("Saving ONNX model components, tokenizer and configuration...")
    model.save_pretrained(onnx_save_directory)
    tokenizer.save_pretrained(onnx_save_directory)

    print("-" * 30)
    print(f"Successfully converted '{hf_model_id}' to ONNX.")
    print(f"Files saved in: {onnx_save_directory}")
    # Let's list the files to see what was generated/saved
    if os.path.exists(onnx_save_directory):
         print("Generated files:", os.listdir(onnx_save_directory))
    else:
         print("Warning: Save directory not found after saving.")
    print("-" * 30)


    # --- Optional: Test the converted ONNX model ---
    print("\n--- Testing the ONNX model ---")

    # Load the exported ONNX model and tokenizer from the save directory
    print("Loading ONNX model and tokenizer for testing...")
    onnx_tokenizer = AutoTokenizer.from_pretrained(onnx_save_directory)

    # Now load the ORTModel from the directory where we saved it
    # It should find the config files saved by model.save_pretrained()
    onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_save_directory)

    german_text = "Hallo, wie geht es Ihnen?"
    print(f"Input (German): {german_text}")

    # Tokenize the input
    inputs = onnx_tokenizer(german_text, return_tensors="pt") # Use PyTorch tensors

    # Generate translation using the ONNX model
    print("Generating translation...")
    # Ensure model is on CPU if testing locally without GPU provider
    # (Usually handled by Optimum, but good to be aware)
    generated_ids = onnx_model.generate(**inputs)

    # Decode the output
    english_translation = onnx_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Output (English): {english_translation}")
    print("--- Test complete ---")


except Exception as e:
    print(f"\nAn error occurred during conversion or testing: {e}")
    import traceback
    traceback.print_exc()
