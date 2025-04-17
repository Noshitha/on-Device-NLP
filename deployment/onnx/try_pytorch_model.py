from transformers import MarianMTModel, MarianTokenizer

# Load the tokenizer and model
model_name = "Helsinki-NLP/opus-mt-fr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Text to translate
french_text = "je regarde la tele et je mange du fromage"

# Tokenize the text
tokens = tokenizer(french_text, return_tensors="pt", padding=True)

# Perform translation
translated = model.generate(**tokens)

# Decode the translated tokens
english_translation = tokenizer.decode(translated[0], skip_special_tokens=True)

print(english_translation)
