# Install the necessary libraries if you haven't already
# !pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the input text
input_text = "We wish you a"

# Encode the input text and create tensor inputs for the model
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate predictions (next two words)
# We can specify the number of words we want to predict
output = model.generate(input_ids, max_length=len(input_ids[0]) + 2, num_return_sequences=1)

# Decode the output tensor to get the predicted text
predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Extract the predicted words
predicted_words = predicted_text.split()[len(input_text.split()):]
next_two_words = predicted_words[:2]

# Print the predicted next two words
print("Predicted next two words:", ' '.join(next_two_words))
