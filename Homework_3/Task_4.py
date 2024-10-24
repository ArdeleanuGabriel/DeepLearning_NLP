from transformers import GPT2LMHeadModel, GPT2Tokenizer

#GPT-2 model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#sentence start
sentence_start = "We wish that you"
#feed input to model
input_ids = tokenizer.encode(sentence_start, return_tensors='pt')

num_words_to_generate = 3

output = model.generate(input_ids, max_length=len(input_ids[0]) + num_words_to_generate, num_return_sequences=1)

predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

predicted_words = predicted_text.split()[len(sentence_start.split()):]
next_generated_words = predicted_words[:num_words_to_generate]
print(f"Predicted next words: {sentence_start} + {next_generated_words}")
