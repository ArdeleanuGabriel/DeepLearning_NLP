from collections import defaultdict, Counter
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")

def compute_pair_freqs(word_freqs, splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a, b, word_freqs, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits

def tokenize(text, tokenizer, merges):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

def train_ngram(tokenized_sentences, n):
    ngrams = defaultdict(Counter)
    for sentence in tokenized_sentences:
        sentence = ['<s>'] * (n - 1) + sentence + ['</s>']
        for i in range(len(sentence) - n + 1):
            context = tuple(sentence[i:i + n - 1])
            token = sentence[i + n - 1]
            ngrams[context][token] += 1
    return ngrams

def predict_next(context, ngrams):
    if context in ngrams:
        return max(ngrams[context], key=ngrams[context].get)
    else:
        return '<UNK>'

if __name__ == '__main__':
    corpus = [
        "there is a big house",
        "i buy a house",
        "they buy the new house"
    ]

    word_freqs = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = []
    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()

    vocab = ["<|endoftext|>"] + alphabet.copy()
    splits = {word: [c for c in word] for word in word_freqs.keys()}

    merges = {("Ġ", "t"): "Ġt"}
    vocab.append("Ġt")

    vocab_size = 50

    splits = merge_pair("Ġ", "t", word_freqs, splits)

    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(word_freqs, splits)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq

        splits = merge_pair(best_pair[0], best_pair[1], word_freqs, splits)

        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    print("Final Vocabulary:", vocab)

    tokenized_sentences = [tokenize(text, tokenizer, merges) for text in corpus]
    print("Tokenized Sentences:", tokenized_sentences)

    ngram_model = train_ngram(tokenized_sentences, 5)

    context = ('there',)
    next_token = predict_next(context, ngram_model)
    print(f"Predicted next token after '{context}': {next_token}")
