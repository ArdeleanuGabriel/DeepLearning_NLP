from collections import defaultdict

# Example sentences from the training corpus
corpus = [
    "there is a big house",
    "i buy a house",
    "they buy the new house"
]

#tokenize corpus at the character level 
def tokenize_corpus(corpus):
    return [list(word) + ['_'] for sentence in corpus for word in sentence.split()]

#get frequency for each token pair
def get_pair_frequencies(tokenized_corpus):
    pairs = defaultdict(int)
    for token_list in tokenized_corpus:
        for i in range(len(token_list) - 1):
            pairs[(token_list[i], token_list[i + 1])] += 1
    return pairs

#merge the most frequent pair
def merge_pair(pair, tokenized_corpus):
    new_tokenized = []
    for token_list in tokenized_corpus:
        new_token = []
        i = 0
        while i < len(token_list):
            if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == pair:
                new_token.append(token_list[i] + token_list[i + 1])
                i += 2
            else:
                new_token.append(token_list[i])
                i += 1
        new_tokenized.append(new_token)
    return new_tokenized

def byte_pair_encoding(corpus, num_merges):
    tokenized_corpus = tokenize_corpus(corpus)
    merges = []

    for merge_index in range(1, num_merges + 1):
        pair_freqs = get_pair_frequencies(tokenized_corpus)
        if not pair_freqs:
            break
        best_pair = max(pair_freqs, key=pair_freqs.get)
        merges.append(best_pair)
        tokenized_corpus = merge_pair(best_pair, tokenized_corpus)
        print(f"Merge {merge_index} pair: {best_pair}")
    return tokenized_corpus, merges

def apply_bpe_to_new_corpus(new_corpus, original_merges):
    tokenized_corpus = tokenize_corpus(new_corpus)
    for merge_index, pair in enumerate(original_merges, 1):
        tokenized_corpus = merge_pair(pair, tokenized_corpus)
        print(f"Applying merge {merge_index} pair: {pair}")
    return tokenized_corpus

#train BPE
num_merges = 10
final_tokens, original_merges = byte_pair_encoding(corpus, num_merges)
print("\nTrained tokens:")
for token_list in final_tokens:
    print(' '.join(token_list))

vocabulary = set()
for token_list in final_tokens:
    vocabulary.update(token_list)

print("\nVocabulary after training BPE:")
print(f"{vocabulary}\n")

#apply trained BPE
new_corpus = ["there was a bug house"]
final_new_tokens = apply_bpe_to_new_corpus(new_corpus, original_merges)

print("\nApplied on test sentence:")
for token_list in final_new_tokens:
    print(' '.join(token_list))
