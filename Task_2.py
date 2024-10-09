import os
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.probability import FreqDist

filename = 'common_word_list.txt'
score = 0 
unique_words = set()

if os.path.exists(filename) and os.path.getsize(filename) > 0:
    with open(filename, 'r') as file:
        unique_words = set(file.read().splitlines())
    unique_words_list = sorted(unique_words)
    filtered_words_list = [word for word in unique_words_list if word.isalpha() and len(word) > 4]
else:
    brown_words = brown.words()
    freq_dist = FreqDist(brown_words)
    common_words = [word for word in freq_dist if freq_dist[word] > 20]
    filtered_words_list = [word for word in common_words if word.isalpha() and len(word) > 4]
    filtered_words_list = sorted(filtered_words_list)

    with open(filename, 'w') as file:
        for word in filtered_words_list:
            file.write(word + '\n')

random_words = random.sample(filtered_words_list,10)
target_word = random.choice(random_words)

while True:
    print("_" * 100)
    print("Selected words:", random_words)
    print("Target word:", target_word)
    print("Score:", score)

    player_word = input("Please enter a single word: ").strip()

    similarity_scores = []
    player_synsets = wn.synsets(player_word)

    for word in random_words:
        word_synsets = wn.synsets(word)
        max_similarity = 0
        for ps in player_synsets:
            for ws in word_synsets:
                similarity = wn.wup_similarity(ps, ws)
                if similarity is not None:
                    max_similarity = max(max_similarity, similarity)
        similarity_scores.append((word, max_similarity))

    sorted_words = sorted(similarity_scores, key=lambda x: x[1], reverse=True)


    for idx, (word, s_score) in enumerate(sorted_words):
        print(f"{idx + 1}.{word} ( {s_score} )")
        
    top_5_words = sorted_words[:5]
    target_in_top_5 = any(word == target_word for word, score in top_5_words)

    sorted_words = [word for word, score in sorted_words]

    if target_in_top_5:
        for idx, (word, _) in enumerate(top_5_words):
            if word == target_word:
                sorted_words = sorted_words[5:]
                points = 6 - int(idx + 1)
                score += points
                print(f"You gained: {points} points!")

                new_random_words = random.sample(
                    [word for word in filtered_words_list if word not in random_words], 5
                )
                print(random_words)
                print(new_random_words)
                print(sorted_words)
                random_words = sorted_words + new_random_words
                target_word = random.choice(random_words)
                break
    else:
        print(f"Target word '{target_word}' was not in the top 5.")
