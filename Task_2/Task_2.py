import os
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.probability import FreqDist
import tkinter as tk
from tkinter import font as tkFont

#main window
window = tk.Tk()
window.title("Definitely not Semantris")
window.geometry("800x900")  # Changed height to 800

#font style and size
default_font = tkFont.Font(family="Helvetica", size=14)

#game logic 
filename = 'common_word_list.txt'
score = 0 
unique_words = set()
time_left = 30

#source words
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

#first list of words
random_words = random.sample(filtered_words_list, 10)
target_word = random.choice(random_words)
game_started = False

def display_random_words():
    input_display.delete(1.0, tk.END)  # Clear previous text
    for word in random_words:
        if word == target_word:
            input_display.insert(tk.END, word + "\n", "highlight")  #tag for the target word
        else:
            input_display.insert(tk.END, word + "\n")
    input_display.see(tk.END)

def update_timer():
    global time_left
    if time_left > 0:
        time_left -= 1
        timer_label.config(text=f"Time left: {time_left} seconds")
        window.after(1000, update_timer)
    else:
        feedback_label.config(text=f"You achieved {score} points!")
        entry.config(state='disabled')  #disable input when time is up

def play_game(event=None):
    global score, target_word, random_words, game_started, time_left
    
    player_word = entry.get().strip()
    if player_word:  # Ensure the input is not empty
        input_display.insert(tk.END, player_word + "\n")
        input_display.see(tk.END)
        
        if not game_started: #start the timer
            game_started = True
            update_timer()
        
        similarity_scores = []
        player_synsets = wn.synsets(player_word)

        #check and order by similarity
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

        top_5_words = sorted_words[:5]
        target_in_top_5 = any(word == target_word for word, score in top_5_words)

        sorted_words = [word for word, score in sorted_words]

        if target_in_top_5:
            for idx, (word, _) in enumerate(top_5_words):
                if word == target_word:
                    sorted_words = sorted_words[5:]
                    points = 6 - int(idx + 1)
                    score += points
                    time_left += max(0,points-2)
                    feedback_label.config(text=f"You gained: {points} points!")
                    score_label.config(text=f"Score: {score}")

                    #source 5 new words
                    new_random_words = random.sample(
                        [word for word in filtered_words_list if word not in random_words], 5
                    )
                    random_words = sorted_words + new_random_words
                    target_word = random.choice(random_words)
                    break
        else:
            feedback_label.config(text=f"Target word '{target_word}' was not in the top 5.")

        display_random_words()

    entry.delete(0, tk.END)


instructions_label = tk.Label(window, text="Try writing a word so that the highlighted one is in the \ntop 5 by semantic similarity,you will receive points based on\n how strong the similarity is!\nAfter your first word, you have 30 seconds to get as many points as possible.", font=default_font, wraplength=600)
instructions_label.pack(pady=20)

entry = tk.Entry(window, width=50, font=default_font)
entry.pack(pady=20)

submit_button = tk.Button(window, text="Submit", command=play_game, font=default_font)
submit_button.pack(pady=10)

input_display = tk.Text(window, height=20, width=20, font=default_font)
input_display.pack(pady=20)

input_display.tag_configure("highlight", foreground="red")

feedback_label = tk.Label(window, text="", wraplength=400, font=default_font)
feedback_label.pack(pady=10)

score_label = tk.Label(window, text=f"Score: {score}", font=default_font)
score_label.pack(pady=10)

timer_label = tk.Label(window, text=f"Time left: {time_left} seconds", font=default_font)
timer_label.pack(pady=10)

window.bind('<Return>', play_game)

display_random_words()

window.mainloop()
