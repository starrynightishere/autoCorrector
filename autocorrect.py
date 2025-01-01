
import nltk
nltk.download('all')
import re
import pattern
from pattern.en import lexeme
from nltk.stem import WordNetLemmatizer

# Initialize an empty list to store words
w = []
# Read the text file
with open('/content/final.txt', 'r', encoding="utf8") as f:
	file_name_data = f.read()
	file_name_data = file_name_data.lower()
	w = re.findall('\w+', file_name_data)  # Tokenize the text into words

# Create a vocabulary set
main_set = set(w)

# Function to count the frequency of words in the entire text file
def counting_words(words):
	word_count = {}
	for word in words:
		if word in word_count:
			word_count[word] += 1
		else:
			word_count[word] = 1
	return word_count

# Function to calculate the probability of each word
def prob_cal(word_count_dict):
	probs = {}
	m = sum(word_count_dict.values())
	for key in word_count_dict.keys():
		probs[key] = word_count_dict[key] / m
	return probs

# Function to extract the root word (Lemma) using the 'pattern' module
def LemmWord(word):
	return list(lexeme(wd) for wd in word.split())[0]

# Function to delete letters from words
def DeleteLetter(word):
	delete_list = []
	split_list = []

	# Consider letters from 0 to i then i to -1, leaving the ith letter
	for i in range(len(word)):
		split_list.append((word[0:i], word[i:]))

	for a, b in split_list:
		delete_list.append(a + b[1:])
	return delete_list

# Function to switch two letters in a word
def Switch_(word):
	split_list = []
	switch_l = []

	# Create pairs of the words (and break them)
	for i in range(len(word)):
		split_list.append((word[0:i], word[i:]))

	# Print the first word (i.e., 'a') then replace the first and second character of 'b'
	switch_l = [a + b[1] + b[0] + b[2:] for a, b in split_list if len(b) >= 2]
	return switch_l

def Replace_(word):
	split_l = []
	replace_list = []

	# Replace the letter one-by-one from the list of alphabets
	for i in range(len(word)):
		split_l.append((word[0:i], word[i:]))
	alphabets = 'abcdefghijklmnopqrstuvwxyz'
	replace_list = [a + l + (b[1:] if len(b) > 1 else '') for a, b in split_l if b for l in alphabets]
	return replace_list

def insert_(word):
	split_l = []
	insert_list = []

	# Make pairs of the split words
	for i in range(len(word) + 1):
		split_l.append((word[0:i], word[i:]))

	# Store new words in a list, adding one new character at each location
	alphabets = 'abcdefghijklmnopqrstuvwxyz'
	insert_list = [a + l + b for a, b in split_l for l in alphabets]
	return insert_list

# Collect all the words in a set (so that no word repeats)
def colab_1(word, allow_switches=True):
	colab_1 = set()
	colab_1.update(DeleteLetter(word))
	if allow_switches:
		colab_1.update(Switch_(word))
	colab_1.update(Replace_(word))
	colab_1.update(insert_(word))
	return colab_1

# Collect words by allowing switches
def colab_2(word, allow_switches=True):
	colab_2 = set()
	edit_one = colab_1(word, allow_switches=allow_switches)
	for w in edit_one:
		if w:
			edit_two = colab_1(w, allow_switches=allow_switches)
			colab_2.update(edit_two)
	return colab_2

# Only store values that are in the vocabulary
def get_corrections(word, probs, vocab, n=2):
	suggested_word = []
	best_suggestion = []
	suggested_word = list(
		(word in vocab and word) or colab_1(word).intersection(vocab)
		or colab_2(word).intersection(vocab))

	# Find words with high frequencies
	best_suggestion = [[s, probs[s]] for s in list(reversed(suggested_word))]
	return best_suggestion

# Input
my_word = input("Enter any word:")

# Count word occurrences
word_count = counting_words(main_set)

# Calculate word probabilities
probs = prob_cal(word_count)

# Only store correct words
tmp_corrections = get_corrections(my_word, probs, main_set, 2)
for i, word_prob in enumerate(tmp_corrections):
	if(i < 3):
		print(word_prob[0])
	else:
		break
