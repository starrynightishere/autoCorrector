# Word Correction and Probability Analysis

This Python library implements an **advanced, corpus-based spell checker** that analyzes word frequencies, suggests corrections based on **edit distance**, **phonetic similarity**, and **contextual bigrams**, and provides rich insights from the input corpus.

---

## Features

- Extracts vocabulary, word frequency, and bigram patterns from any text file  
- Suggests spelling corrections using:
  - Edit distance (1 & 2)
  - Phonetic transformations
  - Word probability and frequency
  - Context-aware scoring using bigrams
- Interactive CLI menu for demo and testing
- Saves vocabulary and statistics to a JSON file
- Handles contractions and punctuation robustly

---

## Prerequisites

- Python 3.7 or later
- Required libraries:
  - [`nltk`](https://www.nltk.org/)
  - [`pattern`](https://github.com/clips/pattern) (for lemmatization, optional)

Install dependencies using pip:

```bash
pip install nltk
pip install pattern

