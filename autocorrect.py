import nltk
import re
import json
import time
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import math
import logging
from dataclasses import dataclass

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SpellingSuggestion:
    """Data class to represent a spelling suggestion"""
    word: str
    probability: float
    edit_distance: int
    similarity_score: float
    frequency: int = 0
    
    def __post_init__(self):
        # Enhanced confidence calculation
        freq_boost = math.log(self.frequency + 1) / 10  # Logarithmic frequency boost
        edit_penalty = 1 / (self.edit_distance + 1)
        self.confidence = self.probability * edit_penalty * self.similarity_score * (1 + freq_boost)

class CorpusProcessor:
    """Processes text corpus to extract vocabulary and statistics"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_vocabulary(self, text: str) -> Tuple[Set[str], Dict[str, int], Dict[Tuple[str, str], int]]:
        """Extract vocabulary, word frequencies, and bigrams from text"""
        
        print("🔄 Processing corpus...")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract words (only alphabetic, length > 1)
        words = re.findall(r'\b[a-z]{2,}\b', cleaned_text)
        
        print(f" Found {len(words)} total words in corpus")
        
        # Build vocabulary (exclude stop words for main vocabulary)
        vocabulary = set()
        word_frequencies = defaultdict(int)
        
        for word in words:
            # Always count frequency
            word_frequencies[word] += 1
            
            # Add to vocabulary (include stop words as they're useful for spell checking)
            if len(word) >= 2:
                vocabulary.add(word)
        
        # Build bigram frequencies for context
        bigram_frequencies = defaultdict(int)
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            bigram_frequencies[bigram] += 1
        
        print(f" Built vocabulary with {len(vocabulary)} unique words")
        print(f" Calculated {len(bigram_frequencies)} bigram patterns")
        
        return vocabulary, dict(word_frequencies), dict(bigram_frequencies)
    
    def analyze_corpus(self, file_path: str) -> Dict:
        """Analyze corpus and return comprehensive statistics"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"Loaded corpus from {file_path}")
            print(f"Original text length: {len(text):,} characters")
            
            # Extract vocabulary and statistics
            vocabulary, word_frequencies, bigram_frequencies = self.extract_vocabulary(text)
            
            # Calculate additional statistics
            total_words = sum(word_frequencies.values())
            unique_words = len(vocabulary)
            avg_word_length = sum(len(word) for word in vocabulary) / len(vocabulary)
            
            # Most common words
            most_common = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # Word length distribution
            length_dist = defaultdict(int)
            for word in vocabulary:
                length_dist[len(word)] += 1
            
            analysis = {
                'file_path': file_path,
                'text_length': len(text),
                'total_words': total_words,
                'unique_words': unique_words,
                'vocabulary_coverage': unique_words / total_words * 100,
                'average_word_length': round(avg_word_length, 2),
                'most_common_words': most_common[:10],
                'word_length_distribution': dict(length_dist),
                'bigram_count': len(bigram_frequencies),
                'vocabulary': vocabulary,
                'word_frequencies': word_frequencies,
                'bigram_frequencies': bigram_frequencies
            }
            
            return analysis
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found!")
            print("Make sure your data.txt file is in the same directory as this script")
            return None
        except Exception as e:
            print(f"Error processing corpus: {e}")
            return None

class AdvancedSpellChecker:
    """Advanced Spell Checker using corpus-based vocabulary"""
    
    def __init__(self, corpus_file: str = None):
        """Initialize the spell checker with corpus file"""
        self.lemmatizer = WordNetLemmatizer()
        self.corpus_processor = CorpusProcessor()
        
        # Initialize empty structures
        self.vocabulary: Set[str] = set()
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self.word_probabilities: Dict[str, float] = {}
        self.bigram_frequencies: Dict[Tuple[str, str], int] = defaultdict(int)
        self.corpus_stats: Dict = {}
        
        # Load corpus if provided
        if corpus_file:
            self.load_corpus(corpus_file)
        else:
            print("No corpus file provided. Use load_corpus() method to load your data.txt file")
    
    def load_corpus(self, file_path: str) -> bool:
        """Load and process corpus from file"""
        print(f"\n Loading corpus from: {file_path}")
        
        analysis = self.corpus_processor.analyze_corpus(file_path)
        
        if analysis is None:
            return False
        
        # Store analysis results
        self.corpus_stats = analysis
        self.vocabulary = analysis['vocabulary']
        self.word_frequencies = defaultdict(int, analysis['word_frequencies'])
        self.bigram_frequencies = defaultdict(int, analysis['bigram_frequencies'])
        
        # Calculate probabilities
        total_words = sum(self.word_frequencies.values())
        self.word_probabilities = {
            word: freq / total_words 
            for word, freq in self.word_frequencies.items()
        }
        
        # Display corpus statistics
        self.display_corpus_stats()
        
        return True
    
    def display_corpus_stats(self):
        """Display comprehensive corpus statistics"""
        stats = self.corpus_stats
        
        print(f"\n CORPUS ANALYSIS RESULTS")
        print(f"="*50)
        print(f" File: {stats['file_path']}")
        print(f" Text length: {stats['text_length']:,} characters")
        print(f" Total words: {stats['total_words']:,}")
        print(f" Unique words: {stats['unique_words']:,}")
        print(f" Vocabulary coverage: {stats['vocabulary_coverage']:.2f}%")
        print(f" Average word length: {stats['average_word_length']} characters")
        print(f" Bigram patterns: {stats['bigram_count']:,}")
        
        print(f"\n Most Common Words:")
        for i, (word, freq) in enumerate(stats['most_common_words'], 1):
            print(f"  {i:2d}. {word:<12} ({freq:,} times)")
        
        print(f"\n Word Length Distribution:")
        length_dist = stats['word_length_distribution']
        for length in sorted(length_dist.keys())[:10]:  # Show first 10 lengths
            count = length_dist[length]
            bar = "█" * min(50, count // 20)  # Visual bar
            print(f"  {length:2d} chars: {count:,} words {bar}")
    
    def edit_distance_1(self, word: str) -> Set[str]:
        """Generate all possible words with edit distance 1"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        # Deletions
        deletes = [L + R[1:] for L, R in splits if R]
        
        # Transpositions (switches)
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        
        # Replacements
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        
        # Insertions
        inserts = [L + c + R for L, R in splits for c in letters]
        
        return set(deletes + transposes + replaces + inserts)
    
    def edit_distance_2(self, word: str) -> Set[str]:
        """Generate all possible words with edit distance 2"""
        edit_1 = self.edit_distance_1(word)
        edit_2 = set()
        
        for e1 in edit_1:
            if e1 in self.vocabulary:  # Only expand words that are in vocabulary
                continue
            edit_2.update(self.edit_distance_1(e1))
        
        return edit_2
    
    def phonetic_similarity(self, word1: str, word2: str) -> float:
        """Calculate phonetic similarity between two words"""
        # Enhanced phonetic patterns
        phonetic_patterns = {
            'ph': 'f', 'gh': 'f', 'ck': 'k', 'qu': 'kw', 'x': 'ks',
            'z': 's', 'c': 'k', 'ow': 'o', 'ou': 'o', 'tion': 'shun',
            'sion': 'shun', 'ch': 'k', 'sh': 'sh', 'th': 't'
        }
        
        def phonetic_transform(word):
            for pattern, replacement in phonetic_patterns.items():
                word = word.replace(pattern, replacement)
            return word
        
        phone1 = phonetic_transform(word1)
        phone2 = phonetic_transform(word2)
        
        # Enhanced similarity calculation
        max_len = max(len(phone1), len(phone2))
        if max_len == 0:
            return 1.0
        
        # Count matching positions
        matches = sum(c1 == c2 for c1, c2 in zip(phone1, phone2))
        
        # Add bonus for common subsequences
        common_chars = set(phone1) & set(phone2)
        char_bonus = len(common_chars) / max(len(set(phone1)), len(set(phone2)))
        
        return (matches / max_len + char_bonus) / 2
    
    def get_suggestions(self, word: str, max_suggestions: int = 5, 
                       context: List[str] = None, debug: bool = False) -> List[SpellingSuggestion]:
        """Get spelling suggestions for a word"""
        word = word.lower().strip()
        
        if debug:
            print(f"\n Analyzing: '{word}'")
        
        # Check if word is already correct
        if word in self.vocabulary:
            freq = self.word_frequencies.get(word, 0)
            if debug:
                print(f" Word is correct! (frequency: {freq})")
            return [SpellingSuggestion(
                word=word,
                probability=self.word_probabilities.get(word, 0),
                edit_distance=0,
                similarity_score=1.0,
                frequency=freq
            )]
        
        if debug:
            print(f" Word not found in vocabulary")
        
        suggestions = []
        
        # Try edit distance 1
        candidates_1 = self.edit_distance_1(word) & self.vocabulary
        if debug:
            print(f" Edit distance 1: {len(candidates_1)} candidates")
        
        for candidate in candidates_1:
            freq = self.word_frequencies.get(candidate, 0)
            similarity = self.phonetic_similarity(word, candidate)
            
            suggestion = SpellingSuggestion(
                word=candidate,
                probability=self.word_probabilities.get(candidate, 0),
                edit_distance=1,
                similarity_score=similarity,
                frequency=freq
            )
            suggestions.append(suggestion)
            
            if debug:
                print(f" {candidate}: freq={freq}, sim={similarity:.3f}, conf={suggestion.confidence:.4f}")
        
        # Try edit distance 2 if needed
        if len(suggestions) < max_suggestions:
            candidates_2 = self.edit_distance_2(word) & self.vocabulary
            if debug:
                print(f" Edit distance 2: {len(candidates_2)} candidates")
            
            for candidate in candidates_2:
                if candidate not in [s.word for s in suggestions]:
                    freq = self.word_frequencies.get(candidate, 0)
                    similarity = self.phonetic_similarity(word, candidate)
                    
                    suggestion = SpellingSuggestion(
                        word=candidate,
                        probability=self.word_probabilities.get(candidate, 0),
                        edit_distance=2,
                        similarity_score=similarity,
                        frequency=freq
                    )
                    suggestions.append(suggestion)
        
        # Sort by confidence score
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply context if provided
        if context and len(context) > 0:
            suggestions = self._apply_context(suggestions, context, debug)
        
        return suggestions[:max_suggestions]
    
    def _apply_context(self, suggestions: List[SpellingSuggestion], 
                      context: List[str], debug: bool = False) -> List[SpellingSuggestion]:
        """Apply context to rerank suggestions"""
        if debug:
            print(f" Applying context: {context}")
        
        for suggestion in suggestions:
            context_boost = 0
            for ctx_word in context:
                if ctx_word in self.vocabulary:
                    # Check both directions
                    bigram1 = (ctx_word, suggestion.word)
                    bigram2 = (suggestion.word, ctx_word)
                    
                    boost1 = self.bigram_frequencies.get(bigram1, 0)
                    boost2 = self.bigram_frequencies.get(bigram2, 0)
                    context_boost += max(boost1, boost2)
            
            # Apply context boost
            if context_boost > 0:
                original_confidence = suggestion.confidence
                suggestion.confidence *= (1 + context_boost * 0.1)
                if debug:
                    print(f" {suggestion.word}: context boost {context_boost} -> "
                          f"confidence {original_confidence:.4f} -> {suggestion.confidence:.4f}")
        
        return sorted(suggestions, key=lambda x: x.confidence, reverse=True)
    
    def check_word(self, word: str, context: List[str] = None, debug: bool = False) -> Dict:
        """Check a single word and return detailed results"""
        suggestions = self.get_suggestions(word, context=context, debug=debug)
        
        if suggestions and suggestions[0].edit_distance == 0:
            status = "correct"
        elif suggestions:
            status = "misspelled"
        else:
            status = "unknown"
        
        return {
            'word': word,
            'status': status,
            'suggestions': [
                {
                    'word': s.word,
                    'confidence': round(s.confidence, 4),
                    'edit_distance': s.edit_distance,
                    'frequency': s.frequency,
                    'probability': round(s.probability, 6)
                }
                for s in suggestions
            ]
        }
    
    def save_vocabulary(self, filename: str = "vocabulary.json"):
        """Save the vocabulary and statistics to a file"""
        data = {
            'corpus_stats': self.corpus_stats,
            'vocabulary': list(self.vocabulary),
            'word_frequencies': dict(self.word_frequencies),
            'bigram_frequencies': {f"{k[0]}|{k[1]}": v for k, v in self.bigram_frequencies.items()},
            'total_words': sum(self.word_frequencies.values())
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f" Vocabulary saved to {filename}")


def main():
    """Main demonstration function"""
    print(" CORPUS-BASED SPELL CHECKER")
    print("="*50)
    
    # Initialize spell checker
    checker = AdvancedSpellChecker()
    
    # Try to load data.txt file
    corpus_files = ['data.txt']
    corpus_loaded = False
    
    for file_path in corpus_files:
        try:
            with open(file_path, 'r'):
                print(f" Found corpus file: {file_path}")
                if checker.load_corpus(file_path):
                    corpus_loaded = True
                    break
        except FileNotFoundError:
            continue
    
    if not corpus_loaded:
        print(f"\n No corpus file found!")
        print(f" Please make sure you have one of these files in your directory:")
        for file_path in corpus_files:
            print(f"   • {file_path}")
        
        file_path = input(f"\n Enter the path to your corpus file (or press Enter to use demo): ").strip()
        
        if file_path and checker.load_corpus(file_path):
            corpus_loaded = True
    
    if not corpus_loaded:
        print(f" Using demo mode with basic vocabulary...")
        # Add some basic words for demonstration
        demo_words = ["hello", "world", "tomorrow", "today", "python", "code", "spell", "check"]
        for word in demo_words:
            checker.vocabulary.add(word)
            checker.word_frequencies[word] = 10
        
        total = sum(checker.word_frequencies.values())
        checker.word_probabilities = {
            word: freq / total for word, freq in checker.word_frequencies.items()
        }
    
    # Interactive spell checking
    while True:
        print(f"\n" + "="*50)
        print(f" SPELL CHECKER MENU")
        print(f"="*50)
        print(f"1. Check single word")
        print(f"2. Check word with context")
        print(f"3. Show vocabulary statistics")
        print(f"4. Test with your examples")
        print(f"5. Save vocabulary")
        print(f"6. Exit")
        
        choice = input(f"\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            word = input(f"\n Enter word to check: ").strip()
            if word:
                result = checker.check_word(word, debug=True)
                
                print(f"\n RESULTS:")
                print(f"─"*30)
                
                if result['status'] == 'correct':
                    print(f" '{word}' is spelled correctly!")
                elif result['status'] == 'misspelled':
                    print(f" '{word}' appears to be misspelled.")
                    print(f" Suggestions:")
                    for i, suggestion in enumerate(result['suggestions'], 1):
                        print(f"  {i}. {suggestion['word']} "
                              f"(confidence: {suggestion['confidence']}, "
                              f"frequency: {suggestion['frequency']})")
                else:
                    print(f" No suggestions found for '{word}'")
        
        elif choice == '2':
            word = input(f"\n Enter word to check: ").strip()
            context_input = input(f" Enter context words (space-separated): ").strip()
            
            if word:
                context = context_input.split() if context_input else None
                result = checker.check_word(word, context=context, debug=True)
                
                print(f"\n RESULTS (with context):")
                print(f"─"*30)
                
                if result['status'] == 'correct':
                    print(f" '{word}' is spelled correctly!")
                elif result['suggestions']:
                    print(f" Context-aware suggestions:")
                    for i, suggestion in enumerate(result['suggestions'], 1):
                        print(f"  {i}. {suggestion['word']} "
                              f"(confidence: {suggestion['confidence']})")
        
        elif choice == '3':
            if corpus_loaded:
                checker.display_corpus_stats()
            else:
                print(f" Basic vocabulary statistics:")
                print(f"   Vocabulary size: {len(checker.vocabulary)}")
                print(f"   Sample words: {list(checker.vocabulary)[:10]}")
        
        elif choice == '4':
            test_words = ["tommrow", "tomorow", "tommorrow", "wrold", "recieve", "seperate"]
            print(f"\n Testing with common misspellings:")
            
            for word in test_words:
                result = checker.check_word(word)
                print(f"\n'{word}' -> ", end="")
                if result['suggestions']:
                    top_suggestion = result['suggestions'][0]
                    print(f"'{top_suggestion['word']}' (confidence: {top_suggestion['confidence']})")
                else:
                    print("No suggestions")
        
        elif choice == '5':
            filename = input(f"\n Enter filename (default: vocabulary.json): ").strip()
            checker.save_vocabulary(filename or "vocabulary.json")
        
        elif choice == '6':
            print(f"\n Thanks for using the Corpus-Based Spell Checker!")
            break
        
        else:
            print(f" Invalid choice. Please try again.")


if __name__ == "__main__":
    main()