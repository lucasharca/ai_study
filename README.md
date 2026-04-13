# AI Study: Natural Language Processing & Machine Learning

A comprehensive collection of study materials and practical examples covering NLP (Natural Language Processing) and ML (Machine Learning) techniques for text analysis. This repository contains foundational concepts, implementations, and hands-on projects.

## 📚 Project Overview

This project explores the complete NLP pipeline from raw text preprocessing to advanced machine learning applications. Each module builds upon previous concepts with both theoretical examples and practical exercises using real-world datasets.

---

## 🏗️ Project Structure

### 1. **Text Processing** (`text_processing/`)
Fundamental techniques for cleaning and preparing text data for analysis.

#### Core Concepts:
- **Lowercasing** - Normalize text by converting to lowercase
- **Tokenization** - Break text into sentences and individual words
- **Stopwords Removal** - Filter out common words (the, and, a) that add little value
- **Stemming** - Reduce words to their root form (e.g., "connecting" → "connect") using rule-based approach
- **Lemmatization** - Convert words to their base form using dictionary lookup (more accurate than stemming)
- **Regular Expressions** - Pattern matching and text manipulation
- **N-grams** - Analyze sequences of words (unigrams, bigrams, trigrams) and visualize frequency

#### Practical Task (`practical_task/`)
- **TripAdvisor Hotel Reviews Processing** - End-to-end pipeline applying all text processing techniques to real hotel review data
- Demonstrates proper text cleaning while preserving important context

---

### 2. **Vectorizing Text** (`vectorizing_text/`)
Convert text into numerical representations suitable for machine learning algorithms.

#### Techniques:
- **Bag of Words (BoW)** - Creates word frequency matrix counting word occurrences in documents
  - Simple and fast but loses word order and context
  
- **TF-IDF** - Term Frequency-Inverse Document Frequency
  - Weights words by importance across documents
  - Words appearing in many documents get lower weights
  - Better context preservation than Bag of Words

**Use Case:** Both techniques prepare text for classification and analysis by machines

---

### 3. **Text Classification** (`text_classifier/`)
Classify text into predefined categories using supervised machine learning.

#### Algorithms Implemented:
- **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
  - Fast and effective for text, assumes word independence
  
- **Logistic Regression** - Linear classification model
  - Good baseline, interpretable predictions, efficient
  
- **Linear Support Vector Machine (SVM)** - Finds optimal decision boundary
  - Effective for high-dimensional text data

**Application:** Sentiment classification on sample positive/negative sentence pairs

---

### 4. **Sentiment Analysis** (`sentiment_analysis/`)
Determine the emotional tone or opinion expressed in text.

#### Approaches:
- **Rule-Based Methods** 
  - TextBlob - Simple polarity and subjectivity analysis
  - VADER (Valence Aware Dictionary and sEntiment Reasoner) - Optimized for social media text
  - Uses predefined lexicons of words and their sentiment scores
  
- **Deep Learning Models**
  - Pre-trained Transformer Models via Hugging Face
  - Uses models like BERT-based classifiers for nuanced understanding
  - Can leverage specialized models (e.g., BERTweet for Twitter sentiment)

#### Practical Task (`practical_task/`)
- **Book Reviews Analysis** - Compare rule-based (VADER) vs transformer-based approaches
  - Clean book review data
  - Generate sentiment scores and labels
  - Compare performance of both methods on real-world data

---

### 5. **Text Tagging** (`text_tagging/`)
Identify and label linguistic properties of text.

#### Techniques:
- **Named Entity Recognition (NER)** - Extract and classify named entities
  - Identifies: People names, Organizations, Locations, Dates, Quantities
  - Uses spaCy's pre-trained models
  - Important lesson: Capitalization and punctuation affect recognition accuracy
  
- **Part of Speech (POS) Tagging** - Assign grammatical labels to words
  - Tags: Noun, Verb, Adjective, Adverb, Pronoun, etc.
  - Analyzes sentence structure and word relationships
  - Uses spaCy for tagging

#### Practical Task (`practical_task/`)
- **BBC News Headlines Analysis** - Apply NER and POS tagging to news titles
  - Identify entities in news content
  - Analyze grammatical structure of headlines
  - Visualize patterns in news language

---

### 6. **Topic Modeling** (`topic_modeling/`)
Discover hidden themes and topics in large document collections.

#### Approach:
- **Latent Dirichlet Allocation (LDA)** - Unsupervised learning algorithm
  - Assumes each document contains multiple topics
  - Each topic is a distribution of words
  - Useful for exploring document collections without labels
  - Uses Gensim library for implementation

**Application:** News articles topic extraction
- Analyze coherence of discovered topics
- Visualize topic-word relationships
- Determine optimal number of topics

---

### 7. **Fake News Detection** (`fake_news/`)
Comprehensive analysis and classification of fake news using multiple NLP techniques.

#### Techniques Applied:
- **Data Exploration** - Analyze distribution of fake vs factual news
- **POS Tagging Comparison** - Compare grammatical structures between fake and factual content
- **Named Entity Recognition** - Extract and analyze entities in news articles
- **Text Preprocessing Pipeline** - Clean and prepare text data
- **N-grams Analysis** - Identify common word patterns
- **Sentiment Analysis** - Compare emotional tone using VADER
- **Topic Modeling** - Discover themes in fake news using LDA and LSA
- **Classification Models** - Build Logistic Regression and SVM classifiers to detect fake news

#### Practical Example (`practical_exemple.py`)
- **End-to-End Fake News Analysis** - Demonstrates complete NLP workflow from preprocessing to classification
- Integrates multiple techniques learned throughout the course
- Achieves high accuracy in fake news detection

---

## 🛠️ Dependencies

All required packages are listed in `requirements.txt`:
- **NLTK** - Natural Language Toolkit (tokenization, stemming, POS tagging)
- **spaCy** - Industrial NLP library (named entity recognition, POS tagging)
- **scikit-learn** - Machine learning library (text vectorization, classifiers)
- **TextBlob** - Rule-based sentiment analysis
- **VADER** (vaderSentiment) - Social media sentiment analysis
- **Transformers** (Hugging Face) - Pre-trained deep learning models
- **Gensim** - Topic modeling and word embeddings
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **PyTorch** - Deep learning framework (required by transformers)

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🎓 Learning Path

**Recommended study order:**

1. **Start with fundamentals** → Text Processing modules
2. **Learn vectorization** → Vectorizing Text modules
3. **Apply to classification** → Text Classification examples
4. **Analyze sentiment** → Sentiment Analysis with both approaches
5. **Extract information** → Text Tagging (NER & POS)
6. **Discover patterns** → Topic Modeling
7. **Detect misinformation** → Fake News Detection

Each module includes working examples that can be executed independently or as part of the workflow.

---

## 💡 Key Concepts Illustrated

- **Text Preprocessing Pipeline:** How to properly clean text while preserving important information
- **Feature Engineering for Text:** Converting unstructured text to numerical features
- **Supervised vs Unsupervised Learning:** Classification vs topic modeling approaches
- **Rule-based vs ML Approaches:** Traditional methods vs deep learning (sentiment analysis)
- **Trade-offs:** Stemming vs lemmatization, BoW vs TF-IDF, speed vs accuracy
- **Real-world Applications:** Practical examples with actual datasets (hotels, reviews, news, fake news detection)

---

## 📊 Practical Tasks

This project includes four comprehensive practical exercises:

1. **Text Processing Task** - Clean and prepare TripAdvisor reviews for analysis
2. **Sentiment Analysis Task** - Compare VADER and transformer approaches on book reviews
3. **Text Tagging Task** - Extract entities and analyze grammatical structure of news headlines
4. **Fake News Detection Task** - Complete analysis pipeline applying all learned techniques to classify fake vs factual news

Each practical task demonstrates how to apply multiple techniques to solve real-world NLP problems.

---

## 🚀 Usage

Each Python file can be run independently as a learning example:

```bash
# Text processing example
python text_processing/tokenization.py

# Classification example
python text_classifier/logistic_regression.py

# Fake news detection example
python fake_news/practical_exemple.py

# Practical tasks
python sentiment_analysis/practical_task/index.py
python text_processing/practical_task/text_processing.py
python text_tagging/practical_task/index.py
```

---

## 📝 Note

This is a **study and reference repository**, created for learning and exploring NLP/ML concepts. Each file includes explanatory comments and demonstrates practical implementations of the techniques.

---

## 🔗 Resources Used

- NLTK Documentation & Tutorial
- spaCy Official Documentation
- Scikit-learn Documentation
- Hugging Face Transformers
- Gensim Topic Modeling
- Various NLP and ML best practices

---

**Last Updated:** 2026
**Purpose:** Educational reference for NLP and ML concepts
