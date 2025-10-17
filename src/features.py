import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import string
import os
from collections import Counter
import textstat

from data_loader import load_dataset

# NLTK Setup
project_root = os.path.dirname(os.path.dirname(__file__))
nltk_data_path = os.path.join(project_root, 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download required NLTK resources
required_resources = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
for resource in required_resources:
    try:
        if resource in ['punkt', 'punkt_tab']:
            nltk.data.find(f'tokenizers/{resource}')
        elif resource == 'averaged_perceptron_tagger':
            nltk.data.find('taggers/averaged_perceptron_tagger')
        else:
            nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path, quiet=True)

# Load stopwords and common words
STOPWORDS = set(stopwords.words('english'))
COMMON_WORDS = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 
                    'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 
                    'do', 'at', 'this', 'but', 'his', 'by', 'from'])

def extract_basic_features(text):
    """Extract basic text statistics"""
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    word_count = len(words)
    char_count = len(text)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length
    }

def extract_punctuation_features(text):
    """Extract punctuation-related features"""
    punctuation_chars = ['.', ',', ';', ':', '!', '?', '-', '(', ')', '"', "'"]
    punctuation_count = sum(text.count(p) for p in punctuation_chars)
    
    return {
        'punctuation_count': punctuation_count,
        'punctuation_ratio': punctuation_count / len(text) if len(text) > 0 else 0
    }

def extract_case_features(text):
    """Extract case-related features"""
    letters = [c for c in text if c.isalpha()]
    uppercase_count = sum(1 for c in letters if c.isupper())
    
    return {
        'uppercase_ratio': uppercase_count / len(letters) if letters else 0
    }

def extract_vocabulary_features(text):
    """Extract vocabulary richness and stopword features"""
    words = word_tokenize(text.lower())
    words_alpha = [w for w in words if w.isalpha()]
    
    unique_words = set(words_alpha)
    stopwords_in_text = [w for w in words_alpha if w in STOPWORDS]
    rare_words = [w for w in words_alpha if w not in COMMON_WORDS]
    
    return {
        'unique_word_ratio': len(unique_words) / len(words_alpha) if words_alpha else 0,
        'stopword_ratio': len(stopwords_in_text) / len(words_alpha) if words_alpha else 0,
        'rare_word_ratio': len(rare_words) / len(words_alpha) if words_alpha else 0
    }

def extract_pos_features(text):
    """Extract part-of-speech distribution features"""
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    total_tags = len(pos_tags)
    if total_tags == 0:
        return {
            'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0,
            'adv_ratio': 0, 'pronoun_ratio': 0
        }
    
    pos_counts = Counter(tag for word, tag in pos_tags)
    
    noun_tags = sum(pos_counts.get(tag, 0) for tag in ['NN', 'NNS', 'NNP', 'NNPS'])
    verb_tags = sum(pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
    adj_tags = sum(pos_counts.get(tag, 0) for tag in ['JJ', 'JJR', 'JJS'])
    adv_tags = sum(pos_counts.get(tag, 0) for tag in ['RB', 'RBR', 'RBS'])
    pronoun_tags = sum(pos_counts.get(tag, 0) for tag in ['PRP', 'PRP$', 'WP', 'WP$'])
    
    return {
        'noun_ratio': noun_tags / total_tags,
        'verb_ratio': verb_tags / total_tags,
        'adj_ratio': adj_tags / total_tags,
        'adv_ratio': adv_tags / total_tags,
        'pronoun_ratio': pronoun_tags / total_tags
    }

def extract_repetition_features(text):
    """Extract word repetition metrics"""
    words = word_tokenize(text.lower())
    words_alpha = [w for w in words if w.isalpha()]
    
    if not words_alpha:
        return {'repetition_rate': 0}
    
    word_counts = Counter(words_alpha)
    repeated_words = sum(count for count in word_counts.values() if count > 1)
    
    return {
        'repetition_rate': repeated_words / len(words_alpha)
    }

def extract_readability_features(text):
    """Extract readability scores using textstat library"""
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text)
    }

def extract_all_features(text):
    """Extract all features from text"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {}
    
    features = {}
    features.update(extract_basic_features(text))
    features.update(extract_punctuation_features(text))
    features.update(extract_case_features(text))
    features.update(extract_vocabulary_features(text))
    features.update(extract_pos_features(text))
    features.update(extract_repetition_features(text))
    features.update(extract_readability_features(text))
    
    return features

def build_feature_matrix(df, text_column='text'):
    """Build feature matrix from DataFrame"""
    features_list = []
    
    print(f"Processing {len(df)} texts...")
    for idx, text in enumerate(df[text_column]):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} texts...")
        features = extract_all_features(str(text))
        features_list.append(features)
    
    feature_df = pd.DataFrame(features_list)
    feature_df = feature_df.fillna(0)
    
    print(f"✓ Feature extraction complete!")
    return feature_df


def process_csv(csv_path, text_column='text', label_column='generated', output_path=None):
    
    print(f"\n{'='*60}")
    print(f"CSV Text Feature Extraction")
    print(f"{'='*60}\n")
    
    df = load_dataset(csv_path, text_column='text', label_column='generated')
    # Show data preview
    print(f"\nData preview:")
    print(f"-" * 60)
    print(df.head(3))
    print(f"-" * 60)
    
    # Show label distribution if available
    if label_column:
        print(f"\nLabel distribution in '{label_column}':")
        print(df[label_column].value_counts())
    
    # Extract features
    print(f"\n{'='*60}")
    print(f"Extracting features from text...")
    print(f"{'='*60}\n")
    
    print('number of rows loaded ', len(df))
    feature_matrix = build_feature_matrix(df, text_column=text_column)
    
    # Combine original data with features
    print(f"\nCombining features with original data...")
    result_df = pd.concat([df, feature_matrix], axis=1)
    
    # Save to CSV
    if output_path is None:
        output_path = 'processed_features.csv'
    
    print(f"\nSaving results to: {output_path}")
    result_df.to_csv(output_path, index=False)
    print(f"✓ File saved successfully!")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total texts processed: {len(result_df)}")
    print(f"Original columns: {len(df.columns)}")
    print(f"Feature columns added: {len(feature_matrix.columns)}")
    print(f"Total columns in output: {len(result_df.columns)}")
    print(f"\nFeature columns added:")
    for i, col in enumerate(feature_matrix.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n{'='*60}\n")
    
    return result_df

# Example usage
if __name__ == "__main__":
    # MODIFY THESE PARAMETERS FOR YOUR CSV
    csv_file = "/Users/susmitapaudel/projects/ai-text-detector/data/raw/AI_Human.csv"          # Path to your CSV file
    text_col = "text"                   # Name of column with text
    label_col = "generated"             # Name of column with labels (0/1, human/ai, etc.)
    output_file = "/Users/susmitapaudel/projects/ai-text-detector/data/raw/features_output.csv" # Where to save results
    
    # Process the CSV
    result = process_csv(
        csv_path=csv_file,
        text_column=text_col,
        label_column=label_col,
        output_path=output_file
    )
    
    # Display some results
    if result is not None:
        print("\nFirst few rows with features:")
        print(result.head())
        
        print("\nFeature statistics:")
        print(result.describe())