"""
# News Article Classification Pipeline: Usage Instructions

This script implements a machine learning pipeline to classify news articles using a 
combination of NLP (TF-IDF) and metadata analysis.

## Prerequisites
Ensure you have the following libraries installed:
    pip install pandas numpy scikit-learn matplotlib

## Input Files
The script expects two CSV files in the working directory:
1. 'development.csv': Training data containing 'title', 'article', 'source', and 'label'.
2. 'evaluation.csv': Data to be predicted, containing the same structure (except 'label').

## Pipeline Workflow
1. TEXT CLEANING: Normalizes HTML entities, removes tags, and fixes encoding artifacts.
2. FEATURE EXTRACTION:
   - Extracts image context from 'alt' tags.
   - Categorizes RSS feeds into topics (business, tech, etc.).
   - Parses URLs to extract domain names and path keywords.
   - Calculates 'Style' features (digit density, word counts).
3. CLASSIFICATION:
   - Combines Word-level Tf-idf and Numerical Scaling.

## Output
The script generates 'submission.csv', containing the predictions 
for the evaluation set regarding the ensemble method.
(We also included the possibility of generating the output for the 
LogisticRegression and RandomForestClassifier methods at the bottom of the code)
"""
import pandas as pd
import numpy as np
import re
import html
import string
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

# --- TEXT & URL PREPROCESSING ---
def clean_text(text):
    """Basic normalization: fixes HTML, encodings, and white spaces."""
    if not isinstance(text, str): 
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\\', '')
    text = text.replace('Ã¢Â€Â”', ' — ').replace('Ã¢Â€Â™', "'").replace('Ã¢Â€Âœ', '"').replace('Ã¢Â€Â', '"')
    text = re.sub(r'\s+', ' ', text)
    return text.strip() 


def url_preprocess(df):
    """ Extraction of informations contained in the url (article) """
    def extract_row_features(row):
        text = str(row['article'])
        if text == 'nan': text = ""

        text = html.unescape(text)
        text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\\', '')
        text = text.replace('Ã¢Â€Â—', ' — ').replace('Ã¢Â€Â™', "'").replace('Ã¢Â€Âœ', '"').replace('Ã¢Â€Â', '"')

        def clean_alt(t):
            if not t: return ""
            return re.sub(r'\s*\([^)]*\)[^)]*$', '', t).strip() # cleaning ('source') at the end of the string 
            
        # 'image text' from url
        alt_texts = re.findall(r'alt=["\']([^"\']+)["\']', text, re.IGNORECASE)        
        image_text = " ".join([clean_alt(a) for a in alt_texts])       
        
        # keywords extraction and subdivision in categories
        feed_keywords = []
        text_lower = text.lower()
        if any(x in text_lower for x in ['rss', 'feed', 'feedburner']):
            mapping = {
                'business': 'cat_business', 
                'economy': 'cat_business', 
                'sports': 'cat_sports',
                'tech': 'cat_tech', 
                'technology': 'cat_tech',
                'science': 'cat_tech',
                'internet': 'cat_tech',
                'entertainment': 'cat_entertainment',
                'movies': 'cat_entertainment',
                'health': 'cat_health', 
                'world': 'cat_international',
                'nation': 'cat_international', 
                'elections': 'cat_international', 
                'us': 'cat_international', 
                'europe': 'cat_international',
                'africa': 'cat_international', 
                'asia': 'cat_international', 
                'politics': 'cat_international',            
                'topstories': 'cat_general',
                'cnn_topstories': 'cat_general'
                }
            for key, val in mapping.items():
                if key in text_lower:
                    feed_keywords.append(val)
        keyword_str = " ".join(sorted(list(set(feed_keywords))))

        # domains extraction
        urls = re.findall(r'https?://(?:www\.|rss\.|feeds\.|us\.|d\.|ad\.)?([^/.]+)\.[^/\s]+/?([\w/-]+)?', text)
        
        url_context_words = []
        for domain_name, path in urls:
            if len(domain_name) > 1:
                url_context_words.append(domain_name)
            
            if path:
                clean_path = re.sub(r'[/_-]', ' ', path)
                clean_path = re.sub(r'\.(?:html|php|htm|aspx)$|\d{5,}', '', clean_path)
                url_context_words.append(clean_path)
        
        url_info = " ".join(url_context_words)
        
        clean_art = re.sub(r'<[^>]*>', ' ', text) # remove tags
        clean_art = re.sub(r'https?://\S+', ' ', clean_art) # remove urls
        clean_art = re.sub(r'&\w+;', ' ', clean_art) # remove noisy characters

        article_cleaned = f"{clean_art}{url_info}"
        article_cleaned = re.sub(r'\s+', ' ', article_cleaned).strip()

        return pd.Series([article_cleaned, image_text, keyword_str])

    df[['article_cleaned', 'image', 'keywords']] = df.apply(extract_row_features, axis=1)
    return df


def clean_source(text):
    if not isinstance(text, str):
        return text
    
    text = text.lower().strip()
    text = re.sub(r'\.com|\.net|\.org|\.co\.uk|\.ca|\.au|\.mw|\.info', '', text) # remove .com, .net, .org, .co.uk, ...
    
    mapping = {
        'bcc': 'bbc', 'cbbc': 'bbc', 'abs': 'abc', 'xinhuanet': 'xinhua',
        'al-jazeera': 'al jazeera', 'aljazeera': 'al jazeera',
        'macworld': 'macworld','pcworld': 'pc world','e! online': 'e!',
        'CP': 'Canadian Press','SPACE.com / LiveScience.com': 'SPACE.com',
        'newratings': 'new ratings','businessweek online': 'businessweek',
        'the motley fool': 'motley'}

    for key, value in mapping.items():
        if key in text:
            return value.title()            
    return text.title() 

# --- DATA LOADING & CLEANING ---
df = pd.read_csv('development.csv')
df_eval = pd.read_csv('evaluation.csv')

df['title'] = df['title'].fillna('').apply(clean_text)
df['source'] = df['source'].fillna('unknown').apply(clean_text)
df = df[df['article'] != 'To the Editor:.']

# source extraction from 'title' and noise removal
pattern_title = r"\s*\(([^()]*)\)\s*$" 
df['title'] = df['title'].str.replace(r'\s*\(?update\s*\d*\)?\s*$', '', regex=True, flags=re.IGNORECASE)
df["source_title"] = df["title"].str.extract(pattern_title, expand=False)
df["title_cleaned"] = df["title"].str.replace(pattern_title, "", regex=True).str.strip()

title_prefixes = ('editorial:','editorial notebook:', 'editorial |', '[editorial]', 'editor:',
                  'opinion:', 'commentary:', 'analysis:', 'news analysis:','update 1:', 'update 1-',
                  'update 2:', 'update 2-','update 3:', 'update 3-','update 4:', 'update 4-',
                  'update 5:', 'update 5-','update 6:', 'update 6-','update 7:', 'update 7-',
                  'update 8:', 'update 8-','update 9:', 'update 9-','update 10:', 'update 10-',
                  'update:','corrected:', 'correction:', 'corrected -')
df['title_cleaned'] = df['title_cleaned'].apply(lambda x: next((x[len(p):].lstrip() for p in title_prefixes if x.lower().startswith(p)), x))

df['article'] = df['article'].astype(str).str.replace(r'\N', '', regex=False)
df = url_preprocess(df)

df['full_text'] = df['title_cleaned'] + " " + df['article_cleaned'] 
df['full_text'] = df['full_text'].str.lower()



df_eval['title'] = df_eval['title'].fillna('').apply(clean_text)
df_eval['source'] = df_eval['source'].fillna('unknown').apply(clean_text)

# source extraction from 'title' and noise removal
df_eval['title'] = df_eval['title'].str.replace(r'\s*\(?update\s*\d*\)?\s*$', '', regex=True, flags=re.IGNORECASE)
df_eval["source_title"] = df_eval["title"].str.extract(pattern_title, expand=False)
df_eval["title_cleaned"] = df_eval["title"].str.replace(pattern_title, "", regex=True).str.strip()
df_eval['title_cleaned'] = df_eval['title_cleaned'].apply(lambda x: next((x[len(p):].lstrip() for p in title_prefixes if x.lower().startswith(p)), x))

df_eval['article'] = df_eval['article'].astype(str).str.replace(r'\N', '', regex=False)
df_eval = url_preprocess(df_eval)

df_eval['full_text'] = df_eval['title_cleaned'] + " " + df_eval['article_cleaned'] 
df_eval['full_text'] = df_eval['full_text'].str.lower()


# ---DUPLICATE HANDLING ---
# Logic: If the same article appears multiple times with different labels,
# we pick the most frequent label (majority vote) and drop ties.
counts = df.groupby(['title', 'article', 'label']).size().reset_index(name='vote_count')
counts['max_votes'] = counts.groupby(['title', 'article'])['vote_count'].transform('max')
ties = counts[counts['vote_count'] == counts['max_votes']].groupby(['title', 'article'])['label'].transform('count') > 1
counts['is_tie'] = ties.astype(bool)

winners = counts[(counts['vote_count'] == counts['max_votes']) & (counts['is_tie'] == False)] 
df_final = df.merge(winners[['title', 'article', 'label']], on=['title', 'article', 'label'], how='inner')

df_final = df_final.drop_duplicates(subset=['title', 'article'])

df_final = df_final[df_final['full_text'].str.len() > 50].copy()

# source cleaning
df_final['source'] = df_final['source'].apply(clean_source)
df_final["source_title"] = df_final["source_title"].replace("", pd.NA)

mask_changing_source = (
    df_final['source'].notna() & 
    ~df_final['source'].astype(str).str.lower().isin(['nan', 'n', '', 'none', 'unknown']) &
    df_final['source_title'].notna() & 
    ~df_final['source_title'].astype(str).str.lower().isin(['nan', 'n', '', 'none', 'unknown']))

df_final['source_changed'] = (
    mask_changing_source & 
    (df_final['source'].astype(str).str.lower().str.strip() != 
     df_final['source_title'].astype(str).str.lower().str.strip())).astype(int)

# remove infrequent sources
final_freq = df_final["source"].value_counts()
df_final.loc[df_final["source"].map(final_freq) < 6, "source"] = "Other"

final_freq = df_eval["source"].value_counts()
df_eval.loc[df_eval["source"].map(final_freq) < 6, "source"] = "Other"
#-------------
df_eval['source'] = df_eval['source'].apply(clean_source)
df_eval["source_title"] = df_eval["source_title"].replace("", pd.NA)

mask_changing_source = (
    df_eval['source'].notna() & 
    ~df_eval['source'].astype(str).str.lower().isin(['nan', 'n', '', 'none', 'unknown']) &
    df_eval['source_title'].notna() & 
    ~df_eval['source_title'].astype(str).str.lower().isin(['nan', 'n', '', 'none', 'unknown']))

df_eval['source_changed'] = (
    mask_changing_source & 
    (df_eval['source'].astype(str).str.lower().str.strip() != 
     df_eval['source_title'].astype(str).str.lower().str.strip())).astype(int)

# ---FEATURE ENGINEERING ---
def build_advanced_features(df):
    """
    - Temporal: hour, weekday, day_period (morning/night).
    - Stylistic: text length, digit density.
    """
    df = df.copy()
    
    # 'Time' features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['hour'].fillna(-1).astype(int)
    df['weekday'] = df['weekday'].fillna(-1).astype(int)

    dt = pd.to_datetime(df['timestamp'], errors='coerce')
    hour = dt.dt.hour.fillna(12)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    def get_period(h):
        if h == -1: return '-1' #unknown
        if 5 <= h < 12: return '1' #morning
        if 12 <= h < 18: return '2' #afternoon
        if 18 <= h < 22: return '3' #evening
        return '4' #night

    df['day_period'] = df['hour'].apply(get_period)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 'Style' features
    word_list = df['full_text'].str.split()
    word_count = word_list.str.len().fillna(0)
    df['art_len_word'] = word_count
    df['title_art_ratio'] = df['title_cleaned'].str.split().str.len() / (word_count + 1) 

    char_len = df['full_text'].str.len() + 1
    df['digit_density'] = df['full_text'].apply(lambda x: len(re.findall(r'\d', x))) / char_len
    
    return df

numeric_features = ['page_rank', 'art_len_word', 'title_art_ratio', 'digit_density']
numerical_features_noENC = ['is_weekend', 'source_changed','hour_sin','hour_cos']

df_final = build_advanced_features(df_final)
df_eval = build_advanced_features(df_eval)

custom_stops = [
    'said', 'new', 'year', 'years', 'says', 'monday', 'tuesday', 
    'wednesday', 'thursday', 'friday', 'sunday', 'saturday', 
    'world', 'time', 'according', 'week', 'just', 'day',
    'told', 'reported', 'added', 'tells', 'people']
my_stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_stops))

# --- MODEL PIPELINE ---
# We use a ColumnTransformer to apply different scaling/vectorization to different types of data columns.
preprocessor = ColumnTransformer([
    ("word", TfidfVectorizer(ngram_range=(1,2), min_df = 2, max_features=50000,sublinear_tf = True, stop_words=my_stop_words), "full_text"),
    ("img", TfidfVectorizer(ngram_range=(1,2), sublinear_tf = True, stop_words=my_stop_words), "image"),  
    ("key", TfidfVectorizer(ngram_range=(1,2), sublinear_tf = True, stop_words=my_stop_words), "keywords"),
    ("source_period", OneHotEncoder(handle_unknown="ignore"), ["source", "day_period"]),                                                              
    ("rank", StandardScaler(), numeric_features),
    ("pass", "passthrough", numerical_features_noENC)
])

# model training
X_train = df_final[['full_text', 'source','keywords','image','day_period'] + numeric_features + numerical_features_noENC]
y_train = df_final['label']

X_eval = df_eval[['full_text', 'source','keywords','image','day_period'] + numeric_features + numerical_features_noENC]

X_train_final = preprocessor.fit_transform(X_train)
X_eval_final = preprocessor.transform(X_eval)

# ---TRAINING & ENSEMBLING ---

#LinearSVC
model_linSVC =  LinearSVC(C=0.09, class_weight='balanced', loss = 'squared_hinge', penalty = 'l2', max_iter=1000, random_state=42, dual=False)
model_linSVC.fit(X_train_final, y_train)
y_pred_linSVC = model_linSVC.predict(X_eval_final)
print('fine svc')

submission_linSVC = pd.DataFrame({'Id': df_eval['Id'], 'Predicted': y_pred_linSVC})
submission_linSVC = submission_linSVC.sort_values(by='Id', ascending=True).reset_index(drop=True)
submission_linSVC.to_csv('submission_linSVC.csv', index=False)
print('fine svc')

'''
other selected classifiers output

# RandomForestClassifier
model_rf =  RandomForestClassifier()
model_rf.fit(X_train_final, y_train)
y_pred_rf = model_rf.predict(X_eval_final)
print('fine rf')

submission_rf = pd.DataFrame({'Id': df_eval['Id'], 'Predicted': y_pred_rf})
submission_rf = submission_rf.sort_values(by='Id', ascending=True).reset_index(drop=True)
submission_rf.to_csv('submission_rf.csv', index=False)
print('fine rf')

# LogisticRegression
model_lr =  LogisticRegression(C = 1.15, class_weight='balanced', max_iter = 1000, solver = 'lbfgs', penalty = 'l2', random_state=42)
model_lr.fit(X_train_final, y_train)
y_pred_lr = model_lr.predict(X_eval_final)
print('fine lr')

submission_lr = pd.DataFrame({'Id': df_eval['Id'], 'Predicted': y_pred_lr})
submission_lr = submission_lr.sort_values(by='Id', ascending=True).reset_index(drop=True)
submission_lr.to_csv('submission_lr.csv', index=False)
print('fine lr')
'''
