# News Articles Classification Pipeline

This repository contains a machine learning pipeline and a detailed theoretical report for the multi-class categorization of online news articles. The implementation leverages Natural Language Processing (TF-IDF) combined with advanced metadata feature engineering to effectively classify articles into seven thematic categories using optimized linear models (LinearSVC and Logistic Regression). 

---

## Algorithm Overview

* 1. **Text Cleaning & URL Parsing:** 
    The raw text is normalized by removing HTML artifacts, correcting encodings, and handling noisy characters. Embedded URLs are parsed to extract hidden contextual information, such as image 'alt' texts, domain names, and path-level keywords.

* 2. **Feature Engineering:** 
    The dataset is enriched with temporal and stylistic features. Publication times are converted into cyclic representations (sine/cosine) and day periods, while stylistic metrics like digit density, word counts, and title-to-article length ratios are computed.

* 3. **Vectorization and Encoding:**
    Heterogeneous data is processed using a ColumnTransformer. Textual components are independently vectorized using TF-IDF with custom domain-specific stopping words and sublinear term-frequency scaling. Categorical variables are one-hot encoded, and numerical features are standardized.

4. **Classification:** 
    High-dimensional sparse feature matrices are fed into classifiers optimized for text data, such as LinearSVC and Logistic Regression. Class imbalances are handled via balanced class weights, and models output predictions for the evaluation dataset.

--- 

## Results

The classification pipeline was evaluated using the Macro F1-score metric to ensure equal importance across all classes, given the imbalanced nature of the dataset. The linear models demonstrated superior performance for this high-dimensional text data, significantly outperforming the tree-based baseline. 

The main results are as follows:
* **LinearSVC:** Achieved the highest performance with an F1-macro score of 0.7292 on the validation set and 0.728 on the leaderboard.
* **Logistic Regression:** Achieved an F1-macro score of 0.7241 on the validation set and 0.721 on the leaderboard.
* **RandomForestClassifier (Baseline):** Achieved an F1-macro score of 0.6236 on the validation set.

Error analysis through the confusion matrix revealed that the most significant classification challenge was distinguishing between "General News" and "International News" due to thematic ambiguity and a shared political lexicon. Conversely, "Sports" and "Technology" demonstrated high classification accuracy thanks to their domain-specific nomenclature.

---

## Authors

* [Arianna Miori](https://github.com/ariannamiori)
* [Giulia De Santis](https://github.com/giuliadesantis)