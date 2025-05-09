# WhatsApp Review Sentiment Analysis

This project performs sentiment analysis on WhatsApp application reviews, classifying them as either positive or negative. The system uses both traditional machine learning algorithms and deep learning approaches for comparison.

## Project Overview

This sentiment analysis system analyzes user reviews of the WhatsApp messaging application. The project includes:

1. Data preprocessing and cleaning
2. Exploratory data analysis with visualizations
3. Feature extraction using TF-IDF
4. Implementation of multiple machine learning and deep learning models
5. Model evaluation and comparison
6. Inference capabilities for new text inputs

## Dataset

The project uses the `ulasan_aplikasi_whatsapp.csv` dataset containing:
- User reviews of the WhatsApp application
- Ratings/scores
- Other metadata

## Data Preprocessing Pipeline

The preprocessing pipeline includes:
- Text cleaning (removing mentions, hashtags, links, numbers, etc.)
- Case folding (converting text to lowercase)
- Slang word normalization (converting informal Indonesian words to formal ones)
- Tokenization
- Stopword removal
- Text vectorization using TF-IDF

## Models Implemented

### Machine Learning Models
- Support Vector Machine (SVM)
- Naive Bayes (Bernoulli)
- Random Forest
- Logistic Regression
- Decision Tree

### Deep Learning Models
- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM)
- Gated Recurrent Unit (GRU)

## Project Structure

```
.
├── README.md                              # Project documentation
├── requirements.txt                       # Required dependencies
├── scraping_data.py                       # Data collection script
├── Scraping.ipynb                         # Data collection notebook
├── submission_sentiment_analysis_inference.py  # Inference script
├── Submission_Sentiment_Analysis.ipynb    # Main analysis notebook
├── ulasan_aplikasi_whatsapp.csv           # Dataset
└── Model/                                 # Saved models directory
    ├── decision_tree_sentiment_model.pkl
    ├── logistic_regression_sentiment_model.pkl
    ├── naive_bayes_sentiment_model.pkl
    ├── random_forest_sentiment_model.pkl
    ├── svm_sentiment_model.pkl
    ├── tfidf_vectorizer.pkl
    └── DeepLearning/
        ├── ann_sentiment_model.h5
        ├── bilstm_sentiment_model.h5
        ├── cnn_sentiment_model.h5
        ├── gru_sentiment_model.h5
        └── lstm_sentiment_model.h5
```

## Features

- **Lexicon-based Sentiment Analysis**: Uses positive and negative word lexicons to determine sentiment polarity
- **Text Cleaning and Normalization**: Comprehensive preprocessing including slang word normalization
- **Multiple Model Comparison**: Implements and compares various machine learning and deep learning approaches
- **Visualization**: Includes bar charts, pie charts, and word clouds for data exploration
- **Inference API**: Ready-to-use inference code for practical applications

## How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Running the Analysis
1. Open and run `Submission_Sentiment_Analysis.ipynb` for the complete analysis
2. Use `submission_sentiment_analysis_inference.py` for making predictions on new text

### Example Usage
```python
# Load models and make prediction
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load TF-IDF and model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
svm_model = joblib.load('Model/svm_sentiment_model.pkl')

# Text input for prediction
text_input = ["Aplikasi ini sangat berguna dan memiliki fungsi yang baik"]

# Preprocessing + TF-IDF transform
X_input = vectorizer.transform(text_input)

# Predict
predicted_label = svm_model.predict(X_input)
print("Prediction:", predicted_label[0])
```

## Dependencies

Main dependencies include:
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn
- TensorFlow
- Sastrawi (for Indonesian stemming)
- WordCloud

## Results

The project achieved sentiment classification with varying accuracies across different models. Refer to the notebook for detailed accuracy metrics and comparisons between machine learning and deep learning approaches.

## Future Improvements

- Implement cross-validation for more robust model evaluation
- Explore more advanced preprocessing techniques
- Test with larger and more diverse datasets
- Optimize hyperparameters for better performance
- Deploy as a web service for real-time sentiment analysis

## Author

Muhammad Thariq