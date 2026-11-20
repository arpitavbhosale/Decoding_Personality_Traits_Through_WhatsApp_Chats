Decoding Personality Traits Through WhatsApp Chats

The project is based on the Big Five Personality Trait Model (OCEAN), which categorizes 
personality into Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. A 
dataset of WhatsApp conversations is collected and preprocessed, including steps like 
tokenization, stopword removal, stemming, and lemmatization. Feature extraction methods 
include TF-IDF vectorization, sentiment analysis using VADER, emoji frequency analysis, 
response time measurement, and active hours tracking to derive meaningful insights from chat 
behavior.
A Random Forest classifier is trained using the extracted features tested for comparison. The 
model's performance is evaluated based on accuracy, precision-recall metrics, and F1-score, with 
Random Forest achieving the highest accuracy of 98.69%

Implementation:
![Screenshot 2025-05-28 230204](https://github.com/user-attachments/assets/da22b9ae-fef2-475d-abf7-ed0dc3aebff4)

![Screenshot 2025-05-28 230341](https://github.com/user-attachments/assets/6ca54af6-723d-40a9-95a6-25afca0b8f2b)

![Screenshot 2025-05-28 230425](https://github.com/user-attachments/assets/b204639a-8837-4e4d-ac8c-18865a9e38e3)


![Screenshot 2025-05-28 231434](https://github.com/user-attachments/assets/a61bc033-d850-4245-ae75-bd8410156186)

# Decoding Personality Traits Through WhatsApp Chats

An end-to-end NLP and machine learning mini-project that parses an exported WhatsApp group chat, extracts sentiment and behavioral features for every participant, estimates **Big Five (OCEAN)** personality scores, and trains a **Random Forest** classifier to predict personality categories from message text and metadata.

## Features

- **Robust chat parser:** regex-based timestamp detection with support for multiline messages and system messages
- **Sentiment analysis:** message-level positive, negative, neutral and compound scores using VADER
- **Chat statistics:** total and per-user messages, words, media, links and deleted messages
- **Big Five estimation:** per-user Openness, Conscientiousness, Extraversion, Agreeableness and Neuroticism scores
- **Personality classification:** Random Forest trained on TF-IDF and engineered features
- **Model evaluation:** accuracy, classification report, 5-fold cross-validation, confusion matrix and multi-class ROC-AUC
- **Rich visual analytics:** radar charts, activity timelines, word clouds and sentiment rankings

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data handling | Pandas, NumPy, Regex |
| NLP | NLTK (VADER, stopwords), TF-IDF, `emoji` |
| Machine learning | scikit-learn (Random Forest, One-vs-Rest, metrics) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Environment | Jupyter Notebook |

---

## Project Workflow

```
WhatsApp .txt export
        │
        ▼
Regex parsing (date, time, author, message; multiline handling)
        │
        ▼
Structured DataFrame
        │
        ├──► VADER sentiment analysis
        ├──► Chat statistics (overall and per user)
        ├──► Feature engineering (length, emojis, links, media, hour)
        │
        ▼
Big Five estimation (per user)      Heuristic personality labels (per message)
        │                                         │
        ▼                                         ▼
Radar / bar charts                  TF-IDF + features → Random Forest
                                                  │
                                                  ▼
                              Accuracy, CV, Confusion Matrix, ROC-AUC
```

### 1. Data Parsing
- Detects message starts with a regex such as `dd/mm/yy, hh:mm AM/PM -`
- Splits each line into **date, time, author and message**
- Lines without a timestamp are appended to the previous message (multiline support)
- Lines without an author are treated as system messages

### 2. Sentiment Analysis
VADER (`SentimentIntensityAnalyzer`) scores each message. Scores are averaged per user to find the most positive, neutral and negative contributors.

### 3. Chat Statistics
Total messages, words, media shared, links shared and deleted messages, plus a per-user breakdown (message count, average words per message, media and links).

### 4. Feature Engineering
`Word_Count`, `Emoji_Count`, `Message_Length`, `Sentiment_Score` (compound) and `Hour` of the day.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy matplotlib seaborn nltk emoji wordcloud scikit-learn pillow jupyter
```

Download the required NLTK data:

```python
import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")
```

### Export your WhatsApp chat

1. Open the chat in WhatsApp
2. Tap **⋮ / More → Export chat**
3. Choose **Without media**
4. Save the `.txt` file in the project folder

### Run

1. Open `SEM_VI_ML_MINI_PROJECT.ipynb`
2. Update the file path in the notebook to your own chat file:
   ```python
   conversation = r"path/to/your_chat.txt"
   ```
3. Run all cells in order

> **Date format:** the parser expects the `dd/mm/yy, hh:mm AM/PM -` style used by Android exports. If your export looks different (for example iOS or another locale), adjust the regex in `date_time()` and the `dayfirst` argument in `pd.to_datetime`.

---

## Big Five Mapping

Each trait is estimated from behavioral features, then min-max normalized to a 0 to 1 scale across users.

| Trait | Estimation logic | Rationale |
|---|---|---|
| **Openness** | Average message length + 5 × positive sentiment | Expressive, longer messages suggest curiosity |
| **Conscientiousness** | Links sent + 2 × media sent | Sharing resources suggests structured communication |
| **Extraversion** | Total messages + emoji count | High activity and expressiveness suggest sociability |
| **Agreeableness** | 10 × positive sentiment | Positive tone suggests friendliness |
| **Neuroticism** | 10 × negative sentiment | Negative tone suggests emotional instability |

Scores are **relative to the group**: a score of 1.0 means the highest in this chat, not high in absolute terms.

---

## Machine Learning Model

| Component | Details |
|---|---|
| Text features | TF-IDF (max 500 features) |
| Numeric features | Word count, emoji count, message length, sentiment score, hour |
| Labels | Heuristic rules over sentiment, word count, emoji count and message length |
| Classifier | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| Split | 80% train / 20% test |
| Evaluation | Accuracy, precision/recall/F1, 5-fold CV, confusion matrix, One-vs-Rest ROC-AUC |

---

## Visualizations

- Sentiment distribution (positive, negative, neutral)
- Top 10 most active users
- Most active day, month and year
- Chat activity timeline
- Word cloud and top 10 most used words
- Most used word per user
- Top 10 most positive, neutral and negative users
- Big Five bar charts and radar charts
- Confusion matrix and multi-class ROC curves

<!-- Add screenshots here, for example:
![Radar Chart](images/radar_chart.png)
-->


## Limitations

- **Heuristic labels:** the classifier is trained on labels created from the same features it receives, so high accuracy partly reflects re-learning the rules rather than true personality prediction.
- **No ground truth:** scores are not validated against questionnaires such as the BFI or IPIP-NEO.
- **Relative scoring:** min-max normalization makes trait scores comparable only within one chat.
- **Class imbalance:** the default fallback label can dominate, so macro-F1 is more informative than accuracy.
- **Language coverage:** VADER is English-only and struggles with Hinglish, Marathi, slang and sarcasm.
- **Limited data:** results come from a single group chat and may not generalize.


