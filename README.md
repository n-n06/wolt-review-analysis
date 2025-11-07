# Google Play Reviews Sentiment Analysis using BERT 🛵

### Overview 📎

This project trains a BERT-based model to classify Google Play Store reviews as **positive**, **neutral**, or **negative**. It uses the [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) library to collect app reviews directly from Google Play and applies natural language processing (NLP) techniques using **PyTorch** and **BERT**.

### Features ⚙️

* Automatically scrape reviews from any Google Play app.
* Clean and preprocess text data.
* Tokenize and encode reviews for BERT.
* Fine-tune a pretrained BERT model on labeled sentiment data.
* Evaluate the model using accuracy, precision, recall, and F1-score.
* Predict sentiment on new/unseen reviews.

---

### Requirements 📚

Install dependencies:

```bash
pip install google-play-scraper pytorch-pretrained-bert pytorch-nlp torch numpy pandas scikit-learn matplotlib
```

Optional (for notebook visualization):

```bash
pip install jupyter seaborn
```

---

### Project Structure

```
📁 project-root
 ┣ 📜 notebook.ipynb         # Main Jupyter notebook
 ┗ 📜 README.md              # Project documentation
```

---

### Usage 💻

1. **Open the notebook:**

   ```bash
   jupyter notebook notebook.ipynb
   ```

2. **Collect Reviews**

   * Use `google-play-scraper` to fetch reviews for a given app package name:

     ```python
     from google_play_scraper import reviews_all
     data = reviews_all('com.example.app')
     ```

3. **Preprocess Data**

   * Clean text, remove stopwords, and label sentiments (e.g., positive, neutral, negative).

4. **Fine-tune BERT**

   * Load the pretrained model:

     ```python
     from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
     ```
   * Tokenize and train on your dataset.

5. **Evaluate and Predict**

   * Check model performance.
   * Use the trained model to classify new reviews.



### Example output 📊

| Review                           | Predicted Sentiment |
| :------------------------------- | :-----------------: |
| “Great app, love the features!”  |     😊 Positive     |
| “It crashes sometimes”           |      😐 Neutral     |
| “Terrible update, full of bugs.” |     😡 Negative     |



### Model performance 💹

After training for several epochs, the model typically achieves:

* **Accuracy:** ~85–90%
* **F1-score:** ~0.88

(Results vary by dataset size and quality.)



### Future improvements 👨‍🔬

* Use multilingual BERT for non-English reviews.
* Implement data augmentation for underrepresented sentiments.
* Deploy as a REST API with FastAPI for real-time sentiment prediction.

