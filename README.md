# 📚 Book Recommender System

A powerful **Book Recommendation System** powered by **NLP techniques**, **LLMs**, **Zero-Shot Classification**, **Sentiment Analysis**, and **Vector Search**. It uses cleaned book datasets, enriched with categories and emotional analysis, and integrates modern ML frameworks like **LangChain** and **Hugging Face models**.

---

## 🚀 Features

- Cleaned, preprocessed book dataset ready for machine learning applications.
- Vector database search using **LangChain** & **OpenAI Embeddings**.
- Emotion classification using fine-tuned language models.
- Zero-shot classification for better book category grouping.
- Gradio dashboard for visualizing book recommendations.
  
---

## 📂 Dataset

- Download the latest version from Kaggle:

```python
path = kagglehub.dataset_download("vivekprasadkushwaha/books-dataset")
```

**Dataset Cleaning & Preprocessing:**
1. **Initial Exploration:**
   - Use Google Colab’s recommended plots or view dataset on Kaggle.
2. **Data Quality Checks:**
   - Remove duplicate rows.
   - Handle missing values (threshold: remove columns/rows with >50% missing).
   - Remove unnecessary categories (reduce the number of distinct categories for simplicity).
3. **Specific Column Checks:**
   - **Description:** Few missing values—check if missingness is random.
   - **Subtitle:** Large number of nulls, hence dropped later.
   - **Average Rating, Number of Pages, Rating Count:** Missing values show patterns (likely due to dataset merging).
4. **Bias Detection:**
   - Add indicator columns (1 if missing, 0 if not).
   - Analyze correlations using **Spearman correlation matrix**.
   - Confirmed no strong correlation (checked via heatmap).
5. **Category Analysis:**
   - Visualize category distribution (long tail problem detected—too many categories with few entries).
6. **Description Word Count:**
   - Focus on descriptions with more than **25 words** (sufficient for recommendation).

---

## 🛠️ Data Refinement:

- Subtitle dropped.
- Title and Subtitle merged (to retain info).
- ISBN13 added as identifier.
- Dropped unnecessary columns.
- Cleaned dataset saved to `.csv`.

---

## 🧠 Vector Search Using LangChain & LLMs

**Goal:** Recommend books based on query descriptions using vector embeddings.

### Steps:
1. **Tools Used:**
   - `langchain_community.document_loaders.TextLoader`
   - `langchain_text_splitters.CharacterTextSplitter`
   - `langchain_openai.OpenAIEmbeddings`
   - `langchain_chroma.Chroma` (Vector Database)
2. **Workflow:**
   - Convert book descriptions to document format.
   - Split into meaningful chunks (one per book description).
   - Convert descriptions into vector embeddings.
   - Store vectors in vector DB.
3. **Querying:**
   - Convert user query into vector embedding.
   - Find similarity using **cosine similarity**.
   - Use **ISBN13** to map vector results back to book titles/authors efficiently.
4. **Important Parameters:**
   - `chunk_size`, `chunk_overlap`, `separator` handled carefully.
5. **Visualization:**
   - Integrated via Gradio.

---

## 🔖 Text Classification - Simplifying Categories

**Objective:**  
Reduce and sort diverse book categories using **Zero-Shot Classification**.

**Model Used:**  
`facebook/bart-large-mnli` via **Hugging Face Transformers**

- Predicts if a description belongs to categories like Fiction, Non-Fiction, etc.
- **Accuracy:** ~78% correct on pre-labeled descriptions.

---

## 🎭 Sentiment Analysis

**Goal:**  
Detect **emotions** embedded in book descriptions using a fine-tuned emotion classification model.

- **Fine-Tuned LLM** specialized for sentiment analysis.
- Each description classified into 7 emotion classes (e.g., Joy, Sadness, Fear, etc.).
- Run sentiment analysis **line by line** in descriptions to capture subtle emotions.
- For each emotion class, highest probability score across description is taken.

---

## 🎨 Gradio Dashboard

- Simple, customizable interface to visualize:
  - Book recommendations based on query.
  - Sentiment distributions.
  - Category filtering.
  
---

## 📚 Tech Stack

- **Python, Pandas, Matplotlib, Seaborn**
- **LangChain**
- **OpenAI Embeddings**
- **Chroma Vector Database**
- **Hugging Face Transformers**
- **Gradio**
- **TQDM**

---

## 📊 Future Enhancements

- Integrate **web scraping** to fill missing data.
- Fine-tune classification models further for higher accuracy.
- Add advanced filters like genre, author popularity, etc.
- Deploy full system as a web app.

---

## 🔗 Links

- Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/vivekprasadkushwaha/books-dataset)
- Demo: _Coming Soon_

---

**Happy Reading & Recommending! 📖✨**