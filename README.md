# **Fake News Detection Project**

## **Project Overview**
This project focuses on building a machine learning-based system to detect and classify news articles as either **fake** or **real**. By leveraging data analytics and machine learning techniques, the system aims to address the growing problem of misinformation and fake news in the digital era.

---

## **Features**
- Preprocessing and cleaning of raw textual data.
- Exploratory Data Analysis (EDA) to gain insights into the dataset.
- Feature extraction using techniques like TF-IDF or Word Embeddings.
- Training and evaluation of machine learning models for classification.
- Performance metrics visualization, including accuracy, precision, recall, and F1 score.

---

## **Tech Stack**
- **Programming Language:** Python
- **Libraries/Tools:**  
  - Data Handling: `pandas`, `numpy`
  - Data Visualization: `matplotlib`, `seaborn`
  - Text Processing: `nltk`, `re`
  - Machine Learning: `scikit-learn`, `XGBoost`
  - Optional Deployment: `Streamlit`, `Flask`

---

## **Dataset**
The dataset used in this project is publicly available and was downloaded from Kaggle. It contains labeled news articles categorized as **fake** or **real**.  

**Path to Dataset:** `/root/.cache/kagglehub/datasets/algord/fake-news/versions/1`  

If using a different dataset, ensure it contains the necessary fields, such as `text`, `title`, and `label`.

---

## **How to Run the Project**

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies
Create a virtual environment (optional) and install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
- Download the dataset from Kaggle or use the one included in the specified directory.

### 4. Run the Code
- **Exploratory Data Analysis (EDA):**
  ```bash
  python eda.py
  ```
- **Model Training and Evaluation:**
  ```bash
  python model_training.py
  ```
- **Optional Deployment:**
  Launch a web app for predictions:
  ```bash
  streamlit run app.py
  ```

---

## **Project Workflow**
1. **Data Preprocessing:**
   - Clean and preprocess raw text data by removing punctuation, stopwords, and special characters.
2. **EDA:**
   - Understand the dataset through visualizations and summary statistics.
3. **Feature Engineering:**
   - Convert text to numerical features using TF-IDF or embeddings.
4. **Modeling:**
   - Train baseline and advanced machine learning models for classification.
5. **Evaluation:**
   - Evaluate models using metrics like accuracy, precision, recall, and F1 score.
6. **Deployment (Optional):**
   - Build a simple user interface to classify news articles.

---

## **Directory Structure**
```
fake-news-detection/
├── data/
│   └── fake_news.csv              # Dataset file
├── notebooks/
│   └── eda.ipynb                  # Jupyter Notebook for EDA
├── src/
│   ├── preprocess.py              # Script for data preprocessing
│   ├── feature_engineering.py     # Script for feature extraction
│   └── model_training.py          # Script for model training
├── app/
│   └── app.py                     # Streamlit or Flask app for deployment
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
```

---

## **Results**
- **Best Model:** Logistic Regression with TF-IDF features.
- **Performance:**
  - Accuracy: 92%
  - Precision: 90%
  - Recall: 88%
  - F1 Score: 89%

---

## **Future Improvements**
- Use deep learning models like LSTM or BERT for better text understanding.
- Explore ensemble methods to improve model accuracy.
- Implement real-time fake news detection in a deployed system.

---

## **Author**
- **Name:** Sahil Sharma  
- **Contact:** sahilsharmamrp@gmail.com  
- **GitHub:** [Your GitHub Profile](https://github.com/Developer-Sahil)

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more details.  

--- 
