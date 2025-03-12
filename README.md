# Sentiment Analysis Project

This project performs sentiment analysis using NLP techniques, Word2Vec for feature extraction, and MLflow for model deployment.

## Prerequisites

Ensure you have the following installed:

- Python 3.10 or above
- Conda (for managing virtual environments)
- MLflow
- TensorFlow
- NLTK
- Gensim

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Create and Activate the Virtual Environment

```bash
conda create --name sent_analysis python=3.10
conda activate sent_analysis
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
### 4. Perform EDA on the dataset

Run blocks from EDA.ipynb file to visualize and understand the data for further processing. 

### 4. Download NLTK Data

Inside a Python shell, run:

```python
import nltk
nltk.download('punkt')
```

### 5. Train the Model

Run the training script to train the sentiment analysis model:

```bash
python src/train.py
```

### 6. Save Model with MLflow

After training, register the model with MLflow:

```bash
mlflow models serve -m models/sentiment_model -p 1234 --no-conda
```

### 7. Test the Deployed Model

Run the test script to send sample text data to the REST API endpoint:

```bash
python src/server_test.py
```

### 8. Example Payload for Testing (if using `curl`)

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"instances": [[0.2, 0.1, ... , 0.5]]}' \
  http://localhost:1234/invocations
```

## Expected Output

```
{'predictions': [4, 0]}  # Example output for Positive and Negative Sentiments
```

## Troubleshooting

- If `pyenv` errors appear, ensure `pyenv` is properly installed and environment variables like `PYENV`, `PYENV_ROOT`, and `PYENV_HOME` are set.
- Ensure your test data is reshaped correctly (2D array format) before sending it to the model.

## Acknowledgments

- This project leverages Word2Vec for feature extraction and MLflow for deployment.

For further questions or issues, feel free to reach out!

