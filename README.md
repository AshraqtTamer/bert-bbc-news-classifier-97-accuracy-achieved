# README: Multi-Class News Classification with BERT

This project demonstrates the fine-tuning of a **BERT (Bidirectional Encoder Representations from Transformers)** model to classify BBC news articles into five distinct categories. The model achieves high accuracy by leveraging pre-trained transformer embeddings and adapting them to the specific nuances of BBC journalistic content.

---

## 🚀 Project Overview

The notebook implements a full machine learning pipeline, from raw data ingestion to real-time inference. It utilizes the `bert-base-uncased` architecture to perform multi-class classification on text data.

### Key Features

* **Dataset**: BBC news articles categorized into Business, Entertainment, Politics, Sport, and Tech.
* **Model**: Fine-tuned `BertForSequenceClassification`.
* **Performance**: Achieved **~97.8% validation accuracy** within 3 training epochs.
* **Frameworks**: Built using `Transformers`, `Datasets`, `Evaluate`, and `PyTorch`.

---

## 🛠️ Technical Workflow

### 1. Data Preparation

* **Loading**: Reads BBC news data from a CSV format.
* **Label Encoding**: Converts text labels (e.g., "business") into numerical IDs for the model.
* **Splitting**: The dataset is divided into training (80%) and testing (20%) sets to ensure robust evaluation.

### 2. Tokenization

The text is processed using the `BertTokenizer` with the following configurations:

* **Max Length**: 256 tokens.
* **Padding**: "max_length" to ensure uniform input size.
* **Truncation**: Enabled to handle articles exceeding the token limit.

### 3. Training Configuration

The model is trained using the Hugging Face `Trainer` API with these hyperparameters:

* **Learning Rate**: 
* **Batch Size**: 16 per device.
* **Weight Decay**: 0.01.
* **Epochs**: 3.

### 4. Evaluation Metrics

The primary metric used is **Accuracy**. The training logs indicate a significant drop in loss and a steady increase in accuracy across epochs:
| Epoch | Training Loss | Validation Loss | Accuracy |
| :--- | :--- | :--- | :--- |
| 1 | 0.4300 | 0.3037 | 97.18% |
| 2 | 0.1172 | 0.1142 | 97.88% |
| 3 | 0.0604 | 0.0944 | 97.88% |

---

## 💻 How to Use

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install pandas torch datasets transformers evaluate pyarrow

```

### Running Inference

You can test the model with custom text as shown in the notebook:

```python
text = "The team won the championship after a spectacular goal in the final minute!"
# Result: Predicted Category: sport

```

---

## 📂 Dataset Structure

The model expects a CSV with at least two columns:

* `text`: The raw news article content.
* `labels`: The category of the news.

Would you like me to help you generate a script to export this fine-tuned model for deployment?
