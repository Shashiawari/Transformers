# Tweets Sarcasm Detector
This project focuses on detecting sarcasm in tweets using deep learning techniques. It employs advanced models such as LSTM, Transformers, and CNNs to accurately identify sarcastic content in social media text data.
Key Features
Data Preprocessing: Cleaning and preprocessing text data to remove noise and standardize the input for better model performance.
Modeling Techniques: Utilization of different deep learning models including:
LSTM (Long Short-Term Memory): For capturing long-term dependencies in the text.
Transformers: Specifically using the Roberta model for its efficiency in understanding contextual information.
CNN (Convolutional Neural Network): For feature extraction and pattern recognition in text.
Training and Evaluation:
Splitting the dataset into training and testing sets.
Training the model on the training set and evaluating its performance on the test set.
Performance Metrics: Measuring accuracy and generating classification reports to assess model performance.
Dataset
The dataset used in this project consists of tweets labeled as sarcastic or non-sarcastic. It is preprocessed to remove noise and standardize the text before being fed into the models.

Dependencies
simpletransformers
pandas
scikit-learn
torch
transformers
Installation
bash
Copy code
```
pip install simpletransformers pandas scikit-learn torch transformers
```
Usage
Preprocess the Data:

python
```
df['tweets'] = df['tweets'].apply(preprocess_text)
df = df.dropna()
Train the Model:
```
python
```
from simpletransformers.classification import ClassificationModel
model = ClassificationModel('roberta', 'roberta-base', num_labels=num_unique_labels, args={'reprocess_input_data': True, 'overwrite_output_dir': True}, use_cuda=False)
model.train_model(train_df)
Evaluate the Model:
```
python
```
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(all_true_labels, all_predictions))
```
