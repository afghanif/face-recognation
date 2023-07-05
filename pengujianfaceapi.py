from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np

# Baca file CSV dengan data referensi dari FERPlus dataset
df = pd.read_csv('pengujian/fer2013new.csv')

# Ambil kolom emosi yang merupakan label emosi atau ekspresi
ground_truth = df['neutral']

# print(type(ground_truth))
# ground_truth = ground_truth.astype(int)

# Hasil prediksi Face API
predicted_labels = df[['neutral', 'happiness', 'surprise', 'sadness',
                       'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']]
predicted_labels = predicted_labels.astype(str)
predicted_labels = np.argmax(predicted_labels.values, axis=1)

# print(predicted_labels.dtypes)
accuracy = accuracy_score(ground_truth, predicted_labels)
precision = precision_score(
    ground_truth, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(ground_truth, predicted_labels, average='weighted')
f1 = f1_score(ground_truth, predicted_labels, average='weighted')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)
print(classification_report(ground_truth, predicted_labels, zero_division=0))
