from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np

# Baca file CSV dengan data referensi dari FERPlus dataset
df = pd.read_csv('pengujian/fer2013new.csv')
emosi_columns = ['neutral', 'happiness', 'surprise', 'sadness',
                 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

# Fungsi untuk menggabungkan nilai emosi dalam setiap baris


def gabungkan_emosi(row):
    return ','.join([str(row[emosi]) for emosi in emosi_columns])


# Tambahkan kolom "master emotion" dengan menggabungkan nilai emosi
df['master emotion'] = df.apply(gabungkan_emosi, axis=1)

# Simpan dataset dengan kolom "master emotion"
df.to_csv('master.csv', index=False)

# Ambil kolom emosi yang merupakan label emosi atau ekspresi
ground_truth = df['master emotion']

# print(type(ground_truth))
# ground_truth = ground_truth.astype(int)

ground_truth = pd.DataFrame(ground_truth, columns=['master emotion'])

# Hasil prediksi Face API
predicted_labels = df[['neutral', 'happiness', 'surprise', 'sadness',
                       'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']]
predicted_labels = predicted_labels.astype(str)
predicted_labels = np.argmax(predicted_labels.values, axis=1)

# Konversi predicted_labels menjadi dataframe dengan satu kolom yang sesuai
predicted_labels = pd.DataFrame(predicted_labels, columns=['emotion'])

# Reset indeks pada kedua dataframe
ground_truth = ground_truth.reset_index(drop=True)
predicted_labels = predicted_labels.reset_index(drop=True)
mask = (ground_truth['master emotion'] != predicted_labels['emotion'])

ground_truth_cleaned = ground_truth.loc[~mask]
predicted_labels_cleaned = predicted_labels.loc[~mask]

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
