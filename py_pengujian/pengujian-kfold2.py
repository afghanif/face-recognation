from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Baca file CSV
dataset = pd.read_csv('pengujian/fer2013new.csv')

# Mengatur ulang indeks menjadi indeks bilangan bulat
dataset.reset_index(drop=True, inplace=True)

# Inisialisasi objek KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Membuat variabel untuk menyimpan hasil metrik
accuracies = []
precisions = []
recalls = []
f1_scores = []
classification_reports = []

# Ambil kolom emosi yang merupakan label emosi atau ekspresi
ground_truth = dataset['happiness']

# Hasil prediksi Face API
predicted_labels = dataset[['neutral', 'happiness', 'surprise', 'sadness',
                            'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']]
predicted_labels = predicted_labels.astype(str)
predicted_labels = np.argmax(predicted_labels.values, axis=1)

# Lakukan pelatihan dan pengujian menggunakan K-Fold Cross Validation
for train_index, test_index in kf.split(dataset):
    # Bagi dataset menjadi set pelatihan dan pengujian
    train_set = dataset.iloc[train_index]
    test_set = dataset.iloc[test_index]

    model = DecisionTreeClassifier()

    # Lakukan pelatihan model
    model.fit(train_set)

    # Lakukan prediksi pada set pengujian
    predicted_labels = model.predict(test_set)

    # Evaluasi performa menggunakan metrik-metrik yang diinginkan
    accuracy = accuracy_score(ground_truth, predicted_labels)
    precision = precision_score(
        ground_truth, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, predicted_labels, average='weighted')
    f1 = f1_score(ground_truth, predicted_labels, average='weighted')
    report = classification_report(ground_truth, predicted_labels)

    # Menyimpan hasil metrik ke dalam variabel
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    classification_reports.append(report)

# Menampilkan hasil metrik rata-rata dari seluruh fold
print("Accuracy:", sum(accuracies) / len(accuracies))
print("Precision:", sum(precisions) / len(precisions))
print("Recall:", sum(recalls) / len(recalls))
print("F1 Score:", sum(f1_scores) / len(f1_scores))
print("Classification Report:")
print("\n".join(classification_reports))
