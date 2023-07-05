from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Baca file CSV
dataset = pd.read_csv('pengujian/fer2013new.csv')

# Mengatur ulang indeks menjadi indeks bilangan bulat
dataset.reset_index(drop=True, inplace=True)

# Definisikan jumlah lipatan (K)
k = 5

# Bagi dataset menjadi K subset dengan KFold
kf = KFold(n_splits=k, shuffle=True)

# Inisialisasi list untuk menyimpan metrik evaluasi
accuracies = []

# Ambil kolom emosi yang merupakan label emosi atau ekspresi
ground_truth = dataset['neutral']

# Hasil prediksi Face API
predicted_labels = dataset[['neutral', 'happiness', 'surprise', 'sadness',
                            'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']]
predicted_labels = predicted_labels.astype(str)
predicted_labels = np.argmax(predicted_labels.values, axis=1)

# Lakukan pelatihan dan pengujian menggunakan K-Fold Cross Validation
for train_index, test_index in kf.split(dataset):
    # Dapatkan subset data train dan test berdasarkan indeks
    train_data = dataset.iloc[train_index]
    test_data = dataset.iloc[test_index]

    # Lakukan pelatihan menggunakan train_set dan prediksi pada test_set menggunakan FaceAPI
    # ...

    # Evaluasi performa dengan menghitung akurasi
    accuracy = accuracy_score(ground_truth, predicted_labels)
    accuracies.append(accuracy)

# Hitung rata-rata dan deviasi standar dari akurasi
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Tampilkan hasil evaluasi performa
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Standard Deviation of Accuracy: {std_accuracy}")
