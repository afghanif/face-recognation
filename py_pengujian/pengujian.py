import tensorflow as tf
import numpy as np

# Muat model TensorFlow FaceAPI
model = tf.keras.models.load_model('models')

# Buat objek metrik
accuracy_metric = Accuracy()
precision_metric = Precision()
recall_metric = Recall()
f1_score = 2 * (precision_metric * recall_metric) / \
    (precision_metric + recall_metric)

# Impor data pengujian dari file Python
data = np.load('pengujian/src/generate_training_data.py')

# Pra-pemrosesan data pengujian
# Fungsi preprocess_data adalah fungsi yang harus Anda definisikan sendiri
preprocessed_data = preprocess_data(data)

# Lakukan prediksi menggunakan model
predictions = model.predict(preprocessed_data)

# Ubah label yang diprediksi menjadi one-hot encoding
# Ganti num_classes dengan jumlah kelas yang sesuai
y_pred = tf.one_hot(np.argmax(predictions, axis=1), depth=5)

# Hitung metrik-metrik evaluasi
accuracy_metric.update_state(y_true, y_pred)
precision_metric.update_state(y_true, y_pred)
recall_metric.update_state(y_true, y_pred)
f1_score.update_state(y_true, y_pred)

# Dapatkan nilai metrik evaluasi
accuracy = accuracy_metric.result().numpy()
precision = precision_metric.result().numpy()
recall = recall_metric.result().numpy()
f1_score = f1_score.result()

# Tampilkan hasil pengujian
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
