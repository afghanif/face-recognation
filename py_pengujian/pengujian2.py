import tensorflow as tf
import cv2
import numpy as np

# Muat model TensorFlow FaceAPI
model = tf.keras.models.load_model('model.h5')

file_path = 'pengujian/src/generate_training_data.py'
# Load model Python
with open(file_path, 'r') as file:
    data = file.read()

new_width = 100
new_height = 100


def preprocess_data(data, new_width, new_height):
    preprocessed_data = []

    for image_path in data:
        # Load image
        image = cv2.imread(image_path)

        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Convert image to RGB format
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Normalize image pixel values to range [0, 1]
        normalized_image = rgb_image / 255.0

        # Append preprocessed image to the list
        preprocessed_data.append(normalized_image)

    # Convert preprocessed data to NumPy array
    preprocessed_data = np.array(preprocessed_data)

    return preprocessed_data


# Pra-pemrosesan data pengujian
preprocessed_data = preprocess_data(data)

# Lakukan prediksi
y_pred = model.predict(preprocessed_data)
y_true = [0, 1, 1, 0, 1]

# Konversi hasil prediksi menjadi label kelas
y_pred_labels = tf.argmax(y_pred, axis=1)
y_true_labels = tf.argmax(y_true, axis=1)

# Menghitung parameter evaluasi
accuracy_metric = tf.keras.metrics.Accuracy()
accuracy_metric.update_state(y_true_labels, y_pred_labels)
accuracy = accuracy_metric.result().np()

precision_metric = tf.keras.metrics.Precision()
precision_metric.update_state(y_true_labels, y_pred_labels)
precision = precision_metric.result().np()

recall_metric = tf.keras.metrics.Recall()
recall_metric.update_state(y_true_labels, y_pred_labels)
recall = recall_metric.result().np()

f1 = 2 * (precision * recall) / (precision + recall)

# Dapatkan nilai metrik evaluasi
accuracy = accuracy_metric.result().np()
precision = precision_metric.result().np()
recall = recall_metric.result().np()
f1_score = f1.result().np()

# Tampilkan hasil pengujian
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
