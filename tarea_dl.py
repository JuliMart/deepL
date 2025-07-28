# -*- coding: utf-8 -*-
# Tarea de Deep Learning 2025: Clasificación de imágenes con Tiny ImageNet!

# Cargar Tiny ImageNet desde TFDS personalizado
!pip install git+https://github.com/ksachdeva/tiny-imagenet-tfds.git

# Descargar y preparar Tiny ImageNet
import os

# Descargar y descomprimir si no existe
if not os.path.exists("tiny-imagenet-200"):
    import urllib.request, zipfile
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    urllib.request.urlretrieve(url, "tiny-imagenet-200.zip")
    with zipfile.ZipFile("tiny-imagenet-200.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

#Cargar Tiny ImageNet desde directorios
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import shutil
import pandas as pd

# Ruta local al dataset descomprimido
dataset_dir = "tiny-imagenet-200"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val", "images")

# Reorganizar val/imagenes a subdirectorios por clase si es necesario
annotations_path = os.path.join(dataset_dir, "val", "val_annotations.txt")
df = pd.read_csv(annotations_path, sep='\t', header=None)
df.columns = ["image", "label", "x1", "y1", "x2", "y2"]

val_class_dir = os.path.join(dataset_dir, "val", "organized")
os.makedirs(val_class_dir, exist_ok=True)

for _, row in df.iterrows():
    label_dir = os.path.join(val_class_dir, row["label"])
    os.makedirs(label_dir, exist_ok=True)
    src = os.path.join(val_dir, row["image"])
    dst = os.path.join(label_dir, row["image"])
    if os.path.exists(src):
        shutil.copy(src, dst)

# Cargar datasets con image_dataset_from_directory
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=64,
    label_mode="int"
)

val_ds = image_dataset_from_directory(
    val_class_dir,
    image_size=(224, 224),
    batch_size=64,
    label_mode="int"
)

# Normalizar imágenes UNA SOLA VEZ EJECUTAR
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# Mostrar una imagen de ejemplo
for images, labels in train_ds.take(1):
    import matplotlib.pyplot as plt
    plt.imshow(images[6].numpy())
    plt.title(f"Etiqueta: {labels[0].numpy()}")
    plt.axis('off')
    plt.show()

import matplotlib.pyplot as plt

for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(f"Etiqueta: {labels[i].numpy()}")
        plt.axis("off")
    plt.show()

for images, labels in train_ds.take(1):
    print("Forma:", images[0].shape)
    print("Valor mínimo:", tf.reduce_min(images[0]).numpy())
    print("Valor máximo:", tf.reduce_max(images[0]).numpy())

# 1. Bloque de Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

# 2. Base ResNet50 preentrenada (sin top)
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
# Congelar las primeras capas, liberar solo las últimas
for layer in base_model.layers[:-50]:
    layer.trainable = False

# 3. Armado del modelo completo
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(200, activation='softmax')(x)

resnet_model = tf.keras.Model(inputs, outputs)

# 4. Compilar modelo
resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Extraer etiquetas reales desde train_ds
y_train = []
for _, label in train_ds.unbatch():
    y_train.append(label.numpy())

y_train = np.array(y_train)
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# 5. Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint("resnet_tinyimagenet.keras", monitor='val_accuracy', save_best_only=True)

# 6. Entrenamiento inicial (solo la cabeza)
history1 = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

from sklearn.metrics import classification_report
import numpy as np

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = resnet_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, digits=4)
print(report)

import matplotlib.pyplot as plt

def plot_history(history_obj):
    history = history_obj.history
    epochs = range(len(history['loss']))

    plt.figure(figsize=(14, 5))

    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy')
    plt.title('Precisión por época')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Pérdida por época')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy

#  Solo libero las últimas capas de ResNet
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompilar con menor LR para evitar sobreajuste
from tensorflow.keras import metrics


resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=[
        SparseCategoricalAccuracy(name='accuracy'),
        SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
)

# 8. Segundo entrenamiento (fine-tuning total)
history2 = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    initial_epoch=10,  # para continuar desde la epoch 10
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

from sklearn.metrics import classification_report
import numpy as np

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = resnet_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, digits=4)
print(report)

import matplotlib.pyplot as plt

def plot_combined_history(history1, history2):
    # Combinar las métricas
    combined = {}
    for key in history1.history.keys():
        combined[key] = history1.history[key] + history2.history[key]

    epochs = range(len(combined['loss']))

    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, combined['accuracy'], label='Train Accuracy')
    plt.plot(epochs, combined['val_accuracy'], label='Val Accuracy')
    plt.title('Precisión por época')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, combined['loss'], label='Train Loss')
    plt.plot(epochs, combined['val_loss'], label='Val Loss')
    plt.title('Pérdida por época')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_combined_history(history1, history2)

# 8. Tercer entrenamiento (fine-tuning total)
history3 = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    initial_epoch=20,  # para continuar desde la epoch 20
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

from sklearn.metrics import classification_report
import numpy as np

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = resnet_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, digits=4)
print(report)

import matplotlib.pyplot as plt

def plot_combined_history(history1, history2,history3):
    # Combinar las métricas
    combined = {}
    for key in history1.history.keys():
        combined[key] = history1.history[key] + history2.history[key] + history3.history[key]

    epochs = range(len(combined['loss']))

    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, combined['accuracy'], label='Train Accuracy')
    plt.plot(epochs, combined['val_accuracy'], label='Val Accuracy')
    plt.title('Precisión por época')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, combined['loss'], label='Train Loss')
    plt.plot(epochs, combined['val_loss'], label='Val Loss')
    plt.title('Pérdida por época')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_combined_history(history1, history2, history3)

history4 = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    initial_epoch=30,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

import matplotlib.pyplot as plt

def plot_combined_history(history1, history2,history3, history4):
    # Combinar las métricas
    combined = {}
    for key in history1.history.keys():
        combined[key] = history1.history[key] + history2.history[key] + history3.history[key] + history4.history[key]

    epochs = range(len(combined['loss']))

    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, combined['accuracy'], label='Train Accuracy')
    plt.plot(epochs, combined['val_accuracy'], label='Val Accuracy')
    plt.title('Precisión por época')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, combined['loss'], label='Train Loss')
    plt.plot(epochs, combined['val_loss'], label='Val Loss')
    plt.title('Pérdida por época')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

from sklearn.metrics import classification_report
import numpy as np

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = resnet_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, digits=4)
print(report)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Asegurate de tener y_true y y_pred como listas de etiquetas
cm = confusion_matrix(y_true, y_pred, labels=range(200))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalizada por fila

plt.figure(figsize=(20, 16))
sns.heatmap(cm_norm, cmap="Blues", square=False,
            xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'Proporción'})
plt.title("Matriz de Confusión Normalizada (Tiny ImageNet)")
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.show()

from sklearn.metrics import classification_report
import pandas as pd

# Obtener el reporte como diccionario
report_dict = classification_report(y_true, y_pred, output_dict=True)

# Convertir a DataFrame
df_report = pd.DataFrame(report_dict).transpose()

# Filtrar solo las filas que son clases numéricas
df_clases = df_report[df_report.index.str.isnumeric()]

# Ordenar por recall ascendente (menor a mayor)
peor_recall = df_clases.sort_values(by='recall').head(10)

print(peor_recall[['precision', 'recall', 'f1-score']])

def oversample_clases(ds, clases_ids, veces=3):
    # Lista de datasets sobre-replicados
    ds_extra = []
    for clase in clases_ids:
        ds_clase = filtrar_por_clase(ds, clase)
        ds_clase_rep = ds_clase.repeat(veces)
        ds_extra.append(ds_clase_rep)

    # Combinar todas las clases reforzadas
    ds_oversampled = ds_extra[0]
    for ds_add in ds_extra[1:]:
        ds_oversampled = ds_oversampled.concatenate(ds_add)

    return ds_oversampled

# Clases con bajo recall (ejemplo: clases 131, 175, 159, ...)
clases_a_reforzar = [131, 175, 159, 80, 172, 49, 180, 135, 132, 168]

# Crear dataset con oversampling
ds_oversampled = oversample_clases(train_ds, clases_a_reforzar, veces=3)

# Mezclar todo
train_ds_final = train_ds.concatenate(ds_oversampled).shuffle(10000).batch(64)

"""# Prototipado con Gradio, para subir foto del set de TEST"""

from tensorflow.keras.models import load_model

resnet_model = load_model("resnet_tinyimagenet.keras")

with open("words200.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = resnet_model.predict(image_array)[0]  # <--- usa el nombre correcto
    top_5 = predictions.argsort()[-5:][::-1]
    return {class_names[i]: float(predictions[i]) for i in top_5}

import gradio as gr

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Tiny ImageNet Classifier"
).launch()

print(len(class_names))

print(resnet_model.evaluate(val_ds))

