import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Tüm veri yolu
data_path = ".../card_dataset/veriseti2/train"
test_path = ".../card_dataset/veriseti2/test"
valid_path = ".../card_dataset/veriseti2/valid"
checkpoint_path = ".../card_dataset/best_model.h5"

target_size = (224, 224)

# Veri ön işleme ve yükleme
datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=3,  # Rastgele döndürme
    width_shift_range=0.03,  # Genişlik yönlü rastgele kaydırma
    height_shift_range=0.03,  # Yükseklik yönlü rastgele kaydırma
    shear_range=0.03,  # Kesme dönüşü
    zoom_range=0.03,  # Rastgele yakınlaştırma
    horizontal_flip=True,  # Yatay simetri
    fill_mode='nearest')  # Doldurma modu

# Test verilerini yükleme
test_generator = datagen.flow_from_directory(
        test_path,
        target_size=target_size,
        batch_size=16,
        class_mode='categorical')

train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=16,
        shuffle = True,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        valid_path,
        target_size=target_size,
        batch_size=16,
        class_mode='categorical')

# Model oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.11),

    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.11),

    tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.11),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.11),

    tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.11),

    tf.keras.layers.Dense(53, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Modelin ağırlıklarını kaydetmek için bir ModelCheckpoint oluşturma
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Modeli eğitme ve ModelCheckpoint geri çağırımını callbacks listesine ekleyerek kullanma
history=model.fit(train_generator, epochs=50, validation_data=validation_generator, shuffle=True, callbacks=[early_stopping, checkpoint])

# Test seti üzerinde modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

pd.DataFrame(history.history).plot(grid=True)
plt.show()