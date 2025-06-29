from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Reshape, LSTM
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# Data directories
train_data_dir = '/content/train'
validation_data_dir = '/content/test'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Only rescaling for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Model architecture
model = Sequential()

# CNN layers for feature extraction
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Convert CNN feature maps into sequences for LSTM
model.add(Reshape((36, 256)))  # Reshape correctly (6x6=36 time steps, 256 features)
model.add(LSTM(128, return_sequences=False))  # LSTM layer processing sequential features

# Dense layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotion classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),  # Tuned learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
model_checkpoint = ModelCheckpoint('best_model_lstm.keras', monitor='val_accuracy', save_best_only=True)

# Count images
num_train_imgs = sum(len(files) for _, _, files in os.walk(train_data_dir))
num_test_imgs = sum(len(files) for _, _, files in os.walk(validation_data_dir))

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 64,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 64,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Save the model
model.save('my_model_lstm.keras')
