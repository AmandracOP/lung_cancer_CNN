import tensorflow as tf
from tensorflow.keras.applications import VGG19, MobileNetV2, ResNet50, EfficientNetB0, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load dataset using ImageDataGenerator
def load_data(train_dir=None, validation_dir=None, test_dir=None, img_size=(224, 224), batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Initialize data generators to None
    train_gen, val_gen, test_gen = None, None, None
    
    # Train data generator (only if train_dir is provided)
    if train_dir:
        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
    
    # Validation data generator (only if validation_dir is provided)
    if validation_dir:
        val_gen = datagen.flow_from_directory(
            validation_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
    
    # Test data generator (only if test_dir is provided)
    if test_dir:
        test_gen = datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    return train_gen, val_gen, test_gen

# Model builder function to add a custom classification layer
def build_model(base_model, dense_units=1024, num_classes=3, trainable=False):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # 3 classes: ACA, SCC, Normal
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Set base model layers' trainability
    for layer in base_model.layers:
        layer.trainable = trainable
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define a dictionary of base models to easily access in train and evaluation scripts
def get_base_models(input_shape=(224, 224, 3)):
    return {
        'VGG19': VGG19(weights='imagenet', include_top=False, input_shape=input_shape),
        'MobileNet': MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
        'ResNet': ResNet50(weights='imagenet', include_top=False, input_shape=input_shape),
        'EfficientNet': EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape),
        'Inception': InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    }
