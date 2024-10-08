import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from sklearn.utils import shuffle
import random

# Constants
IMG_SIZE = 112  # Image size for resizing
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "./people"
MODEL_WEIGHTS_PATH = "siamese_model_weights.h5"
LEARNING_RATE = 0.0001


# Create pairs of images
def create_pairs(image_folder, n_pairs_per_person=5):
    pairs = []
    labels = []

    people = os.listdir(image_folder)

    for person in people:
        person_folder = os.path.join(image_folder, person)
        images = os.listdir(person_folder)

        for _ in range(2 * n_pairs_per_person):
            img1, img2 = random.sample(images, 2)
            img1_path = os.path.join(person_folder, img1)
            img2_path = os.path.join(person_folder, img2)
            pairs.append((img1_path, img2_path))
            labels.append(1)

        for _ in range(n_pairs_per_person):
            other_person = random.choice([p for p in people if p != person])
            other_person_folder = os.path.join(image_folder, other_person)
            img1 = random.choice(images)
            img2 = random.choice(os.listdir(other_person_folder))
            img1_path = os.path.join(person_folder, img1)
            img2_path = os.path.join(other_person_folder, img2)
            pairs.append((img1_path, img2_path))
            labels.append(0)

    pairs, labels = shuffle(pairs, labels, random_state=42)
    return pairs, labels


# Load and preprocess images
def load_and_preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE), grayscale=False):
    image = load_img(image_path, target_size=target_size, color_mode='grayscale' if grayscale else 'rgb')
    image = img_to_array(image)
    image /= 255.0
    return image


# Data generator
def data_generator(pairs, labels, batch_size, target_size=(IMG_SIZE, IMG_SIZE), grayscale=False, augment=False):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    while True:
        batch_indices = np.random.choice(len(pairs), size=batch_size)
        imgsA = []
        imgsB = []
        batch_labels = []

        for idx in batch_indices:
            imgA_path, imgB_path = pairs[idx]
            label = labels[idx]

            imgA = load_and_preprocess_image(imgA_path, target_size, grayscale)
            imgB = load_and_preprocess_image(imgB_path, target_size, grayscale)

            if augment:
                imgA = datagen.random_transform(imgA)
                imgB = datagen.random_transform(imgB)

            imgsA.append(imgA)
            imgsB.append(imgB)
            batch_labels.append(label)

        yield [np.array(imgsA), np.array(imgsB)], np.array(batch_labels)


# Define base network
def build_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


# Distance function
def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))


def thresholded_output(x):
    return K.cast(x > 0.5, dtype='float32')


# Build and compile model
def build_model(input_shape):
    base_model = build_base_network(input_shape)

    inputA = Input(input_shape)
    inputB = Input(input_shape)

    featsA = base_model(inputA)
    featsB = base_model(inputB)

    distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)

    model = Model(inputs=[inputA, inputB], outputs=outputs)

    return model


# Main function
def main():
    pairs, labels = create_pairs(DATASET_PATH, n_pairs_per_person=3)

    train_pairs, train_labels = pairs[:int(len(pairs) * 0.8)], labels[:int(len(labels) * 0.8)]
    val_pairs, val_labels = pairs[int(len(pairs) * 0.8):], labels[int(len(labels) * 0.8):]

    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model = build_model(input_shape)

    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=LEARNING_RATE), metrics=["accuracy"])

    train_gen = data_generator(train_pairs, train_labels, BATCH_SIZE, augment=True)
    val_gen = data_generator(val_pairs, val_labels, BATCH_SIZE)

    steps_per_epoch = len(train_pairs) // BATCH_SIZE
    validation_steps = len(val_pairs) // BATCH_SIZE

    model.fit(train_gen,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_gen,
              validation_steps=validation_steps,
              epochs=EPOCHS)

    model.save(MODEL_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
