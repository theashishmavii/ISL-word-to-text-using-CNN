# ------------------------------- Import Necessary Libraries -------------------------------

import numpy as np
import pandas as pd
import keras
from keras import layers
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# ------------------------------- Define File Path, Dictonary and Parameters -------------------------------

excel_file="C:/Sign-language Project/datasets/ISL_CSLRT_Corpus/corpus_csv_files/ISL_CSLRT_Corpus_word_details.xlsx"
image_size=64
words=['A LOT', 'ABUSE', 'AFRAID', 'AGREE', 'ALL', 'ANGRY', 'ANYTHING', 'APPRECIATE', 'BAD', 'BEAUTIFUL',
'BECOME', 'BED', 'BORED', 'BRING', 'CHAT', 'CLASS', 'COLD', 'COLLEGE_SCHOOL', 'COMB', 'COME',
'CONGRATULATIONS', 'CRYING', 'DARE', 'DIFFERENCE', 'DILEMMA', 'DISAPPOINTED', 'DO', "DON'T CARE",
'ENJOY', 'FAVOUR', 'FEVER', 'FINE', 'FOOD', 'FREE', 'FRIEND', 'FROM', 'GO', 'GOOD', 'GRATEFUL',
'HAD', 'HAPPENED', 'HAPPY', 'HEAR', 'HEART', 'HELLO_HI', 'HELP', 'HIDING', 'HOW', 'HUNGRY',
'HURT', 'I_ME_MINE_MY', 'KIND', 'LEAVE', 'LIKE', 'LIKE_LOVE', 'MEAN IT', 'MEDICINE', 'MEET',
'NAME', 'NICE', 'NOT', 'NUMBER', 'OLD_AGE', 'ON THE WAY', 'OUTSIDE', 'PHONE', 'PLACE', 'PLEASE',
'POUR', 'PREPARE', 'PROMISE', 'REALLY', 'REPEAT', 'ROOM', 'SERVE', 'SHIRT', 'SITTING', 'SLEEP',
'SLOWER', 'SO MUCH', 'SOFTLY', 'SOME HOW', 'SOME ONE', 'SOMETHING', 'SORRY', 'SPEAK', 'STOP',
'STUBBORN', 'SURE', 'TAKE CARE', 'TAKE TIME', 'TALK', 'TELL', 'THANK', 'THAT', 'THINGS', 'THINK',
'THIRSTY', 'TIRED', 'TODAY', 'TRAIN', 'TRUST', 'TRUTH', 'TURN ON', 'UNDERSTAND', 'WANT', 'WATER',
'WEAR', 'WELCOME', 'WHAT', 'WHERE', 'WHO', 'WORRY', 'YOU']
num_classes=len(words)
label_dict={i:words[i] for i in range(len(words))}

# ------------------------------- Function to Load Data from Excel -------------------------------

def load_data_from_excel(excel_file):
    df = pd.read_excel(excel_file)
    images, labels = [], []

    for index, row in df.iterrows():
        img_path = os.path.join("C:/Sign-language Project/datasets/",str(row['Frames path']))
        label = str(row['Word']).strip()

        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            continue

        # Read and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        images.append(img)

        if label not in label_dict.values():
            print(f"Warning: Label '{label}' not found in label_dict!")
            continue

        label_index=list(label_dict.values()).index(label)
        labels.append(label_index)

    images = np.array(images).reshape(-1, image_size, image_size, 1) / 255.0  # Normalize
    labels = to_categorical(np.array(labels), num_classes)  # One-hot encoding
    return train_test_split(images, labels, test_size=0.2, random_state=42)

# ------------------------------- Load Dataset -------------------------------

x_train, x_test, y_train, y_test = load_data_from_excel(excel_file)

# ------------------------------- Define CNN Model -------------------------------

model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(num_classes, activation='softmax')
])

# ------------------------------- Compile the Model -------------------------------

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------------------- Train the Model -------------------------------

model.fit(x_train, y_train, epochs=20, batch_size=34, validation_data=(x_test, y_test))

# ------------------------------- Evaluate the Model -------------------------------

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# ------------------------------- Save the Trained Model -------------------------------

model.save("cnn_word_model.h5")