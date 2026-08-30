import numpy as np
import os

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

from labels import GESTURES


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data"

SEQUENCE_LENGTH = 20

FEATURES = 225


# ============================================================
# LOAD ALL DATA
# ============================================================

X = []

y = []


print()
print("==========================================")
print("LOADING DATA")
print("==========================================")


for idx, gesture in enumerate(

    GESTURES

):


    gesture_path = os.path.join(

        DATA_PATH,

        gesture

    )


    if not os.path.exists(

        gesture_path

    ):

        print(

            f"Warning: No data folder for {gesture}"

        )

        continue


    files = [

        f

        for f in os.listdir(

            gesture_path

        )

        if f.endswith(

            ".npy"

        )

    ]


    print(

        gesture,

        ":",

        len(files),

        "sequences"

    )


    for seq_file in files:


        file_path = os.path.join(

            gesture_path,

            seq_file

        )


        data = np.load(

            file_path

        )


        # Check shape

        if data.shape != (

            SEQUENCE_LENGTH,

            FEATURES

        ):


            print(

                "Skipping invalid:",

                file_path,

                data.shape

            )


            continue


        X.append(

            data

        )


        y.append(

            idx

        )


# ============================================================
# CHECK DATA
# ============================================================

if len(X) == 0:

    print()

    print(

        "ERROR: No valid training data found."

    )

    exit()


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(

    X,

    dtype=np.float32

)


y = np.array(

    y

)


print()

print(

    "Total samples:",

    len(X)

)

print(

    "Input shape:",

    X.shape

)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

y_categorical = to_categorical(

    y,

    num_classes=len(

        GESTURES

    )

)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_categorical,

    test_size=0.2,

    random_state=42,

    stratify=y

)


# ============================================================
# BUILD MODEL
# ============================================================

model = Sequential()


# ============================================================
# FIRST BIDIRECTIONAL LSTM
# ============================================================

model.add(

    Bidirectional(

        LSTM(

            64,

            return_sequences=True

        ),

        input_shape=(

            SEQUENCE_LENGTH,

            FEATURES

        )

    )

)


model.add(

    Dropout(

        0.3

    )

)


# ============================================================
# SECOND BIDIRECTIONAL LSTM
# ============================================================

model.add(

    Bidirectional(

        LSTM(

            32

        )

    )

)


model.add(

    Dropout(

        0.3

    )

)


# ============================================================
# DENSE LAYER
# ============================================================

model.add(

    Dense(

        64,

        activation="relu"

    )

)


model.add(

    Dropout(

        0.2

    )

)


# ============================================================
# OUTPUT LAYER
# ============================================================

model.add(

    Dense(

        len(

            GESTURES

        ),

        activation="softmax"

    )

)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=[

        "accuracy"

    ]

)


model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True

)


# ============================================================
# TRAIN
# ============================================================

print()

print(

    "=========================================="

)

print(

    "TRAINING STARTED"

)

print(

    "=========================================="

)


history = model.fit(

    X_train,

    y_train,

    epochs=50,

    batch_size=8,

    validation_data=(

        X_test,

        y_test

    ),

    callbacks=[

        early_stopping

    ]

)


# ============================================================
# EVALUATE
# ============================================================

loss, accuracy = model.evaluate(

    X_test,

    y_test

)


print()

print(

    "Test Accuracy:",

    accuracy

)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(

    "gesture_model.h5"

)


print()

print(

    "=========================================="

)

print(

    "TRAINING COMPLETE"

)

print(

    "=========================================="

)

print(

    "Model saved as gesture_model.h5"

)