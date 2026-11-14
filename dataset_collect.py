import cv2
import numpy as np
import mediapipe as mp
import os

from labels import GESTURES


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data"

SEQUENCE_LENGTH = 20

NO_OF_SEQUENCES = 30

MIN_BRIGHTNESS = 50

MIN_DETECTION_CONFIDENCE = 0.7

MIN_TRACKING_CONFIDENCE = 0.7


# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================

mp_holistic = mp.solutions.holistic

mp_draw = mp.solutions.drawing_utils


holistic = mp_holistic.Holistic(

    static_image_mode=False,

    model_complexity=1,

    min_detection_confidence=MIN_DETECTION_CONFIDENCE,

    min_tracking_confidence=MIN_TRACKING_CONFIDENCE

)
# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(landmarks):

    coords = []

    for lm in landmarks.landmark:

        coords.append([

            lm.x,

            lm.y,

            lm.z

        ])
    coords = np.array(

        coords,

        dtype=np.float32

    )
    # Use first landmark as origin

    origin = coords[0]

    coords = coords - origin
    # Scale normalization

    max_val = np.max(

        np.abs(coords)

    )
    if max_val != 0:

        coords = coords / max_val


    return coords.flatten()


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(results):
    # --------------------------------------------------------
    # POSE LANDMARKS
    # 33 landmarks × 3 = 99 features
    # --------------------------------------------------------

    if results.pose_landmarks:

        pose = normalize_landmarks(

            results.pose_landmarks

        )

    else:

        pose = np.zeros(

            99,

            dtype=np.float32

        )
    # --------------------------------------------------------
    # LEFT HAND LANDMARKS
    # 21 landmarks × 3 = 63 features
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        left_hand = normalize_landmarks(

            results.left_hand_landmarks

        )

    else:

        left_hand = np.zeros(

            63,

            dtype=np.float32

        )
    # --------------------------------------------------------
    # RIGHT HAND LANDMARKS
    # 21 landmarks × 3 = 63 features
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        right_hand = normalize_landmarks(

            results.right_hand_landmarks

        )

    else:

        right_hand = np.zeros(

            63,

            dtype=np.float32

        )
    # --------------------------------------------------------
    # COMBINE ALL FEATURES
    #
    # Pose      = 99
    # Left hand = 63
    # Right     = 63
    #
    # Total     = 225 features
    # --------------------------------------------------------

    features = np.concatenate([

        pose,

        left_hand,

        right_hand

    ])
    return features
# ============================================================
# COUNT EXISTING SEQUENCES
# ============================================================

def get_existing_sequence_count(gesture):


    gesture_path = os.path.join(

        DATA_PATH,

        gesture

    )
    # Create folder if it doesn't exist

    os.makedirs(

        gesture_path,

        exist_ok=True

    )
    # Get all .npy files

    files = [

        f

        for f in os.listdir(

            gesture_path

        )

        if f.endswith(".npy")

    ]
    return len(files)
# ============================================================
# SHOW DATASET STATUS
# ============================================================

def show_dataset_status():

    print()

    print("=" * 50)

    print("CURRENT DATASET STATUS")

    print("=" * 50)

    print(

        f"{'No.':<5}"

        f"{'Gesture':<20}"

        f"{'Sequences':<15}"

    )

    print("-" * 50)


    for index, gesture in enumerate(

        GESTURES,

        start=1

    ):


        count = get_existing_sequence_count(

            gesture

        )


        print(

            f"{index:<5}"

            f"{gesture:<20}"

            f"{count:<15}"

        )


    print("=" * 50)


# ============================================================
# SELECT GESTURE
# ============================================================

def select_gesture():


    while True:


        print()

        print("=" * 50)

        print("SELECT GESTURE")

        print("=" * 50)


        for index, gesture in enumerate(

            GESTURES,

            start=1

        ):

            count = get_existing_sequence_count(

                gesture

            )


            print(

                f"{index}. "

                f"{gesture} "

                f"({count} sequences)"

            )


        print()

        print("0. Exit")


        try:


            choice = int(

                input(

                    "\nEnter gesture number: "

                )

            )


            if choice == 0:

                return None


            if 1 <= choice <= len(

                GESTURES

            ):


                selected_gesture = GESTURES[

                    choice - 1

                ]


                return selected_gesture


            else:


                print()

                print(

                    "Invalid choice."

                )


        except ValueError:


            print()

            print(

                "Please enter a valid number."

            )


# ============================================================
# COLLECT DATA FOR SELECTED GESTURE
# ============================================================

def collect_gesture_data(gesture):


    gesture_path = os.path.join(

        DATA_PATH,

        gesture

    )


    os.makedirs(

        gesture_path,

        exist_ok=True

    )


    # --------------------------------------------------------
    # FIND EXISTING DATA
    # --------------------------------------------------------

    existing_count = get_existing_sequence_count(

        gesture

    )


    start_sequence = existing_count


    print()

    print("=" * 50)

    print("GESTURE SELECTED")

    print("=" * 50)

    print(

        "Gesture:",

        gesture

    )

    print(

        "Existing sequences:",

        existing_count

    )

    print(

        "New sequences:",

        NO_OF_SEQUENCES

    )

    print(

        "Starting sequence number:",

        start_sequence

    )

    print("=" * 50)


    input(

        "\nPress ENTER to open camera..."

    )


    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(

        0

    )


    if not cap.isOpened():


        print()

        print(

            "ERROR: Could not open camera."

        )


        return


    # ========================================================
    # COLLECT SEQUENCES
    # ========================================================

    for sequence_index in range(

        NO_OF_SEQUENCES

    ):


        sequence_number = (

            start_sequence

            +

            sequence_index

        )


        print()

        print(

            f"Preparing sequence "

            f"{sequence_number}..."

        )


        # ----------------------------------------------------
        # WAIT FOR USER TO START
        # ----------------------------------------------------

        while True:


            ret, frame = cap.read()


            if not ret:

                continue


            frame = cv2.flip(

                frame,

                1

            )


            # -----------------------------------------------
            # CHECK LIGHTING
            # -----------------------------------------------

            gray = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2GRAY

            )


            brightness = np.mean(

                gray

            )


            lighting_good = (

                brightness >= MIN_BRIGHTNESS

            )


            # -----------------------------------------------
            # MEDIAPIPE
            # -----------------------------------------------

            rgb = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB

            )


            results = holistic.process(

                rgb

            )


            # -----------------------------------------------
            # HAND DETECTION
            # -----------------------------------------------

            left_detected = (

                results.left_hand_landmarks

                is not None

            )


            right_detected = (

                results.right_hand_landmarks

                is not None

            )


            hand_detected = (

                left_detected

                or

                right_detected

            )


            # -----------------------------------------------
            # DRAW POSE
            # -----------------------------------------------

            if results.pose_landmarks:


                mp_draw.draw_landmarks(

                    frame,

                    results.pose_landmarks,

                    mp_holistic.POSE_CONNECTIONS

                )


            # -----------------------------------------------
            # DRAW LEFT HAND
            # -----------------------------------------------

            if left_detected:


                mp_draw.draw_landmarks(

                    frame,

                    results.left_hand_landmarks,

                    mp_holistic.HAND_CONNECTIONS

                )


            # -----------------------------------------------
            # DRAW RIGHT HAND
            # -----------------------------------------------

            if right_detected:


                mp_draw.draw_landmarks(

                    frame,

                    results.right_hand_landmarks,

                    mp_holistic.HAND_CONNECTIONS

                )


            # -----------------------------------------------
            # DISPLAY GESTURE
            # -----------------------------------------------

            cv2.putText(

                frame,

                f"Gesture: {gesture}",

                (30, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 255),

                2

            )


            # -----------------------------------------------
            # DISPLAY BRIGHTNESS
            # -----------------------------------------------

            cv2.putText(

                frame,

                f"Brightness: {int(brightness)}",

                (30, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 0),

                2

            )


            # -----------------------------------------------
            # LIGHTING STATUS
            # -----------------------------------------------

            if lighting_good:


                cv2.putText(

                    frame,

                    "Lighting: GOOD",

                    (30, 120),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 0),

                    2

                )


            else:


                cv2.putText(

                    frame,

                    "Lighting: TOO LOW",

                    (30, 120),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 0, 255),

                    2

                )


            # -----------------------------------------------
            # HAND STATUS
            # -----------------------------------------------

            if hand_detected:


                cv2.putText(

                    frame,

                    "Hand: DETECTED",

                    (30, 160),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 0),

                    2

                )


            else:


                cv2.putText(

                    frame,

                    "Hand: NOT DETECTED",

                    (30, 160),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 0, 255),

                    2

                )


            # -----------------------------------------------
            # INSTRUCTIONS
            # -----------------------------------------------

            cv2.putText(

                frame,

                "Press S to record",

                (30, 210),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f"Sequence: "

                f"{sequence_index + 1}/"

                f"{NO_OF_SEQUENCES}",

                (30, 250),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 0),

                2

            )


            cv2.imshow(

                "Sign Language Data Collection",

                frame

            )


            key = cv2.waitKey(

                1

            ) & 0xFF


            # -----------------------------------------------
            # START RECORDING
            # -----------------------------------------------

            if key == ord('s'):


                # Check lighting

                if not lighting_good:


                    print()

                    print(

                        "Cannot record."

                    )

                    print(

                        "Lighting is too low."

                    )

                    continue


                # Check hand

                if not hand_detected:


                    print()

                    print(

                        "Cannot record."

                    )

                    print(

                        "No hand detected."

                    )

                    continue


                break


            # -----------------------------------------------
            # QUIT COLLECTION
            # -----------------------------------------------

            elif key == ord('q'):


                cap.release()

                cv2.destroyAllWindows()


                return


        # ====================================================
        # RECORD 20 FRAMES
        # ====================================================

        sequence = []


        print()

        print(

            f"Recording sequence "

            f"{sequence_number}..."

        )


        for frame_number in range(

            SEQUENCE_LENGTH

        ):


            ret, frame = cap.read()


            if not ret:

                continue


            frame = cv2.flip(

                frame,

                1

            )


            # -----------------------------------------------
            # MEDIAPIPE
            # -----------------------------------------------

            rgb = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB

            )


            results = holistic.process(

                rgb

            )


            # -----------------------------------------------
            # CHECK HAND
            # -----------------------------------------------

            hand_detected = (

                results.left_hand_landmarks

                is not None

                or

                results.right_hand_landmarks

                is not None

            )


            # -----------------------------------------------
            # EXTRACT FEATURES
            # -----------------------------------------------

            features = extract_features(

                results

            )


            sequence.append(

                features

            )


            # -----------------------------------------------
            # DRAW LANDMARKS
            # -----------------------------------------------

            if results.pose_landmarks:


                mp_draw.draw_landmarks(

                    frame,

                    results.pose_landmarks,

                    mp_holistic.POSE_CONNECTIONS

                )


            if results.left_hand_landmarks:


                mp_draw.draw_landmarks(

                    frame,

                    results.left_hand_landmarks,

                    mp_holistic.HAND_CONNECTIONS

                )


            if results.right_hand_landmarks:


                mp_draw.draw_landmarks(

                    frame,

                    results.right_hand_landmarks,

                    mp_holistic.HAND_CONNECTIONS

                )


            # -----------------------------------------------
            # RECORDING STATUS
            # -----------------------------------------------

            cv2.putText(

                frame,

                "RECORDING",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 0, 255),

                3

            )


            cv2.putText(

                frame,

                f"Gesture: {gesture}",

                (30, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f"Frame: "

                f"{frame_number + 1}/"

                f"{SEQUENCE_LENGTH}",

                (30, 130),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 0),

                2

            )


            # -----------------------------------------------
            # HAND STATUS
            # -----------------------------------------------

            if hand_detected:


                cv2.putText(

                    frame,

                    "Hand: DETECTED",

                    (30, 170),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 0),

                    2

                )


            else:


                cv2.putText(

                    frame,

                    "Hand: LOST",

                    (30, 170),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 0, 255),

                    2

                )


            cv2.imshow(

                "Sign Language Data Collection",

                frame

            )


            cv2.waitKey(

                50

            )


        # ====================================================
        # SAVE SEQUENCE
        # ====================================================

        if len(sequence) == SEQUENCE_LENGTH:


            sequence = np.array(

                sequence,

                dtype=np.float32

            )


            file_path = os.path.join(

                gesture_path,

                f"{sequence_number}.npy"

            )


            np.save(

                file_path,

                sequence

            )


            print()

            print(

                "Saved successfully:",

                file_path

            )


        else:


            print()

            print(

                "ERROR: Sequence was incomplete."

            )


            print(

                "Sequence was NOT saved."

            )


    # ========================================================
    # CLOSE CAMERA
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


    print()

    print("=" * 50)

    print(

        "DATA COLLECTION COMPLETED"

    )

    print("=" * 50)

    print(

        "Gesture:",

        gesture

    )

    print(

        "New sequences collected:",

        NO_OF_SEQUENCES

    )

    print(

        "Total sequences now:",

        get_existing_sequence_count(

            gesture

        )

    )

    print("=" * 50)


# ============================================================
# MAIN PROGRAM
# ============================================================

try:


    while True:


        # -----------------------------------------------
        # SHOW CURRENT DATASET
        # -----------------------------------------------

        show_dataset_status()


        # -----------------------------------------------
        # SELECT GESTURE
        # -----------------------------------------------

        selected_gesture = select_gesture()


        # -----------------------------------------------
        # EXIT
        # -----------------------------------------------

        if selected_gesture is None:


            print()

            print(

                "Exiting data collection."

            )


            break


        # -----------------------------------------------
        # COLLECT DATA
        # -----------------------------------------------

        collect_gesture_data(

            selected_gesture

        )


        # -----------------------------------------------
        # ASK WHETHER TO CONTINUE
        # -----------------------------------------------

        print()

        choice = input(

            "Collect another gesture? "

            "(Y/N): "

        ).strip().lower()


        if choice != "y":


            break


finally:


    holistic.close()


    cv2.destroyAllWindows()


print()

print("=" * 50)

print(

    "PROGRAM FINISHED"

)

print("=" * 50)