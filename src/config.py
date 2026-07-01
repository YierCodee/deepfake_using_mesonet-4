MODEL_PATH = "model/mesonet_deepfake.h5"
IMG_SIZE = (256, 256)
THRESHOLD = 0.5
MAX_BATCH = 100

# Model ini ternyata dilatih dengan label terbalik (REAL=1, FAKE=0),
# jadi output mentahnya perlu di-invert agar ≥ 0.5 → FAKE.
INVERT_PROBABILITY = True

# Kustomisasi preprocessing
# Ubah ke True setelah Anda mengisi fungsi face detection / mean subtraction di preprocessing.py
FACE_DETECT_ENABLED = False
MEAN_SUBTRACTION_ENABLED = False
TTA_ENABLED = False
