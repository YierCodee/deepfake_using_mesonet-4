import numpy as np
from PIL import Image
from .config import IMG_SIZE, FACE_DETECT_ENABLED, MEAN_SUBTRACTION_ENABLED, TTA_ENABLED


def load_image(image_file):
    """Load image from file path (str/Path) or file-like object."""
    return Image.open(image_file)


def preprocess_image(
    img,
    target_size=None,
    face_detect=None,
    mean_subtract=None,
):
    """
    Preprocess a single PIL image for MesoNet inference.

    1. Convert to RGB
    2. [Placeholder] Face detection + crop
    3. Resize to target_size
    4. [Placeholder] Mean subtraction
    5. Normalise to [0, 1]
    """
    if target_size is None:
        target_size = IMG_SIZE
    if face_detect is None:
        face_detect = FACE_DETECT_ENABLED
    if mean_subtract is None:
        mean_subtract = MEAN_SUBTRACTION_ENABLED

    img = img.convert("RGB")

    # TODO: Face detection + crop
    # if face_detect:
    #     from your_face_detection_module import detect_and_crop
    #     img = detect_and_crop(img)

    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # TODO: Mean subtraction
    # if mean_subtract:
    #     MEAN = np.array([0.485, 0.456, 0.406])
    #     img_array -= MEAN

    return np.expand_dims(img_array, axis=0)


def preprocess_batch(images, target_size=None, face_detect=None, mean_subtract=None):
    """Preprocess a list of PIL images into a single batch array."""
    batch = [
        preprocess_image(img, target_size, face_detect, mean_subtract)
        for img in images
    ]
    return np.vstack(batch)


# TODO: Test Time Augmentation wrapper
# def predict_with_tta(model, img_array, flips=('none', 'horizontal')):
#     preds = []
#     for flip in flips:
#         aug = np.fliplr(img_array[0]) if flip == 'horizontal' else img_array[0]
#         preds.append(float(model.predict(aug[np.newaxis, ...], verbose=0)[0][0]))
#     return np.mean(preds)
