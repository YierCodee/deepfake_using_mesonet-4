import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('model/mesonet_deepfake.h5', compile=False)

def preprocess_image(image_path, target_size=(256,256)):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ↓ INI HARUS DI LUAR FUNGSI (tidak ada indentasi/spasi di depan)
img = preprocess_image('real2.jpg')
pred = model.predict(img)[0][0]
print(f"Prediksi mentah: {pred:.4f}")