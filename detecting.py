from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Imena klasa u istom redosledu kao kod treniranja
class_names = ['Avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation', 'Greenstick fracture', 'Hairline Fracture', 'Impacted fracture', 'Longitudinal fracture', 'Oblique fracture', 'Pathological fracture', 'Spiral Fracture']

# Glavni test direktorijum (koji sadrži foldere sa slikama)
root_test_dir = 'data2/test/'

# Učitaj model
model = load_model('bone_fracture_classifier.keras')

# Funkcija za učitavanje slika iz svih podfoldera
def preprocess_images_from_directory_recursively(root_dir):
    img_arrays = []
    img_paths = []
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(subdir, filename)
                img = image.load_img(img_path, target_size=(128, 128))
                img_array = image.img_to_array(img)
                img_array = img_array / 255.0  # Normalizacija
                img_arrays.append(img_array)
                img_paths.append(img_path)
    return np.array(img_arrays), img_paths

# Pripremi slike
processed_images, img_paths = preprocess_images_from_directory_recursively(root_test_dir)

# Predikcija
if len(processed_images) == 0:
    print("Nema validnih slika za predikciju u folderu.")
else:
    predictions = model.predict(processed_images)

    # Prikaz rezultata
    for img_path, prediction in zip(img_paths, predictions):
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = prediction[predicted_class_index]
        print(f"{img_path} → {predicted_class_name} ({confidence:.2f})")
