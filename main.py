import os
import pickle

from skimage.color import gray2rgb, rgba2rgb
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report



# prepare data
input_dir = r'D:\downloads\gender_model'
categories = ['Male', 'Female']
i=0
data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        i =i + 1
        print(i)
        img_path = os.path.join(input_dir, category, file)
        try:
            img = imread(img_path)

            # Handle different image formats
            if len(img.shape) == 2:  # Grayscale
                img = gray2rgb(img)
            elif len(img.shape) == 3:
                if img.shape[2] == 4:  # RGBA
                    img = rgba2rgb(img)
                elif img.shape[2] > 4:  # Unusual format, take first 3 channels
                    img = img[:, :, :3]
            elif len(img.shape) > 3:  # 4D or higher (e.g., animated GIF)
                print(f"Skipping {file}: unsupported format with shape {img.shape}")
                continue

            # Now resize to 15x15
            img = resize(img, (15, 15))
            data.append(img.flatten())
            labels.append(category_idx)

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters, verbose=2, n_jobs=-1)

print("Starting training...")
grid_search.fit(x_train, y_train)
print("Training complete!")

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

# Get predictions
y_prediction = best_estimator.predict(x_test)

# Overall accuracy
score = accuracy_score(y_prediction, y_test)
print('{}% of samples were correctly classified'.format(str(score * 100)))
print('nice')


# Class-specific accuracy
print("\n--- Per-Class Accuracy ---")
conf_matrix = confusion_matrix(y_test, y_prediction)

for idx, category in enumerate(categories):
    # For each class, calculate: correct predictions / total actual instances
    total_actual = conf_matrix[idx].sum()
    correct_predictions = conf_matrix[idx, idx]
    class_accuracy = (correct_predictions / total_actual) * 100 if total_actual > 0 else 0

    print(f'{category}: {class_accuracy:.2f}% ({correct_predictions}/{total_actual} samples)')

# Detailed classification report
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_prediction, target_names=categories))

# Show confusion matrix
print("\n--- Confusion Matrix ---")
print("Rows: Actual, Columns: Predicted")
print(f"{'':12} {categories[0]:12} {categories[1]:12}")
for idx, category in enumerate(categories):
    print(f"{category:12} {conf_matrix[idx, 0]:12} {conf_matrix[idx, 1]:12}")

