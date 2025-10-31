input_dir = r'D:\downloads\gender_model'
categories = ['Male', 'Female']
i=0
data, labels = [], []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        i += 1
        print(i)
        img_path = os.path.join(input_dir, category, file)
        try:
            img = imread(img_path)

            # 2.1 Κανονικοποίηση καναλιών
            if len(img.shape) == 2:          # Γκρι εικόνα (H, W)
                img = gray2rgb(img)          # -> (H, W, 3)
            elif len(img.shape) == 3:
                if img.shape[2] == 4:        # RGBA (H, W, 4)
                    img = rgba2rgb(img)      # -> (H, W, 3)
                elif img.shape[2] > 4:       # περίεργα format
                    img = img[:, :, :3]      # κρατά τα 3 πρώτα κανάλια
            elif len(img.shape) > 3:         # π.χ. animated GIF
                print(f"Skipping {file}: unsupported format {img.shape}")
                continue

            # 2.2 Επαναδειγματοληψία (resize) και flatten
            img = resize(img, (15, 15))      # -> πολύ μικρό 15x15
            data.append(img.flatten())       # (15*15*3,) = 675 χαρακτηριστικά
            labels.append(category_idx)

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

data = np.asarray(data)
labels = np.asarray(labels)
