import os
import cv2
import shutil
import random

def file_distrubution(splitRatio=0.8):
    raw_data_dir = 'data/raw/'
    processed_data_dir = 'data/processed/'
    train_data_dir = 'data/processed/train/'
    val_data_dir = 'data/processed/val/'
    image_size = (400, 400)

    split_ratio = splitRatio

    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(val_data_dir, exist_ok=True)

    for person in os.listdir(raw_data_dir):
        person_raw_path = os.path.join(raw_data_dir, person)
        
        images = os.listdir(person_raw_path)
        
        random.shuffle(images)
        
        split_index = int(len(images) * split_ratio)
        
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        os.makedirs(os.path.join(train_data_dir, person), exist_ok=True)
        os.makedirs(os.path.join(val_data_dir, person), exist_ok=True)
        
        for image_name in train_images:
            img_path = os.path.join(person_raw_path, image_name)
            image = cv2.imread(img_path)
            
            resized_image = cv2.resize(image, image_size)
            normalized_image = resized_image / 255.0
            
            processed_img_path = os.path.join(train_data_dir, person, image_name)
            cv2.imwrite(processed_img_path, normalized_image * 255)
        
        for image_name in val_images:
            img_path = os.path.join(person_raw_path, image_name)
            image = cv2.imread(img_path)
            
            resized_image = cv2.resize(image, image_size)
            normalized_image = resized_image / 255.0
            
            processed_img_path = os.path.join(val_data_dir, person, image_name)
            cv2.imwrite(processed_img_path, normalized_image * 255)

    print("Data preprocessing and splitting completed.")


file_distrubution()