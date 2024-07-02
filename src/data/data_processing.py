import numpy as np
from skimage.transform import resize
from src.config.config import IMG_ROWS, IMG_COLS
from src.data.data_loader import load_train_data, load_test_data

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], IMG_ROWS, IMG_COLS), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (IMG_COLS, IMG_ROWS), preserve_range=True)
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def load_and_preprocess_train_data():
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    return imgs_train, imgs_mask_train

def load_and_preprocess_test_data():
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    return imgs_test, imgs_id_test