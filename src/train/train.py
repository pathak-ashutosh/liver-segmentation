import numpy as np
from keras.callbacks import ModelCheckpoint
from src.model.model import get_unet
from src.data.data_processing import load_and_preprocess_train_data

def train_model():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_and_preprocess_train_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    
    # save mean and std
    np.save('mean.npy', mean)
    np.save('std.npy', std)
    
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('../../models/weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=10, epochs=20, verbose=1, shuffle=True,
                        validation_split=0.2, callbacks=[model_checkpoint])
    
    return model, history, mean, std

if __name__ == '__main__':
    train_model()