import os
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from skimage import io
from src.model.model import get_unet
from src.data.data_processing import load_and_preprocess_test_data

def predict(model, mean, std):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_and_preprocess_test_data()

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for k in range(len(imgs_mask_test)):
        a = rescale_intensity(imgs_test[k][:,:,0], out_range=(-1,1))
        b = (imgs_mask_test[k][:,:,0]).astype('uint8')
        io.imsave(os.path.join(pred_dir, str(k) + '_pred.png'), mark_boundaries(a,b))

    return imgs_mask_test