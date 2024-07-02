from keras import backend as K

K.set_image_data_format('channels_last')

IMG_ROWS = int(512/2)
IMG_COLS = int(512/2)
SMOOTH = 1.