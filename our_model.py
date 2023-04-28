import numpy as np
import pandas as pd
import os

from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import  plot_confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model
import timeit


def detect_images():
    start = timeit.default_timer()
    # Load the model and its weights from a .h5 file
    model = load_model('FinalModel.h5')

    df = pd.read_csv('test/cancer/cancer_metadata.csv')
    #df.head()
    #df.info()
    print("shape = ", df.shape)

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # Map the image path to image id
    base_skin_dir = os.path.join("test/cancer/")
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

    # Create additional features by mapping image id to the dictionaries
    df['path'] = df['image_id'].map(imageid_path_dict.get)
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
    df['age'].fillna((df['age'].mean()), inplace=True)
    df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x)))

    print(df.head())

    # Drop the target feature
    features=df.drop(columns=['cell_type_idx'],axis=1)
    target=df['cell_type_idx']

    # Test train split and conversions
    x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)

    x_train = np.asarray(x_train_o['image'].tolist())
    x_test = np.asarray(x_test_o['image'].tolist())

    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)

    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)

    x_train = (x_train - x_train_mean)/x_train_std
    x_test = (x_test - x_test_mean)/x_test_std

    y_train = to_categorical(y_train_o, num_classes = 7)
    y_test = to_categorical(y_test_o, num_classes = 7)

    # Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
    x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))


    # Predict
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Evaluate the model on the test data
    #loss, accuracy = model.evaluate(x_test, y_test)

    print("Predicting results ....")
    print(y_pred_classes)

    # Print the test loss and accuracy
    """ print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    """

    stop = timeit.default_timer()
    print(f'Time: {stop - start}')
    return y_pred_classes
