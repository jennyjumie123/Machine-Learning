import os
import math
import time
import numpy as np
import pandas as pd
from sklearn import svm
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix, accuracy_score
import streamlit as st
#import keras
from PIL import Image, ImageOps
import numpy as np


target = []
images = []
flat_data = []
DATADIR = 'C:/Users/hp/Documents/Project_Implementation/test'
CATEGORIES = ['COVID','Normal']
for category in CATEGORIES:
    class_num = CATEGORIES.index(category) # label Encoding the values
    path = os.path.join(DATADIR, category) # create path to use  all the images
    print(path)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img))
        # print(img_array.shape)
        #plt.imshow(img_array)
        img_resize = resize(img_array, (150, 150, 3)) # NOrmalizes the value from 0 to 1
        flat_data.append(img_resize.flatten())
        images.append(img_resize)
        target.append(class_num)
flat_data = np.array(flat_data)
taget = np.array(target)
images = np.array(images)

x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size = 0.3, random_state = 109)
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred= clf.predict(x_test)
print("The predicted Data is :")
print(y_pred)

from sklearn.datasets import make_blobs
from sklearn import svm
support_vector_indices = clf.support_
print(support_vector_indices)

support_vectors_per_class = clf.n_support_
print(support_vectors_per_class)

support_vectors = clf.support_vectors_
acc = round(accuracy_score(y_pred,y_test)*100,0)


import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import pickle
pickle.dump(clf,open('img_model.p','wb'))

st.title('Chest X-RAY Image Diagnosis')
uploaded_file = st.file_uploader("Upload an X-ray image...", type=["png","jpg","jpeg"])

model = pickle.load(open('img_model.p','rb'))

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded image')

    if st.button('DIAGNOSE'):
        CATEGORIES = ['COVID', 'Normal']
        st.write('Result...')
        flat_data = []
        img = np.array(img)
        img_resized = resize(img, (150, 150, 3))
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)
        y_out = model.predict(flat_data)
        y_out = CATEGORIES[y_out[0]]
        st.title(f'The image has been diagnosed as {y_out}')
