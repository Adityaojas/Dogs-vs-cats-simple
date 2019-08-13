import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from mymodule.dataset.simpledatasetloader import SimpleDatasetLoader
from mymodule.preprocessing.simplepreprocessor import SimplePreprocessor
from imutils import paths
import argparse

train_dataset = 'dogs-vs-cats/train'

train_paths = list(paths.list_images(train_dataset))
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(train_paths, verbose = 500)
data = data.reshape((data.shape[0], 3072))

print('Feature Matrix Size = {} MB'.format(data.nbytes/(1024*1024.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=.25, random_state=111)
model = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
model.fit(X_train, y_train)
print('calculating accuracy...')
acc = model.score(X_val, y_val)*100
print('accuracy: {}'.format(acc))


y_pred_val = model.predict(X_val)

cr = classification_report(y_val, y_pred_val)
cm = confusion_matrix(y_val, y_pred_val)

report = classification_report(y_val, y_pred_val, output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('dogs-vs-cats/knn/val_classification_report.csv', index = False)


# testing
test_dataset = 'dogs-vs-cats/test1'
test_paths = list(paths.list_images(test_dataset))
X_test = []
for (i, path) in enumerate(test_paths):
    image = cv2.imread(path)
    image = sp.preprocess(image)
    X_test.append(image)

X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], 3072))

model.fit(data, labels)

y_pred = model.predict(X_test)

ind = np.arange(1, len(test_paths)+1)
df = pd.DataFrame({'id': ind, 'label': y_pred})
df.to_csv('dogs-vs-cats/knn/predictions_knn.csv', index = False)
















