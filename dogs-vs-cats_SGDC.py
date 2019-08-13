from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mymodule.dataset.simpledatasetloader import SimpleDatasetLoader
from mymodule.preprocessing.simplepreprocessor import SimplePreprocessor
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import cv2
import numpy as np
import pandas as pd

train_dataset = 'dogs-vs-cats/train'
train_paths = list(paths.list_images(train_dataset))

sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(train_paths, verbose = 500)
data = data.reshape((data.shape[0], 3072))


le = LabelEncoder()
labels = le.fit_transform(labels)

(X_train, X_val, y_train, y_val) = train_test_split(data, labels, test_size = 0.25, random_state = 111)

regs = [None, 'l1', 'l2', 'ElasticNet']
accuracies = []

for reg in (None, 'l1', 'l2', 'ElasticNet'):
    print("training using '{}' regularizer".format(reg))
    model = SGDClassifier(loss = 'log', penalty = reg, learning_rate = 'constant', tol = 1e-3, eta0 = 0.01, random_state = 111)
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    print("accuracy with '{}' regularizer: {}".format(reg, acc))
    accuracies.append(acc*100)
    
    
best_reg = regs[accuracies.index(max(accuracies))]

print('best regularizer: {} \n and it\'s accuracy is: {}'.format(best_reg, max(accuracies)))


#testing

test_dataset = 'dogs-vs-cats/test1'
test_paths = list(paths.list_images(test_dataset))
X_test = []
for (i, path) in enumerate(test_paths):
    image = cv2.imread(path)
    image = sp.preprocess(image)
    X_test.append(image)
    

X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], 3072))

print('training on whole training dataset using "{}" regularizer'.format(best_reg))

model_fin = SGDClassifier(loss = 'log', penalty = best_reg, learning_rate = 'constant', tol = 1e-3, eta0 = 0.01, random_state = 111)
model_fin.fit(data, labels) 
y_pred = model.predict(X_test)


ind = np.arange(1, len(test_paths)+1)
df = pd.DataFrame({'id': ind, 'label': y_pred})
df.to_csv('dogs-vs-cats/SGDC/predictions_SGDC.csv', index = False)
