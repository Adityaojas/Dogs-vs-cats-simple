# cats-vs-dogs-Kaggle-solutions
This repo consists of 2 simple approaches to Kaggle's Dogs vs. cats challenge

mymodule consists of the classes and modules I have used in the subsequent approaches

Download the dataset from the following link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
Place the train and test folders in the dogd-vs-cats directory as it is. I have further split the train directory into two directroies namely 'cat' and 'dog' to help my preprocessing. All the cat images would be inside the directory: dogs-vs-cats/train/cat/ and dog images would be in: dogs-vs-cats/train/dog/

The first approach is a simple knn approach. It cannot even be termed as a learning algorithm as it just calculates the distances between the data points. I used it just because the dataset was small, and I wanted to test how it performs. The validation accuracy comes to be 58% which is just slightly better than random guessing. The Kaggle submission fetched a loss score of 17.32, which is of the league of the bottom 10% submissions. Overall, it is a very computationally wasteful and useless algorithm in the case of Image Classification.

SGDC/ Stochastic Gradient Descent Classifier is the second approach. It is also a Machine Learning approach rather than a Deep Learning one. The main idea is to test the performance of each of the different regularizers, and choose the best one to make the final solution. l2 (weight decay) regularizer comes out to be the best of the lot, but the overall performance is still the same as that of the previous one. The final loss score comes to be again 17.2, which is of the bottom league. Deep Learning can get way better accuracies.
