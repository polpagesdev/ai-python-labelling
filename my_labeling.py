__authors__ = '1494769'
__group__ = 'DL.15'

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
import time
# import cv2

start_time = time.time()

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))


def Retrieval_by_color(images, labels, colors):
    hits = np.array([])
    for index, image in enumerate(images):
        add_hit = 0
        for color in colors:
            if color in labels[index]:
                add_hit += 1
        if add_hit == len(colors):
            hits = np.append(hits, 1)
        else:
            hits = np.append(hits, 0)

    indexes = np.where(hits == 1)
    return images[indexes]


def Retrieval_by_shape(images, labels, shapes):
    hits = np.array([])
    for index, image in enumerate(images):
        if shapes == labels[index]:
            hits = np.append(hits, 1)
        else:
            hits = np.append(hits, 0)

    indexes = np.where(hits == 1)
    return images[indexes]


def Retrieval_combined(images, color_labels, shape_labels, colors, shapes):
    hits = np.array([])
    for index, image in enumerate(images):
        add_hit = 0
        for color in colors:
            if color in color_labels[index]:
                add_hit += 1
        if add_hit == len(colors) and shapes == shape_labels[index]:
            hits = np.append(hits, 1)
        else:
            hits = np.append(hits, 0)

    indexes = np.where(hits == 1)
    return images[indexes]


def Kmean_statistics(kmeans, kmax):
    kmeans.K = 2
    iterations = np.array([])
    exec_time = np.array([])
    wcd = np.array([])
    while kmeans.K < kmax:
        init_time = time.time()
        kmeans._init_centroids()
        kmeans.fit()
        exec_time = np.append(exec_time, time.time() - init_time)
        wcd = np.append(wcd, kmeans.whitinClassDistance())
        iterations = np.append(iterations, kmeans.iter)
        kmeans.K += 1

    for i in range(len(iterations)):
        print('Dades per a k = %s:' % (i+2))
        print('Iteracions: %s' % iterations[i], '| WCD: %s' % wcd[i], '| Temps fins a convergir: %s segons' % exec_time[i])
        print('-------------------------------------------------------------------------------------------------------')


def Get_shape_accuracy(shape_labels, gt_shapes):
    rate = 0
    for index, shape in enumerate(gt_shapes):
        if shape_labels[index] == shape:
            rate += 1

    print('Percentatge de etiquetes correctes: %s%%' % (rate/len(gt_shapes)*100))


def Get_color_accuracy(color_labels, gt_colors):
    rate = 0
    total_len = 0
    for index, colours in enumerate(gt_colors):
        total_len += len(colours)
        for label in color_labels[index]:
            if label in colours:
                rate += 1

    print('Percentatge de colors correctes: %s%%' % (rate / total_len * 100))


# ________________SetUp____________________________________________________
img_num = 5
k = 3
test_images = test_imgs[:img_num]
train_images = train_imgs[:img_num]
color_labels = np.array([])
shape_labels = np.array([])
gt_shapes = train_class_labels[:img_num]
gt_colors = test_color_labels[:img_num]
for img in range(img_num):
    km = Kmeans.KMeans(test_images[img], k)
    km.find_bestK(4)
    color_labels = np.append(color_labels, Kmeans.get_colors(km.centroids))
knn = KNN.KNN(train_images, gt_shapes)
shape_labels = np.append(shape_labels, knn.predict(test_images, 3))
color_labels = color_labels.reshape(img_num, k)
colors = np.array(['Black', 'White'])
shapes = np.array(['Heels'])
# _________________________________________________________________________


# ________________Testing__________________________________________________
"""color_test = Retrieval_by_color(test_images, color_labels, colors)
visualize_retrieval(color_test, 10, None, None, 'Color Test')"""
"""shape_test = Retrieval_by_shape(test_images, shape_labels, shapes)
visualize_retrieval(shape_test, 10, None, None, 'Shape test')"""
"""combined_test = Retrieval_combined(test_images, color_labels, shape_labels, colors, shapes)
visualize_retrieval(combined_test, 10, None, None, 'Combined test')"""
"""Kmean_statistics(km, 10)"""
"""Get_shape_accuracy(shape_labels, gt_shapes)"""
"""Get_color_accuracy(color_labels, gt_colors)"""
# _________________________________________________________________________


# ________________Timing___________________________________________________
print("\n""Execution time: %s seconds" % (time.time() - start_time))
# _________________________________________________________________________
