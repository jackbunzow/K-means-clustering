#!/usr/bin/env python3

# Jack Bunzow
# Dr. Phillips
# 12/3/19
# Project 4
# This program performs K-means clustering by taking in a random seed, number of clusters (k), training data,
# and testing data. Classification labels of each cluster are assigned by majority vote. The program outputs the 
# number of correct classifications (0-10) that have been made when the clusters are compared with the testing data.

import sys, math, random
import numpy as np

# perform k means clustering
def Kmeans(k, data, labels, centroids):
    closest = []
    equal = 0
    while equal <= 2: # while centroids keep moving
        newCentroids = []
        distances = distance(data, centroids)
        newClosest = [np.argmin(x) for x in distances]

        for i in range(len(centroids)):
            closestLoc = [data[j] for j in range(len(newClosest)) if newClosest[j] == i]

            if len(closestLoc) != 0:
                # average of the vectors along each column to make new centroid
                newLoc = np.sum(closestLoc, axis=0) / len(closestLoc)
                newCentroids.append(newLoc)

        # are the new centroids and old centroids in the same location?
        if np.array_equal(closest, newClosest):
            equal += 1

        centroids = np.array(newCentroids)
        closest = newClosest

    centroids, centroidsLabels = majority(centroids, closest, labels)

    return centroids, centroidsLabels

# determine distances for all vectors with euclidean formula
def distance(data, centroids):
    distance = np.zeros((data.shape[0], centroids.shape[0]))

    for i in range(data.shape[0]):
        distance[i] = np.sqrt(np.sum((np.square(centroids - data[i])), axis=1))
    
    return distance

# assign class labels based on majority vote and and finalize centroid locations
def majority(centroids, closest, labels):
    centroidLabels = []
    for i in range(len(centroids)):
        closestLabels = [labels[j] for j in range(len(closest)) if closest[j] == i]

        if (len(closestLabels) != 0):
             # get all unique labels and the count of the accurance of each of those labels
            uniqueLabels, counts = np.unique(closestLabels, return_counts=True)
            centroidLabels.append(uniqueLabels[np.argmax(counts)]) # add the label with the max vote to the list
        else:
            np.delete(centroids, i, axis=0)

    return centroids, centroidLabels

# test the training data against the testing data and print the
# number of correct classifications
def test(testingData, testingLabels, centroids, centroidsLabels, correct):
    distances = distance(testingData, centroids) # distances from the centroids to the testing data
    closest = [np.argmin(x) for x in distances] # list of all closest centroids from the testing data

    # for all the closest centroids check if the centroid label and testing label are the same
    for i in range(len(closest)):
        label = testingLabels[i]
        index = closest[i]

        if label == centroidsLabels[index]:
            correct += 1

    print(correct)
    return

def main():
    np.random.seed(int(sys.argv[1]))
    k = int(sys.argv[2]) # number of centroids

    trainingData = np.loadtxt(sys.argv[3])
    if len(trainingData.shape) < 2:
        trainingData = np.array([trainingData])

    testingData = np.loadtxt(sys.argv[4])
    if len(testingData.shape) < 2:
        testingData = np.array([testingData])

    #pick random k number of centroids
    centroids = trainingData[:, :-1][np.random.choice(trainingData[:, :-1].shape[0], k, replace=False)]
    # get centroids and number of votes for majority classification by running k means
    centroids, centroidsLabels = Kmeans(k, trainingData[:, :-1], trainingData[:, -1], centroids)

    # get number of correct classifications
    test(testingData[:, :-1], testingData[:, -1:], centroids, centroidsLabels, 0)


if __name__ == "__main__":
    main()