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

def Kmeans(data, labels, k): # perform k means clustering
    centroids = data[np.random.choice(data.shape[0], k, replace=False)] # pick random centroids
    closest = np.zeros(len(data))

    equal = False
    while not equal:
        newCentroids = []
        distances = distance(data, centroids)
        newClosest = [np.argmin(x) for x in distances]

        for i in range(len(centroids)):
            members = [data[j] for j in range(len(newClosest)) if newClosest[j] == i]

            if len(members) != 0:
                newCentroids.append(np.sum(members, axis=0) / len(members))

        equal = np.array_equal(closest, newClosest)
        closest = newClosest
        centroids = np.array(newCentroids)

    centroids, centroidsLabels = majority(centroids, closest, labels)
    
    return centroids, centroidsLabels

def distance(data, centroids): # determine distances for all vectors with euclidean formula
    distance = np.zeros((data.shape[0], centroids.shape[0]))

    for i in range(data.shape[0]):
        distance[i] = np.sqrt(np.sum((np.square(centroids - data[i])), axis=1))
    
    return distance

def majority(centroids, closest, labels): # get majority class label vote and and finalize centroid locations
    centroidLabels = []
    for i in range(len(centroids)):
        memberLabels = [labels[j] for j in range(len(closest)) if closest[j] == i]

        if (len(memberLabels) == 0):
            np.delete(centroids, i, axis=0)

        else:
            uniqueLabels, counts = np.unique(memberLabels, return_counts=True)
            centroidLabels.append(uniqueLabels[np.argmax(counts)])

    return centroids, centroidLabels

# test the training data against the testing data and return the
# number of correct classifications
def test(testingData, testingLabels, centroids, centroidsLabels, correct):
    distances = distance(testingData, centroids)
    closest = [np.argmin(x) for x in distances]

    for i in range(len(closest)):
        label = testingLabels[i]
        index = closest[i]

        if label == centroidsLabels[index]:
            correct += 1

    return correct

def main():
    np.random.seed(int(sys.argv[1]))
    k = int(sys.argv[2]) # number of centroids

    trainingFile = sys.argv[3]
    trainingData = np.loadtxt(trainingFile)
    if len(trainingData.shape) < 2:
        trainingData = np.array([trainingData])

    testingFile = sys.argv[4]
    testingData = np.loadtxt(testingFile)
    if len(testingData.shape) < 2:
        testingData = np.array([testingData])

    # get centroids and number of votes for majority classification by running k means
    centroids, centroidsLabels = Kmeans(trainingData[:, :-1], trainingData[:, -1], k)

    # get number of correct classifications
    correct = test(testingData[:, :-1], testingData[:, -1:], centroids, centroidsLabels, 0)
    print(correct)

if __name__ == "__main__":
    main()