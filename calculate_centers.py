from os import listdir
from os.path import isfile, join
import argparse
import matplotlib.pyplot as plt

import numpy as np 
import sys
import os
import shutil
import random
import math

# ISODATA
# Based on k-means, split or merge center points


def euclDistance(vector1, vector2):
	return np.sqrt(np.sum(np.power(vector2-vector1, 2)))

def initCentroids(boxes, k):
	numSamples, dim = dataSet.shape
	centroids = np.zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids 

def kmeans(dataSet, k):
	numSamples = dataSet.shape[0]
	clusterAssment = np.mat(np.zeros((numSamples, 2)))
	clusterChanged = True
	# step 1: init centroids
	centroids = initCentroids(dataSet, k)
	while clusterChanged:
		clusterChanged = False
		# for each sample
		for i in range(numSamples):
			minDist = 1000000.0
			minIndex = 0
			# for each centroid
			# step2: find the centroid who is cloest
			for j in range(k):
				distance = euclDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist = distance
					minIndex = j
			# step3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist ** 2
		# step4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] =  np.mean(pointsInCluster, axis=0)
	return centroids, clusterAssment

def showCluster(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	
	for i in range(numSamples):
		markIndex = int(clusterAssment[i, 0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

	for i in range(k):
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
	plt.show()

def find_hw(label, k, centroids, clusterAssment):
	numSamples, dim = label.shape
	# left x and right x
	# left y and right y
	size = [[100000, 100000, 0,0] for i in range(k)]
	for i in range(numSamples):
		markIndex = int(clusterAssment[i, 0])
		left_x, left_y, right_x, right_y = label[i,0], label[i,1], label[i,0] + label[i,2], label[i,1] + label[i,3]
		#print(left_x, left_y, right_x, right_y)
		if size[markIndex][0] > left_x:
			size[markIndex][0] = left_x
		if size[markIndex][1] > left_y:
			size[markIndex][1] = left_y
		if size[markIndex][2] < right_x:
			size[markIndex][2] = right_x
		if size[markIndex][3] < right_y:
			size[markIndex][3] = right_y
	return size

def IOU(box, clusters):
	x = np.minimun(clusters[:, 0], box[0])
	y = np.minimun(clusters[:, 1], box[1])

	intersection = x * y
	box_area = clusters[:, 0] * clusters[:, 1]
	iou_ = intersection / (box_area + cluster_area - intersection)

	return iou_

def avg_iou(boxes, clusters):
	return np.mean([np.max(IOU(boxes[i], clusters)) for i in range(boxes.shape[0])])

def translate_boxes(boxes):
	new_boxes = boxes.copy()
	for row in range(new_boxes.shape[0]):
		new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
		new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
	return np.delete(new_boxes, [0,1], axis=1)

def kmeans_iou(boxes, k, dist=np.median):
	rows = boxes.shape[0]
	distances = np.empty((rows, k))
	last_clusters = np.zeros((rows,))

	clusters = boxes[np.random.choice(rows, k, replace=False)]

	while True:
		for row in range(rows):
			distances[row] = 1 - IOU(boxes[row], clusters)

		nearest_clusters = np.argmin(distances, axis=1)
		if (last_clusters == nearest_clusters).all():
			break

		for cluster in range(k):
			clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

		last_clusters = nearest_clusters
	return clusters

def get_paths(filepath):
	for dirpath, subdir, files in os.walk(filepath):
		label_paths = []
		for file in files:
			if '.txt' in file:
				label_paths.append('/'.join([dirpath, file]))
	return label_paths

from PIL import Image
from PIL.ImageDraw import Draw

if __name__ == '__main__':
	filepath = 'datasets/visDrone/train/annotations'
	gen_filepath = 
	label_paths = get_paths(filepath)
	nums = 0
	for label_path in label_paths:
		imgpath = label_path.replace('.txt', '.jpg')
		imgpath = imgpath.replace('annotations', 'images')
		print(imgpath, label_path)
		img = Image.open(imgpath).convert('RGB')
		draw = Draw(img)

		label = np.loadtxt(label_path, delimiter=',').reshape(-1, 8)
		center_x = label[:, 0] + (label[:, 2]) / 2
		center_y = label[:, 1] + (label[:, 3]) / 2
		centers = list()
		for i in range(len(center_x)):
			#print(center_x[i], center_y[i])
			centers.append([center_x[i], center_y[i]])

		dataSet = np.mat(centers)
		label = np.mat(label)
		k = 4
		centroids, clusterAssment = kmeans(dataSet, k)
		#showCluster(label, k, centroids, clusterAssment)
		size = find_hw(label, k, centroids, clusterAssment)
		#for i in range(k):
		#	print(centroids[i,0], centroids[i,1], size[i][0], size[i][1], size[i][2], size[i][3]) 
		for s in size:
			x1, y1, x2, y2 = s
			draw.rectangle([(x1,y1), (x2,y2)], outline=(255,255,0))
		img.save('vis/' + str(nums) + '.jpg')
		nums += 1
		
		