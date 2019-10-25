import os
from PIL import Image 
import numpy as np 
from data.BaseDataset import BaseDataset
from transforms.bounding_box import BBox 

import torch
import random
import torchvision.transforms as transforms

class VisDrone(BaseDataset):
	def __init__(self, opt):
		# read all the image path
		self.opt = opt
		self.img_size = opt.img_size
		self.dataroot = opt.dataroot
		self.img_paths = self.get_paths(self.dataroot, '.jpg')
		self.label_paths = self.replace(self.img_paths)

		self.max_object =100
		self.multiscale = not opt.nomultiscale
		self.normalized_labels = opt.normalized_labels
		self.min_size = opt.img_size - 3 * 32
		self.max_size = opt.img_size + 3 * 32
		# Non maximum merge
		# NMM()

	# our assumption here is that:
	# the GPU device on UAVs can be very limited
	# and we want to make the large images as small as possible
	# so we split the images into small parts and deal with it
	# def __getitem__(self, index):
	# 	img_path = self.img_paths[index]
	# 	if not os.path.exists(img_path):
	# 		assert RuntimeError("Image path not find")
	# 	label_path = self.label_paths[index]
	# 	if not os.path.exists(label_path):
	# 		assert RuntimeError("label path not find")

	# 	img = Image.open(img_path).convert('RGB')
	# 	bboxes = BBox.from_visDrone(np.loadtxt(label_path, delimiter=',').reshape(-1,8), img.size)
	# 	targets = bboxes.non_max_merge(box_size=540, iou_thresh=0.5).to_tensor()
	def __getitem__(self, index):
		img_path = self.img_paths[index]
		if not os.path.exists(img_path):
			assert RuntimeError('Image path not find')

		label_path = self.label_paths[index]
		if not os.path.exists(label_path):
			assert RuntimeError('Label path not find')

		img = Image.open(img_path).convert('RGB')
		bboxes = BBox.from_visDrone(np.loadtxt(label_path, delimiter=',').reshape(-1,8), img.size)

		# target 
		# still need the NMM to get the cluster
		
		target = bboxes.to_tensor()
		print(target[0])
		img = transforms.ToTensor()(img)

		# handle gray scale channels
		if len(img.shape) != 3:
			img = img.unsqueeze(0)
			img = img.expand((3, img.shape[1:]))

		# add pad
		_, h, w = img.shape
		h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
		img, pad = self.pad_to_square(img, 0)
		_, padded_h, padded_w = img.shape

		# we only need to know whether there is a cluster or not
		x1 = w_factor * (target[:, 1] - target[:, 3] / 2)
		x2 = w_factor * (target[:, 1] + target[:, 3] / 2)
		y1 = h_factor * (target[:, 2] - target[:, 4] / 2)
		y2 = h_factor * (target[:, 2] + target[:, 4] / 2)

		x1 += pad[0]
		x2 += pad[0]
		y1 += pad[2]
		y2 += pad[2]

		target[:, 0] = 1
		# boxes format is (x1,y1, x2, y2)
		target[:, 1] = ((x1 + x2) / 2) / padded_w
		target[:, 2] = ((y1 + y2) / 2) / padded_h
		target[:, 3] *= w_factor / padded_w
		target[:, 4] *= h_factor / padded_h

		img_idx = torch.zeros(target.size(0), 1)
		target = torch.cat((img_idx, target), -1)

		return img, target

	def __len__(self):
		return len(self.img_paths)

	def name(self):
		return 'visDroneDataset'

	def collate_fn(self, batch):
		imgs, targets = list(zip(*batch))
		count = 0
		for i, bboxes in enumerate(targets):
			bboxes[:, 0] = i

		if self.multiscale:
			self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

		imgs = torch.stack([self.resize(img, self.img_size) for img in imgs])
		targets = torch.cat(targets, 0)

		return imgs, targets

	def get_paths(self, path, suffix='.jpg'):
		image_paths = []
		for dirpath, subdirs, files in os.walk(path):
			for file in files:
				if suffix in file:
					image_paths.append('/'.join([dirpath, file]))
		return image_paths

	def replace(self, paths, image='images/', label='annotations/'):
		label_path = []
		for path in paths:
			# replace .jpg to .txt
			p = path[:-4]+'.txt'
			p = p.replace(image, label)
			label_path.append(p)
		return label_path