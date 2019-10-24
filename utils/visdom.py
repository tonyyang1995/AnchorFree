import torch

from PIL import Image, ImageFont
from PIL.ImageDraw import Draw

import random
import os
import visdom

import numpy as np 
import torchvision.transforms as transforms

class Visualizer():
	def __init__(self, opt):
		self.opt = opt
		self.ncols = opt.ncols
		self.log_root = os.path.join(opt.checkpoint_dir, opt.name)
		if not os.path.exists(self.log_root):
			os.makedirs(self.log_root)

		self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
		with open(self.log_name, 'a') as log_file:
			log_file.write('============================== Train Loss =================================\n')

		self.viz = visdom.Visdom()

	def plot_rawdata_target(self, imgs, targets):
		toImg = transforms.ToPILImage()
		for i, img in enumerate(imgs):
			img = toImg(img)
			ori_w, ori_h = img.size
			img = img.resize((270, 270))
			img_w, img_h = img.size
			draw = Draw(img)
			gt = list()
			for j, tar in enumerate(targets):
				if tar[0] == i:
					gt.append(targets[j])
			for idx, cls, cx, cy, w, h in gt:
				#print(x1,y1,x2,y2)
				x1 = int(float(cx) * img_w - float(w) * img_w / 2)
				y1 = int(float(cy) * img_h - float(h) * img_h / 2)
				x2 = int(float(cx) * img_w + float(w) * img_w / 2)
				y2 = int(float(cy) * img_h + float(h) * img_h / 2)

				draw.rectangle([(x1, y1), (x2,y2)], outline=(255,255,0))
			self.viz.image(np.array(img).transpose((2,0,1)), win=i)