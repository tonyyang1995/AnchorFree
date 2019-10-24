import time
from options.TrainOptions import TrainOptions
from options.TestOptions import TestOptions

from data.CustomDataset import get_dataset
from utils.visdom import Visualizer

import torch

def Coarse_Train():
	opt = TrainOptions().parse()
	dataset = get_dataset(opt)
	vis = Visualizer(opt)
	print(len(dataset))

	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size = opt.batch_size,
		shuffle = True,
		collate_fn = dataset.collate_fn
	)

	for epoch in range(0,1):
		start_time = time.time()
		for i, (img, target) in enumerate(dataloader):
			print(img.size())
			print(target.size())
			vis.plot_rawdata_target(img, target)

			break
		break

if __name__ == '__main__':
	Coarse_Train()