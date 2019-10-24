from .BaseOptions import BaseOptions

class TestOptions(BaseOptions):
	# this class include testing options

	def initialize(self):
		parser = BaseOptions.initialize(self)
		# add some testing options here
		self.parser.add_argument('--isTrain', type=bool, default=False)
		self.parser.add_argument('--dataroot', type=str, default='datasets/debug/train/image/DJI_0006')
		self.parser.add_argument('--noagument', action='store_false') # no agument
		self.parser.add_argument('--nomultiscale', action='store_false') # no multiscale
		self.parser.add_argument('--iter_name', type=str, default='latest.pth')
		self.parser.add_argument('--lr', type=float, default=1e-4)
		
		self.parser.add_argument('--output_dir', type=str, default='output/')

		return parser