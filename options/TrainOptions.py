from .BaseOptions import BaseOptions 

class TrainOptions(BaseOptions):
	# this class include training options

	def initialize(self):
		parser = BaseOptions.initialize(self)
		# add some training options here
		self.parser.add_argument("--isTrain", type=bool, default=True)
		self.parser.add_argument("--dataroot", type=str, default='datasets/visDrone/train/images')
		self.parser.add_argument("--noagument", action='store_true')
		self.parser.add_argument("--nomultiscale", action='store_true')
		self.parser.add_argument("--normalized_labels", action='store_false')
		self.parser.add_argument("--load", action='store_true')
		self.parser.add_argument("--iter_name", type=str, default='latest.pth')
		self.parser.add_argument("--lr", type=float, default=1e-4)
		self.parser.add_argument("--w", type=int, default=416)
		self.parser.add_argument("--h", type=int, default=416)

		return parser
