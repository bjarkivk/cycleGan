class TrainOptions():

    def __init__(self, n_epochs=10, epoch_count=0, n_epochs_decays=0):
        self.n_epochs = n_epochs
        self.epoch_count = epoch_count
        self.n_epochs_decays = n_epochs_decays
        self.gpu_ids = 0
        self.checkpoints_dir = './checkpoints'
        self.model = 'cycle_gan'
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.direction = 'AtoB'
        self.batch_size = 1

        #parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        #parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        #parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        
        ##parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        #parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        #parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        #parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        #parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        #parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
