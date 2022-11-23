# path
IMAGE_PATH = './output_image/'
MODEL_PATH = 'encoder_model'

# dataset
DATASET = 'blob'
DATASET_SEED = 42

# network
NETWORK_SEED = 0

# init loop
INIT_EPOCH = 30

INIT_EBCODER_LR = 0.01
INIT_DECODER_LR = 0.005

# training loop
EPOCH = 50
ITER = 5

ENCODER_LR = 0.01
DECODER_LR = 0.005

SCHEDULER_STEP = 10
SCHEDULER_GAMMA = 0.9

# Every `_step` epochs => lambda *= `_gamma`
from py_code.StepVariable import StepVariable

# f_function
LAMBDA = StepVariable(0.0005, _gamma = 1.05 , _step = 5) # weight

# center loss
WITH_CENTER_LOSS = True
CENTER = StepVariable(2e-4, _gamma = 1.2 , _step = 10) # weight
K_CENTER = 2

# number of sample
N = 1000
BATCH_SIZE = 128
x_dim = 2   # X (input dimension)
y_dim = 2   # U (Top `y_dim` eigenVector)
embedding_dim = x_dim

# DEBUG: print something
DEBUG = False

# GIF: every `SAVE_STEP` epochs, save training_result
SAVE_GIF = True
SAVE_STEP = 5

# check_U mode, input dimension(1d, 2d, 3d)(split by space) show U graph
CHECK_U = False