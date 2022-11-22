IMAGE_PATH = './output_image/'
MODEL_PATH = 'encoder_model'

DATASET = 'moon'

NETWORK_SEED = 0
DATASET_SEED = 42

# init loop
INIT_EPOCH = 30

INIT_EBCODER_LR = 0.01
INIT_DECODER_LR = 0.005

# train loop
EPOCH = 50
ITER = 5

ENCODER_LR = 0.01
DECODER_LR = 0.005

SCHEDULER_STEP = 10
SCHEDULER_GAMMA = 0.9

from py_code.StepVariable import StepVariable
# epoch every LAMBDA_STEP, lambda *= LAMBDA_GAMMA
LAMBDA = StepVariable(0.0005, _gamma = 1.05 , _step = 5) # weight

# center loss
WITH_CENTER_LOSS = True
CENTER = StepVariable(2e-6, _gamma = 3 , _step = 10) # weight
K_CENTER = 2

# number of sample
N = 1000
BATCH_SIZE = 128
x_dim = 2 # X
y_dim = 2 # U
embedding_dim = x_dim

DEBUG = False

# gif
SAVE_GIF = True
SAVE_STEP = 5


# check_U mode, input dimension(1d, 2d, 3d)(split by space) show U graph
CHECK_U = False