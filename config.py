import os

cur_path = os.path.abspath(os.path.dirname(__file__))
PROJECT_NAME = os.path.split(cur_path)[1]

# train：tr validation：cv test：tt
# TYPE = 'tr'
# train
EPOCH = 1000
TRAIN_BATCH_SIZE = 64
# TRAIN_DATA_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k/tmp/train.bin'
# TRAIN_DATA_LABEL_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k/tmp/train_label.bin'
TRAIN_DATA_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/tmp/train.wav'
TRAIN_DATA_LABEL_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/tmp/train_label.wav'
TRAIN_DATA_NUM = 4620

# validation
VALIDATION_BATCH_SIZE = 1
VALIDATION_DATA_PATH = '/home/yangyang/userspace/data/envaluation_1/'
VALIDATION_DATA_PATH_NEW = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/test/'
VALIDATION_DATA_LABEL_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/test_label/'
VALIDATION_DATA_NUM = 100

TEST_DATA_PATH = '/mnt/raid/data/public/SPEECH_ENHANCE_DATA_NEW/tt/mix/'
TEST_DATA_PATH_NEW = '/home/yangyang/userspace/data/new/test_data.txt'
TEST_DATA_PATH_NEW_SMALL = '/home/yangyang/userspace/data/new/test_data_small.txt'
TEST_DATA_PATH_NEW_MINI = '/home/yangyang/userspace/data/new/test_data_mini.txt'
TEST_DATA_NUM = 10000
TEST_DATA_NUM_SMALL = 2000
TEST_DATA_NUM_MINI = 1000

# model
MODEL_STORE = os.path.join('/home/yangyang/userspace/module_store/', PROJECT_NAME + '/')
if not os.path.exists(MODEL_STORE):
    os.mkdir(MODEL_STORE)
    print('Create model store file  successful!\n'
          'Path: \"{}\"'.format(MODEL_STORE))
else:
    print('The model store path: {}'.format(MODEL_STORE))

# log
LOG_STORE = os.path.join('/home/yangyang/userspace/log/', PROJECT_NAME + '/')
if not os.path.exists(LOG_STORE):
    os.mkdir(LOG_STORE)
    print('Create log store file  successful!\n'
          'Path: \"{}\"'.format(LOG_STORE))
else:
    print('The log store path: {}'.format(LOG_STORE))


# result
RESULT_STORE = os.path.join('/home/yangyang/userspace/result/', PROJECT_NAME + '/')
if not os.path.exists(RESULT_STORE):
    os.mkdir(RESULT_STORE)
    print('Create validation result store file  successful!\n'
          'Path: \"{}\"'.format(RESULT_STORE))
else:
    print('The validation result store path: {}'.format(RESULT_STORE))

# tag
PRE_TRAIN = True
CUDA_ID = ['cuda:1']
IS_LOG = True
PRE_TRAIN_D = False
PRE_TRAIN_G = False
TRAIN = True
IS_REVERT = False   # d_net的输入是否revert
# other parameters
FILTER_LENGTH = 160 # 窗长
HOP_LENGTH = 80     # 帧移
EPSILON = 1e-8
NUM_WORKERS = 8
LAMBDA_FOR_REC_LOSS = 0.5
if TRAIN:
    LR = 1e-5
else:
    LR = 1e-4
D_STEP = 10
PRINT_TIME = 200
SAVE_TIME = 2000
GAMMA = 2
