import math
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
PROJECT_NAME = os.path.split(cur_path)[1]

# c_train：tr validation：cv e_test：tt
# TYPE = 'tr'
# c_train
EPOCH = 1000
TRAIN_BATCH_SIZE = 64
TRAIN_DATA_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/tmp/train.wav'
TRAIN_LABEL_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/tmp/train_label.wav'
TRAIN_PARAM_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/tmp/'
TRAIN_DATA_NUM = 4620

# validation
VALIDATION_BATCH_SIZE = 1
VALIDATION_DATA_PATH = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/test/'
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
CUDA_ID = ['cuda:1']
PRE_TRAIN_D = False
PRE_TRAIN_G = False
TRAIN = True
IS_REVERT = False   # d_net的输入是否revert
SAMPLING_RATE = 8000
WINDOWS_TIME = 32 * 1e-3   # 32 ms
HOP_TIME = 8 * 1e-3        # 8ms
# other parameters
FILTER_LENGTH = int(SAMPLING_RATE * WINDOWS_TIME)    # window sample num
HOP_LENGTH = int(SAMPLING_RATE * HOP_TIME)       # hop sample num
FEATURE_NUM = int(FILTER_LENGTH // 2 + 1)        # feature number
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
NEED_NORM = True
IS_LOG = True

# for this project
# 2x : 65
# 4x : 33
G_INPUT_NUM = FEATURE_NUM // 2 + 1      # for 2x
# G_INPUT_NUM = FEATURE_NUM // 4 + 1    # for 4x
# 2x : 7
# 4x : 4
OVER_LAP = math.floor(G_INPUT_NUM / 10) + 1
# 2x : 71
# 4x : 100
G_OUTPUT = FEATURE_NUM - (G_INPUT_NUM - OVER_LAP)
