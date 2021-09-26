import os

# train
BATCH_SIZE = 8
NW = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
NUM_CLASSES = 4
START_EPOCH = 1
END_EPOCH = 5