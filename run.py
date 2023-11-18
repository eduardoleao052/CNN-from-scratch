import pandas as pd
import numpy as np
from layers import *
from model import Model
from utils import *
import logging
import logging.handlers
from argparse import ArgumentParser
import os

PATH = os.getcwd()

logger = build_logger('output.logger@gmail.com','bcof jupb ugbh vfll', PATH)
# Get arguments from terminal:
args = parse_arguments(PATH)

#Load MNIST data:
logger.info("\nLoading data...")
xl, xt, yl, yt = load_data(args)

#Build Model:
logger.info("\nBuilding model...")
model = Model(logger, args)
model.add(Conv(input_shape = (1,28,28), num_kernels = 5, kernel_size = 5,padding=2))
model.add(BatchNorm())
model.add(Relu())
model.add(MaxPool((5,28,28)))

model.add(Conv(input_shape = (5,14,14), num_kernels = 5, kernel_size = 3, padding=1))
model.add(BatchNorm())
model.add(Relu())
model.add(MaxPool((5,14,14)))

model.add(Flatten((5,7,7)))

model.add(Dense(245,256,optimizer=Adam))
model.add(BatchNorm())
model.add(Relu())
#model.add(Dropout(p=.8))

model.add(Dense(256,256,optimizer=Adam))
model.add(BatchNorm())
model.add(Relu())
#model.add(Dropout(p=.8))

model.add(Dense(256,10,optimizer=Adam))
model.add(Softmax())

#Train or test:
if args.train:
    train_model(model, xl, yl, args, logger)
if args.test:
    test_model(model, xt, yt, args, logger)