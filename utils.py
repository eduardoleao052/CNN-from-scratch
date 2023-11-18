import numpy as np
import pandas as pd
from argparse import ArgumentParser
import logging

class Augmenter():
    def __init__(self):
        pass

    def fit(self, data):
        self.data = np.array(data)

    def transform(self, data, data_y, dimensions = None, ratio = 4):
        data = np.array(data)
        data_y = np.array(data_y)
        if dimensions == None:
            dimensions = (int(np.sqrt(np.max(data.shape[1]))),int(np.sqrt(np.max(data.shape[1]))))
        new_data_x = []
        new_data_y = []
        for x, y in zip(data,data_y):
            data_reshaped = x.reshape(dimensions[0], dimensions[1])
            for k in range(ratio//4 + 1):
                data_expanded_x_pos = np.roll(data_reshaped, shift = k).reshape(-1,)
                data_expanded_x_neg = np.roll(data_reshaped, shift = -k).reshape(-1,)
                data_expanded_y_pos = np.roll(data_reshaped, shift = k, axis = 0).reshape(-1,)
                data_expanded_y_neg = np.roll(data_reshaped, shift = -k, axis = 0).reshape(-1,)
            new_data_x += [data_expanded_x_pos, data_expanded_x_neg, data_expanded_y_pos, data_expanded_y_neg]
            new_data_y += [y, y, y, y]
            #new_data_x += [data_expanded_x_pos, data_expanded_y_pos]
            #new_data_y += [y, y]

        df_x = pd.DataFrame(np.array(new_data_x))
        df_y = pd.DataFrame(np.array(new_data_y))
        return df_x, df_y
    
    def fit_transform(self, data, data_y, dimensions = None, ratio = 4):
        if ratio == 1:
            return data, data_y
        data = np.array(data)
        data_y = np.array(data_y)
        if dimensions == None:
                dimensions = (int(np.sqrt(np.max(data.shape[1]))),int(np.sqrt(np.max(data.shape[1]))))
        new_data_x = []
        new_data_y = []
        for x, y in zip(data,data_y):
            data_reshaped = x.reshape(dimensions[0], dimensions[1])
            for k in range(1, ratio//4 + 1):
                data_expanded_x_pos = np.roll(data_reshaped, shift = k).reshape(-1,)
                data_expanded_x_neg = np.roll(data_reshaped, shift = -k).reshape(-1,)
                data_expanded_y_pos = np.roll(data_reshaped, shift = k, axis = 0).reshape(-1,)
                data_expanded_y_neg = np.roll(data_reshaped, shift = -k, axis = 0).reshape(-1,)
                new_data_x += [data_expanded_x_pos, data_expanded_x_neg, data_expanded_y_pos, data_expanded_y_neg]
                new_data_y += [y, y, y, y]
                #new_data_x += [data_expanded_x_pos, data_expanded_y_pos]
                #new_data_y += [y, y]
        df_x = pd.DataFrame(np.array(new_data_x))
        df_y = pd.DataFrame(np.array(new_data_y))
        return df_x, df_y

def Adam(b,w,db,dw,config):
    #UPDATE W
    config['m_w'] = (config['m_w']*config['beta1'] + (1 - config['beta1']) * dw) #/ (1- config['beta1']**config['t'])
    config['v_w'] = (config['v_w']*config['beta2'] + (1 - config['beta2']) * np.square(dw)) #/ (1- config['beta2']**config['t'])

    next_w = w - (config["learning_rate"] * config['m_w']) / (np.sqrt(config['v_w']) + config['epsilon']) - config["regularization"] * config["learning_rate"] * w
    #print("BREAKDOWN W")
    #print(config['m_w'])
    #print(config['v_w'])
    #print("END")

    #print("w =========================")
    #print(w)
    #print((config["learning_rate"] * config['m_w']) / (np.sqrt(config['v_w']) + config['epsilon']))
    #UPDATE B
    config['m_b'] = (config['m_b']*config['beta1'] + (1 - config['beta1']) * db) #/ (1- config['beta1']**config['t'])
    config['v_b'] = (config['v_b']*config['beta2'] + (1 - config['beta2']) * np.square(db))# / (1- config['beta2']**config['t'])
    
    next_b = b - (config["learning_rate"] * config['m_b']) / (np.sqrt(config['v_b']) + config['epsilon'])

    #print('b')
    #print((config["learning_rate"] * config['m_b']) / (np.sqrt(config['v_b']) + config['epsilon']))
    #print(b)

    config['t'] += 1

    return next_w, next_b, config

def SGD(b,w,db,dw,config):
    next_w = w - config['learning_rate'] * dw - config['learning_rate'] * config['regularization'] * w
    next_b = b - config['learning_rate'] * db
    return next_w, next_b, config

def Momentum(b,w,db,dw,config):
    config['m_w'] = config['m_w'] * config['beta1'] + (1 - config['beta1']) * dw
    config['m_b'] = config['m_b'] * config['beta1'] + (1 - config['beta1']) * db
    next_w =   w - config['learning_rate'] * config['m_w'] - config['learning_rate'] * config['regularization'] * w
    next_b =   b - config['learning_rate'] * config['m_b']
    return next_w, next_b, config 

def add_padding(x,k):
    if k == 0:
        return x
    H,W = x.shape
    a = np.zeros((H+2*k,W+2*k))
    a[k:-k,k:-k] = x
    return a

def rotate_180(x):
    H,W = x.shape[0], x.shape[1]
    a = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            a[i, W-1-j] = x[H-1-i, j]
    return a
    
def cross_correlate(x,y,stride=1):
    H, W = x.shape
    HH, WW = y.shape
    Ho, Wo = 1 + (H - HH)//stride , 1 + (W - WW)//stride
    a = np.zeros((Ho,Wo))
    for i in range(Ho):
        for j in range(Wo):
            a[i,j] = np.sum(x[i*stride : HH + i*stride, j*stride : WW + j*stride] * y)
    return a

def convolute(x,y,stride=1):
    return cross_correlate(x,rotate_180(y),stride=stride)

def build_logger(sender, pwd, PATH):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")

    file_handler = logging.FileHandler(f"{PATH}/training.log")
    smtpHandler = logging.handlers.SMTPHandler(
    mailhost=("smtp.gmail.com",587),
    fromaddr=sender,
    toaddrs=sender,
    subject="Training Alert",
    credentials=(sender, pwd),
    secure=()
    )

    file_handler.setLevel(logging.INFO)
    smtpHandler.setLevel(logging.WARNING)

    file_handler.setFormatter(formatter)
    smtpHandler.setFormatter(formatter)

    logger.addHandler(smtpHandler)
    logger.addHandler(file_handler)
    return logger

def parse_arguments(PATH):
    parser = ArgumentParser(description='configuration of runtime application')
    
    parser.add_argument('--train', action='store_true',
                        help='train the model with provided image dataset')
    parser.add_argument('--test', action='store_true',
                        help='test the model with image dataset')
    
    parser.add_argument('--train_data', nargs='?', type=str, default=f"{PATH}/data/mnist_train.csv",
                        help='path to training dataset used to train model')
    parser.add_argument('--test_data', nargs='?', type=str, default=f"{PATH}/data/mnist_test.csv",
                        help='path to test dataset used to evaluate model')
    parser.add_argument('--epochs',nargs='?', type=int, default=30,
                        help='number of whole passes through training data in training')
    parser.add_argument('--batch_size',nargs='?', type=int, default=10,
                        help='size of the batch (number of images per batch)')
    parser.add_argument('--augmenter_ratio',nargs='?', type=int, default=1,
                        help='1:ratio is how many times the training dataset will be augmented (new images generated by slightly shifting old images)')
    parser.add_argument('--to_path', nargs='?', type=str, default=f"{PATH}/models/my_model.json",
                        help='path to .json file where model will be stored')
    parser.add_argument('--from_path',nargs='?', default=f"{PATH}/models/my_model.json",
                        help='path to file with model parameters to be loaded')
    
    args = parser.parse_args()

    return args

def load_data(args):
    xl = None
    yl = None
    if args.train:
        #Load train data:
        training_data = pd.read_csv(f'{args.train_data}', header=None,low_memory=False)
        yl = training_data.iloc[1:10010,0].astype('float')
        xl = training_data.iloc[1:10010,1:].astype('float')

        #Data Augmentation (shift horizontal & vertical):
        augmenter = Augmenter()
        xl, yl = augmenter.fit_transform(xl,yl,ratio=args.augmenter_ratio)
        
        #Turn into numpy array:
        xl = xl.to_numpy().astype('float')
        yl = yl.to_numpy().astype('float')

        #Normalization:
        xl = (xl - xl.mean()) / (xl.std() + 1e-5)

    #Load test data:
    testing_data = pd.read_csv(f'{args.test_data}', header=None,low_memory=False)
    yt = testing_data.iloc[1:,0].astype('float')
    xt = testing_data.iloc[1:,1:].astype('float')

    #Turn into numpy array:
    xt = xt.to_numpy().astype('float')
    yt = yt.to_numpy().astype('float')

    #Normalization:
    xt = (xt - xt.mean()) / (xt.std() + 1e-5)
    

    return xl, xt, yl, yt

def train_model(model, xl, yl, args, logger):
    #Training Params:
    validation_size = 0.1
    learning_rate = 2e-3
    regularization = 2e-3
    logger.info("\nTraining model...")
    model.train(xl,yl, epochs = args.epochs, batch_size = args.batch_size, validation_size = validation_size, learning_rate = learning_rate, regularization = regularization) #BEST 5e-4

    model.load(args.to_path)
    for layer in model.layers:
            layer.compile(0,0)
    test_acc = model.evaluate(model.predict(xt),yt)
    print(f"Test accuracy:{test_acc}")

    logger.warning("\nTraining complete!\n\nHyperparameters:\nEpochs: {}\nBatch_size: {}\nValidation_size: {}\nlearning_rate: {}\nregularization: {}\n\nStatistics:\nBest Acc Val: {}\nVal Accs: {}\nTest Acc: {}\n\n\n".format(args.epochs,args.batch_size,validation_size,learning_rate,regularization,np.max(model.accs),model.accs,test_acc))
    return

def test_model(model, xt, yt, args, logger):
    logger.info("\Testing model...")
    #Load model from given path:
    model.load(args.from_path)
    for layer in model.layers:
            layer.compile(0,0)
    model.test(xt,yt)

