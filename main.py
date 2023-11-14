import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from layers import *
from augmenter import Augmenter
from optimizers import *
import logging
import logging.handlers

def build_logger(sender, pwd):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")

    file_handler = logging.FileHandler("C:/Users/twich/OneDrive/Documentos/NeuralNets/cnn/training.log")
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

def load_data():
    #Load train data:
    training_data = pd.read_csv('/Users/eduardoleao/Documents/ML/NN/cnn/data/mnist_train.csv', header=None,low_memory=False)
    yl = training_data.iloc[1:,0].astype('float')
    xl = training_data.iloc[1:,1:].astype('float')
    #Load test data:
    testing_data = pd.read_csv('/Users/eduardoleao/Documents/ML/NN/cnn/data/mnist_test.csv', header=None,low_memory=False)
    yt = testing_data.iloc[1:,0].astype('float')
    xt = testing_data.iloc[1:,1:].astype('float')

    #Data Augmentation (shift horizontal & vertical):
    augmenter = Augmenter()
    xl, yl = augmenter.fit_transform(xl,yl,ratio=4)

    xl = xl.to_numpy().astype('float')
    yl = yl.to_numpy().astype('float')
    xt = xt.to_numpy().astype('float')
    yt = yt.to_numpy().astype('float')

    #Normalization:
    xl = (xl - xl.mean()) / (xl.std() + 1e-5)
    xt = (xt - xt.mean()) / (xt.std() + 1e-5)

    return xl, xt, yl, yt

class Model():
    def __init__(self):
        self.layers = []
        self.accs = []
        self.accs_train = []

    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, a):
        for layer in self.layers:
                    a = layer.forward(a, training = False)
        return [int(np.argmax(k)) for k in a]

    def evaluate(self, y_pred, y):
        acc = np.mean([j==k for (j,k) in zip(y_pred,y)])
        return acc

    def train(self, x, y, epochs=30, batch_size = 15, validation_size = .1, learning_rate = 9e-4, regularization = 1e-3, learning_rate_decay = .4, patience = 4):
        n = len(x)
        for layer in self.layers:
            layer.compile(learning_rate,regularization)

        #Validation Split
        x_v = x[:int(validation_size*n//1),:]
        y_v = y[:int(validation_size*n//1)]
        x = x[int(validation_size*n//1):,:]
        y = y[int(validation_size*n//1):]
        n = len(x)
        print(x_v.shape)
        print(x.shape)
        print(y_v.shape)
        print(y.shape)

        #Vectorize "y":
        new_y = np.zeros((len(y),10))
        for i in range(len(y)):
            new_y[i,int(y[i])] = 1
        

        #Iterate over Epochs:
        n_batches = n // batch_size
        decay_counter = 0
        for i in range(epochs):
            try:
                #Randomize x,y:
                p = np.random.permutation(n)
                x_rand = x[p]
                y_rand = new_y[p]
                for j in range(n_batches):
                    #Set mini_batch:
                    x_batch = x_rand[(j)*batch_size:(j+1)*batch_size,:]
                    y_batch = y_rand[(j)*batch_size:(j+1)*batch_size]
                    
                    #Forwardprop:
                    a = x_batch
                    for layer in self.layers:
                        a = layer.forward(a, training = True)
                    
                    #Backprop:
                    dl = a
                    self.layers.reverse()
                    for layer in self.layers:
                        dl = layer.backward(dl,y_batch)
                    self.layers.reverse()

                #Evaluate Train:
                #acc_train = self.evaluate(self.predict(x), y)
                #self.accs_train.append(acc_train)
                #Evaluate Validation:
                acc = self.evaluate(self.predict(x_v), y_v)
                self.accs.append(acc)
                print("({}) - Validation accuracy:{}".format(i,acc))
                #print("({}) - Training accuracy:{}".format(i,acc_train))
                #Save best model:
                if np.max(self.accs) == acc:
                    print("New Best Model!")
                    #self.best_model = deepcopy(self)
                #Apply Learning Rate Decay:
                decay_counter += 1
                if i - np.argmax(self.accs) >= patience and decay_counter > patience:
                    decay_counter = 0
                    print(learning_rate)
                    learning_rate *= (1 - learning_rate_decay + 1e-2)
                    print(learning_rate)
                    for layer in self.layers:
                        layer.config['learning_rate'] = learning_rate
                    print("BROKE")
                
                #for i in range(5):
                #    plt.imshow(model.layers[0].w[i][0], cmap='hot', interpolation='nearest')
                #    plt.show()
            except:
                logger.exception("\nAn exception has occured on epoch {}:".format(i))

def main():
    global logger
    logger = build_logger("twich.badass@gmail.com", "tjmgyekrpmafqfrw")

    #TEST model with MNIST dataset from kaggle:
    logger.info("\nLoading data...")

    #Load MNIST data:
    xl, xt, yl, yt = load_data()

    #Build Model:
    logger.info("\nBuilding model...")
    model = Model()
    model.add(Conv(input_shape = (1,28,28), num_kernels = 5, kernel_size = 5,padding=2))
    model.add(BatchNorm())
    model.add(Relu())
    model.add(MaxPool((5,28,28)))

    model.add(Conv(input_shape = (5,14,14), num_kernels = 5, kernel_size = 3,padding=1))
    model.add(BatchNorm())
    model.add(Relu())
    model.add(MaxPool((5,14,14)))

    model.add(Flatten((5,7,7),(245)))

    model.add(Dense(245,256,optimizer=Adam))
    model.add(BatchNorm())
    model.add(Relu())
    #model.add(Dropout(p=.8))

    model.add(Dense(256,256,optimizer=Adam))
    model.add(BatchNorm())
    model.add(Relu())
    #model.add(Dropout(p=.8))

    model.add(Dense(256,256,optimizer=Adam))
    model.add(BatchNorm())
    model.add(Relu())
    #model.add(Dropout(p=.8))

    model.add(Dense(256,10,optimizer=Adam))
    model.add(Softmax())

    #Training Params:
    epochs = 60
    batch_size = 10
    validation_size = .05
    learning_rate = 2e-3
    regularization = 2e-3
    logger.info("\nTraining model...")
    model.train(xl,yl, epochs = epochs, batch_size = batch_size, validation_size = validation_size, learning_rate = learning_rate, regularization = regularization) #BEST 5e-4

    test_acc = model.evaluate(model.predict(xt),yt)
    print(test_acc)

    logger.warning("\nTraining complete!\n\nHyperparameters:\nEpochs: {}\nBatch_size: {}\nValidation_size: {}\nlearning_rate: {}\nregularization: {}\n\nStatistics:\nBest Acc Val: {}\nVal Accs: {}\nTest Acc: {}\n\n\n".format(epochs,batch_size,validation_size,learning_rate,regularization,np.max(model.accs),model.accs,test_acc))
    
    #for i in range(len(model.layers[0].w.T)):
    #            plt.imshow(model.layers[0].w.T[i].reshape(28,28), cmap='hot', interpolation='nearest')
    #            plt.show()
    #            print(i)
    return

if __name__ == "__main__":
    main()
