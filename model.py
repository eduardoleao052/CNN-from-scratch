import numpy as np
import json

class Model():
    def __init__(self, logger,args):
        self.layers = []
        self.accs = []
        self.accs_train = []
        self.logger = logger
        self.args = args

    def add(self, layer):
        self.layers.append(layer)
    
    def save(self, path:str) -> None:
        """
        Save current model parameters on separate file, to later be loaded.

        @param path (str): file path where model parameters will be saved
        """

        params = []
        for layer in self.layers:
            params.append({'w': layer.w.tolist(), 'b': layer.b.tolist()})

        params = json.dumps(params)
        file = open(path, 'w')
        file.write(params)
        file.close()
     
    def load(self, path:str) -> None:
        """
        Load model params from json file

        @param path (str): file path where model parameters are
        """

        self.preloaded = True
        file = open(path, 'r')
        param_list = file.read()
        param_list = json.loads(param_list)

        for i, layer in enumerate(self.layers):
            layer.w = np.array(param_list[i]['w'])
            layer.b = np.array(param_list[i]['b'])

    def predict(self, a):
        for layer in self.layers:
            a = layer.forward(a, training = False)
        return [int(np.argmax(k)) for k in a]

    def evaluate(self, y_pred, y):
        #print(y)
        #print(y_pred)
        acc = np.mean([j==k for (j,k) in zip(y_pred,y)])
        return acc
    
    def test(self, xt, yt):
        acc = self.evaluate(self.predict(xt), yt)
        print("Test accuracy:{}".format(acc))
        return

    def train(self, x, y, epochs=30, batch_size = 15, validation_size = .1, learning_rate = 9e-4, regularization = 1e-3, learning_rate_decay = 0.05, patience = 4):
        n = len(x)
        for layer in self.layers:
            layer.compile(learning_rate,regularization)

        #Validation Split
        x_v = x[:int(validation_size*n//1),:]
        y_v = y[:int(validation_size*n//1)]
        x = x[int(validation_size*n//1):,:]
        y = y[int(validation_size*n//1):]
        n = len(x)

        print(f"Number of training instances: {y.shape[0]}")
        print(f"Number of validation instances: {y_v.shape[0]}")

        #One-hot encode "y":
        new_y = np.zeros((len(y),10))
        for i in range(len(y)):
            new_y[i,int(y[i])] = 1
        

        #Iterate over Epochs:
        n_batches = n // batch_size
        decay_counter = 0
        for epoch in range(epochs):
            #try:
                #Randomize x,y:
                p = np.random.permutation(n)
                x_rand = x[p]
                y_rand = new_y[p]
                for current_batch in range(n_batches):
                    #Set mini_batch:
                    x_batch = x_rand[(current_batch)*batch_size:(current_batch+1)*batch_size,:]
                    y_batch = y_rand[(current_batch)*batch_size:(current_batch+1)*batch_size]
                   
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
                acc_train = self.evaluate(self.predict(x), y)
                self.accs_train.append(acc_train)
                #Evaluate Validation:
                acc = self.evaluate(self.predict(x_v), y_v)
                self.accs.append(acc)
                print(f"Epoch ({epoch}/{epochs}) - Val accuracy:{acc} - Train accuracy:{acc_train}")
                self.logger.info(f"Epoch ({epoch}/{epochs}) - Val accuracy:{acc} - Train accuracy:{acc_train}")
                #Save best model:
                if np.argmax(self.accs) == epoch:
                    print("New best model")
                    self.save(self.args.to_path)
                #Apply Learning Rate Decay:
                elif decay_counter > patience:
                    decay_counter = 0
                    learning_rate *= (1 - learning_rate_decay)
                    for layer in self.layers:
                        layer.config['learning_rate'] *= (1 - learning_rate_decay)
                    print(f"BROKE - lr: [{learning_rate/(1-learning_rate_decay)}] -> [{learning_rate}]")
                decay_counter += 1
                
                #for i in range(5):
                #    plt.imshow(self.layers[0].w[i][0], cmap='hot', interpolation='nearest')
                #    plt.show()
            # except:
            #     self.logger.exception("\nAn exception has occured on epoch {}:".format(epoch))