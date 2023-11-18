import numpy as np
import pandas as pd
from optimizers import *
import scipy.signal

class Dense():
    def __init__(self, in_size, out_size, optimizer = Momentum):
        self.in_size = in_size
        self.out_size = out_size
        self.b = np.zeros((1,out_size))
        self.w = np.random.randn(in_size, out_size)/np.sqrt(in_size)
        self.config = {}
        self.optimizer = optimizer

    def compile(self,lr, reg):
        self.config = {'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_b':np.zeros(self.b.shape),
                       'v_b':np.zeros(self.b.shape),
                       'm_w':np.zeros(self.w.shape),
                       'v_w':np.zeros(self.w.shape),
                       't':30}
    def forward(self,x, training = False):
        #x_flat = x.reshape(x.shape[0],-1)
        #x_flat = x.reshape(1,-1)
        z = np.dot(x,self.w) + self.b
        self.x = x
        self.z = z
        return z

    def backward(self, dz, y):
        x = self.x
        #x_flat = x.reshape(x.shape[0],-1)
        #x_flat = x.reshape(1,-1)
        
        db = dz.sum(axis = 0).T
        dw = np.dot(x.T, dz)
        dx = np.dot(dz, self.w.T)

        #Apply Optimizer:
        w, b, self.config = self.optimizer(self.b,self.w,db,dw,self.config)


        self.w = w 
        self.b = b

        return dx


class Relu():
    def __init__(self):
        self.config = {}
        self.w = np.ndarray([])
        self.b = np.ndarray([])

    def compile(self,lr, reg):
        self.config = {'learning_rate': lr}

    def forward(self,z, training = False):
        self.z = z
        a = np.maximum(z,-0.01*z)
        return a

    def backward(self,da,y):
        z = self.z
        zeroer = lambda z: 1 if z>0 else -0.01
        zeroer_vect = np.vectorize(zeroer)
        dz_mask = zeroer_vect(z)
   
    
        dz = da * dz_mask        
        return dz


class Softmax():
    def __init__(self):
        self.config = {}
        self.w = np.ndarray([])
        self.b = np.ndarray([])

    def compile(self,lr, reg):
        self.config = {'learning_rate': lr}

    def forward(self, z, training = False):
        n = z.shape[0]
        self.n = n
        z -= np.max(z,axis=1,keepdims=True)
        a = np.exp(z) / np.sum(np.exp(z),axis=1,keepdims=True)
        return a

    def backward(self,a,y):
        n = self.n
        
        #correct_class_probabilities = a[range(n),y]
        #loss = np.sum(-np.log(correct_class_probabilities)) / n

        dz = (a - y)/n
        #print('a ==========')
        #print(a)
        #print(y)
        #print(dz)
        return dz


class Dropout():
    def __init__(self, p=0.8):
        self.config = {}
        self.p = p
        self.w = np.ndarray([])
        self.b = np.ndarray([])
        
    def compile(self,lr, reg):
        self.config = {'learning_rate': lr}
        
    def forward(self,z,training = True):
        if training == True:
            self.z = z
            self.mask = np.random.choice((0,1),size = z.shape,  p=[(1-self.p), self.p]) / self.p
            a = z * self.mask
            return a
        else:
            return z 
    def backward(self,da,y):
        dz = da * self.mask
        return dz


class BatchNorm():
    def __init__(self, gamma = 1, beta = 0):
        self.config = {}
        self.gamma = gamma
        self.beta = beta
        self.spatial = False
        self.w = 0
        self.b = 0

    def compile(self,lr, reg):
        self.config['learning_rate'] = lr
        self.config['running_mean'] = self.w
        self.config['running_var'] = self.b
        
        
    def forward(self,z,training = True):
        #If input is spatial, flatten so that batchnorm occurs for every "C" channel.
        if z.ndim == 4:
            N, C, H, W = z.shape
            z = z.transpose(0,2,3,1).reshape(N*H*W, C)
            self.spatial = True
        if training == True:
            self.z = z
            #Calculate mean
            mean = np.mean(z, axis = 0)
            self.mean = mean

            #Calculate standard deviation
            var = np.var(z,axis=0)
            self.var = var
            std = np.sqrt(var + 1e-5)
            self.std = std

            #Execute normalization
            self.x_norm = (z - mean) / std

            #Applt internal params
            a = self.gamma * self.x_norm + self.beta

            #Update running mean and variance:
            self.config['running_mean'] = 0.99 * self.config['running_mean'] + (1 - 0.99) * self.mean
            self.config['running_var'] = 0.99 * self.config['running_var'] + (1 - 0.99) * self.var
            # print('ayo')
            # print(self.config['running_var'].shape)
            # print(self.config['running_mean'].shape)
            self.w = self.config['running_mean']
            self.b = self.config['running_var']

        elif training == False:
            #On test, apply average normaliz
            a = self.gamma*((z - self.config['running_mean']) / np.sqrt(self.config['running_var'] + 1e-5)) + self.beta

        if self.spatial == True:
            a = a.reshape(N, H, W, C).transpose(0,3,1,2)
        
        return a
    
        
    def backward(self,da,y):
        #If input is spatial, flatten so that batchnorm occurs for every "C" channel.
        if self.spatial == True:
            N, C, H, W = da.shape
            da = da.transpose(0,2,3,1).reshape(N*H*W, C)
        n,d = da.shape
        z = self.z
        #Beta gradient:
        dbeta = np.sum(da, axis=0)

        # out_gamma = gamma * x_norm:
        dgamma = np.sum(da * self.x_norm, axis=0)     #upstream: da
        dx_norm = self.gamma * da                     #upstream: da

        # x_norm = x_centered / self.std
        dx_centered = (1 / self.std) * dx_norm                               #upstream: dx_norm
        dstd = np.sum(dx_norm * (z - self.mean) * (- self.std**(-2)),axis=0) #upstream: dx_norm
        # std = sqrt(var)
        dvar = dstd / 2 / self.std                   #upstream: dstd
        
        # x_norm = z - self.mean / var(self.mean)
        d_mean = -(np.sum(dx_centered, axis=0) + (2/n) * np.sum(z - self.mean,axis=0))

        dx = dx_centered + (d_mean + 2 * dvar * (z - self.mean)) / n

        self.gamma -= self.config['learning_rate'] * dgamma / n
        self.beta -= self.config['learning_rate'] * dbeta / n

        if self.spatial == True:
            dx = dx.reshape(N, H, W, C).transpose(0,3,1,2)

        return dx
    

class Conv():
    def __init__(self, input_shape, kernel_size, num_kernels, padding = 0, stride = 0, optimizer = Momentum):
        self.C, self.H, self.W = input_shape
        self.input_shape = input_shape
        self.F = num_kernels
        self.output_shape = (num_kernels, self.H - kernel_size + 1 + 2 * padding, self.W - kernel_size + 1 + 2 * padding)
        self.kernel_shape = (num_kernels, self.C, kernel_size, kernel_size)
        self.w = np.random.randn(*self.kernel_shape) / np.sqrt(np.prod(self.input_shape))
        self.b = np.random.randn(*self.output_shape)
        self.config = {}
        self.padding = padding
        self.stride = stride
        self.optimizer = optimizer

    def compile(self,lr, reg):
        self.config = {'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_b':np.zeros(self.b.shape),
                       'v_b':np.zeros(self.b.shape),
                       'm_w':np.zeros(self.w.shape),
                       'v_w':np.zeros(self.w.shape),
                       't':30}
    def forward(self,z, training = False):
        self.z = z.reshape(z.shape[0],*self.input_shape) #self.z = z.reshape(self.input_shape) 
        self.pad_z = np.zeros((z.shape[0],self.C, self.H + 2 *self.padding, self.W + 2 *self.padding))
        a = np.zeros((z.shape[0],*self.output_shape)) #a = np.zeros((self.output_shape)) #
        for n in range(z.shape[0]):
            a[n] += self.b    #a += self.b OLHAR ISSO
            for i in range(self.F):
                for j in range(self.C):
                    self.pad_z[n,j] = np.pad(self.z[n,j], pad_width = self.padding)
                    a[n,i] += scipy.signal.correlate2d(self.pad_z[n,j], self.w[i,j], mode = 'valid') #a[n,i] = z[n,j], w[i,j]

        return a

    def backward(self, dz, y):
        z = self.z
        dw = np.zeros(self.kernel_shape)
        db = np.sum(dz,axis = 0)
        dx = np.zeros((z.shape[0],*self.input_shape)) #dx = np.zeros(self.input_shape) 



        for n in range(z.shape[0]):
            for i in range(self.F):
                for j in range(self.C):
                    dw[i,j] += scipy.signal.correlate2d(self.pad_z[n,j],dz[n,i], mode = 'valid') #[i,j] = [n,j], [n,i]
                    #p = int((dx.shape[-1] + self.w.shape[-1] - dz.shape[-1] - 1) / 2) # Ho = H - HH + 1 + 2p  ==== OR ==== p = self.kernel_shape[-1]-1
                    dz_padded = np.pad(dz[n,i], pad_width= (self.kernel_shape[2] - 1 - self.padding))
                    dx[n,j] +=  scipy.signal.convolve2d(dz_padded, self.w[i,j], mode = 'valid') #[n,j] = [n,i], [i,j] #[self.padding : self.H - self.padding,self.padding : self.W - self.padding]


        #Apply Optimizer:
        w, b, self.config = self.optimizer(self.b,self.w,db,dw,self.config)

        self.w = w
        self.b = b

        return dx
    

class Flatten():
    def __init__(self, in_shape):
        C, H, W = in_shape
        self.in_shape = in_shape
        self.out_shape = C * H * W
        self.config = {}
        self.w = np.ndarray([])
        self.b = np.ndarray([])

    def compile(self,lr, reg):
        self.config = {'learning_rate': lr}

    def forward(self,x, training = False):
        x_flat = x.reshape((x.shape[0], self.out_shape))
        #x_flat = np.reshape(x, self.out_shape)
        return x_flat

    def backward(self, dz_flat, y):
        dz = dz_flat.reshape((dz_flat.shape[0],*self.in_shape))
        #dz_flat = np.reshape(dz, self.in_shape)
        return dz
    

class MaxPool():
    def __init__(self, in_shape, pool_size = 2, stride = 2):
        self.in_shape = in_shape
        self.stride = stride
        self.pool = pool_size
        self.config = {}
        self.w = np.ndarray([])
        self.b = np.ndarray([])

    def compile(self,lr, reg):
        self.config = {'learning_rate': lr}

    def forward(self,x, training = False):
        self.x = x
        stride = self.stride
        N, C, H, W = self.x.shape
        HH = H // stride
        WW = W // stride
        a = np.zeros((N,C,HH,WW))
        for n in range(N):
            for c in range(C):
                for h in range(HH):
                    for w in range(WW):
                        a[n,c,h,w] = np.max(x[n,c,h*stride : (h)*stride+self.pool , w*stride : (w)*stride+self.pool])
        return a

    def backward(self, da, y):
        x = self.x
        stride = self.stride
        N, C, H, W = self.x.shape
        HH = H // stride
        WW = W // stride
        dx = np.zeros((N,C,H,W))
        for n in range(N):
            for c in range(C):
                for h in range(HH):
                    for w in range(WW):
                        X = np.argmax(x[n,c,h*stride : (h)*stride+self.pool , w*stride : (w)*stride+self.pool])//self.pool 
                        Y = np.argmax(x[n,c,h*stride : (h)*stride+self.pool , w*stride : (w)*stride+self.pool])%self.pool
                        dx[n,c,h*stride +X , w*stride +Y] = da[n,c,h,w]
        return dx