import numpy as np

def Adam(b,w,db,dw,config):
    #UPDATE W
    config['m_w'] = (config['m_w']*config['beta1'] + (1 - config['beta1']) * dw) / (1- config['beta1']**config['t'])
    config['v_w'] = (config['v_w']*config['beta2'] + (1 - config['beta2']) * np.square(dw)) / (1- config['beta2']**config['t'])

    next_w = w - (config["learning_rate"] * config['m_w']) / (np.sqrt(config['v_w']) + config['epsilon']) - config["regularization"] * config["learning_rate"] * w
    #print("BREAKDOWN W")
    #print(config['m_w'])
    #print(config['v_w'])
    #print("END")

    #print("w =========================")
    #print(w)
    #print((config["learning_rate"] * config['m_w']) / (np.sqrt(config['v_w']) + config['epsilon']))
    #UPDATE B
    config['m_b'] = (config['m_b']*config['beta1'] + (1 - config['beta1']) * db) / (1- config['beta1']**config['t'])
    config['v_b'] = (config['v_b']*config['beta2'] + (1 - config['beta2']) * np.square(db)) / (1- config['beta2']**config['t'])
    
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