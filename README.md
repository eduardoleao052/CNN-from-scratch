# CV_Models
## Full implementation of most used Computer Vision layers in plain code (numpy)

### Inspiration
- This idea started as a personal project after reading <i>Neural Networks and Deep Learning</i> by Michael Nielsen.
- The OOP-based approach was built after watching CS231n on youtube, and that is the current implementation.

### Requirements
- The only packaged used for the model is numpy. Other libraries are listed on requirements.txt.
- Note: scipy is used for faster implementation of Correlation and Convolution. I also made fully numpy-based implementations. They work and are in the functions.py file. The scipy implementation is only being used due to efficiency gains in training.
- To setup a miniconda virtual environment, run on terminal:
```
conda create -n environment_name python=3.8
```
- The requirements can be installed on a virtual environment with the command
```
pip install -r requirements.txt
```
- Note: The training is only implemented on CPU (no torch, tensorflow or CUDA support).
- To run, install the necessary requirements and a image dataset (.csv format).
- There must be a training and a test files. The files must have the label as the first column, and the features as the remaining columns.
- You can download your image file in the data directory.
  
### Training
- To train a CNN on your image dataset, go into run.sh and set the flag to --train, chose your --train_data file (full path to your training data file), --test_data (should be inside data directory), and a -to_path (.json file that will store the model - you do not need to create it, just provide a name).
```
python3 run.py --train config.json your_text_file.txt -to_path name_of_json_that_will_store_model.json
```
- Run on terminal:
```
./run.sh
```
- Whenever you feel like the samples are good enough, you can kill the training at any time. This will NOT corrupt the model saved .json file, and you may proceed to testing and fine_tuning on smaller datasets.
- Note: for pretraining, a really large text corpus is usually necessary. I obtained good results with ~1M characters.
- Note: if you want to alter layers/dimensions, do so in the __init__ of Model at model_torch.py.

### Testing
- To test your RNN, go into run.sh and set the flag to --test, chose a -sample_size (number of characters to generate), a -seed (The start to the string your model generates, it has to "continue" it), and a -from_path (.json file that contains pretrained model).
```
python3 run.py --test -sample_size 400 -seed  -from_path name_of_pretrained_model_file.json
```
- Run on terminal:
```
./run.sh
```
- Note: for fine-tuning, a you can get adventurous with smaller text files. I obtained really nice results with ~10K characters, such as a small Shakespeare dataset and Bee Gees' songs.

### Results
- The Convolutional Neural Network implementation in main.py achieved 99.36% accuracy on the validation set of the MNIST handwritten digit dataset.
- The Recurrent Neural Network implementation in main.py achieved a loss of 1.22 with a 78 vocabulary size and ~2M tokens of training for 100,000 timesteps (32 batch_size, 200 n_iterations).
- The LSTM achieved a loss of 1.11 with the same settings.
- Training times seemed to be a little faster with GPU, but the improvement was not dramatic (maybe due to iterative and non-paralellizeable nature of RNNs).
- Total training times: RNN ~4h, LSTM ~10h on one GTX1070 Nvidia GPU.
- Result with ~4h of pretraining on reduced version of COCA (around 10M tokens) and ~1h of fine-tuning on <i>tiny_shakespeare</i> dataset:
  
```
CORIOLANUS:
I am the guilty us, friar is too tate.

QUEEN ELIZABETH:
You are! Marcius worsed with thy service, if nature all person, thy tear. My shame;
I will be deaths well; I say
Of day, who nay; embrace
The common on him;
To him life looks,
Yet so made thy breast,
Wrapte,
He kiss,
Take up;
From nightly:
Stand good.

MENENIUS HISHOP:
O felt people
Two slaund that strangely but conscience to me.

BENVOLIO:
Why, whom I come in his own share; so much for it;
For that O, they say they shall, for son that studies soul
Having done,
And this is the rest in this in a fellow.
```
- Note: results achieved with the model configuration exactly as presented in this repo.
- Thanks for reading!
