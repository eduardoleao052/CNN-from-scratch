# CV_Models
## Full implementation of most used Computer Vision layers in plain code (numpy)

### Inspiration
- This idea started as a personal project after reading <i>Neural Networks and Deep Learning</i> by Michael Nielsen.
- The OOP-based approach was built after watching CS231n on youtube, and that is the current implementation.

### Requirements
- The only required packages are numpy, logging, pandas and scipy (for faster implementation of Correlation and Convolution). The numpy-based implementations are in the functions.py file, and the scipy implementation is only being used due to efficiency gains in training.
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
- Run main.py to train CNN on given dataset. Feel free to alter layers (scaling down is recommended, implemented CNN size took ~40h to converge on my CPU).

### Results
- The Convolutional Neural Network implementation in main.py achieved 99.36% accuracy on the validation set of the MNIST handwritten digit dataset.
