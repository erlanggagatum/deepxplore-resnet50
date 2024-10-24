
# DeepXplore on ResNet-50 & CIFAR-10

This is a code implementation of neural network testing tools (DeepXplore) on ResNet-50 using CIFAR-10 dataset. Below is the env requirements and how to setup docker to run the script. 

## Requirements
```
python==3.5
tensorflow==1.15.0
keras==2.2.4
Pillow
imageio
```

## Running on local machine
After fulfiling the requirements, run the following command. The description of the arguments are described in the next section.
```
gen_diff_resnet50-cifar10.py blackout 0.5 0.3 7 250 250 0.2
```

## Runing on Docker (suggested)
Dockerfile already created inside ```Dockerfile```, which will automatically setup the code inside docker container python:3.5. To setup the container, run the following command inside the project directory containing Dockerfile:
```
docker build -t deepxplore-cifar10 .
```
After it finished, run the container using the following command:
```
docker run -v $(pwd)/generated_inputs_cifar10_docker:/app/generated_inputs_cifar10 deepxplore-cifar10
```
This command will mount the folder of generated input inside the container with a local directory, which contains the images as a result from perturbation using DeepXplore tool. This test automatically run ```python gen_diff_resnet50-cifar10.py blackout 0.5 0.3 7 250 250 0.2```. The description of the argument are described as follows based on DeepXplore documentation:
```
python script.py transformation weight_diff weight_nc step seeds grad_iterations threshold
```

```
transformation = "realistic transformation type", choices=['light', 'occl', 'blackout']
weight_diff = "weight hyperparm to control differential behavior", type=float
weight_nc = "weight hyperparm to control neuron coverage", type=float
step = "step size of gradient descent", type=float
seeds = "number of seeds of input", type=int
grad_iterations = "number of iterations of gradient descent", type=int
threshold = "threshold for determining neuron activated", type=float
```

## Weights
Weights is not available on github due to size limit. Therefore, please download it in the 
[following link](https://drive.google.com/drive/folders/1nIJxMXNlgaBa9nlE9zXTWec1PQ_ZH8Qd?usp=sharing)
, and extract it in the root folder. As an additional information, resnet50_5k3e means Resnet-50 was trained using 5k sample dataset in 3 epochs. Pretrained ResNet-50 are frozen, and the training only performed in the custom classification head.

## Other information
The whole originaly from [DeepXplore official repository](https://github.com/peikexin9/deepxplore), and it is being modified to match the structure of ResNet-50 and its implementation on CIFAR-10 dataset. Moreover, some updates has been made to match the newer version of python (v3.5).

## Author
- [@erlanggagatum](https://www.github.com/erlanggagatum)