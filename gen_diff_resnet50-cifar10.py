'''
usage: gen_diff_resnet50-cifar10.py blackout 0.5 0.3 7 250 250 0.2
'''

from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input

from resnet50_3k3e import ResNet50_3k3e
from resnet50_5k3e import ResNet50_5k3e
from configs import bcolors
from utils import *

import tensorflow as tf
import imageio
import numpy as np
from tensorflow.keras import backend as K

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(32, 32), type=tuple)

args = parser.parse_args()

print('=========== args ==========', args)

# input image dimensions
img_rows, img_cols = 28, 28

# load dataset cifar10
(_, _) , (x_test, _) = tf.keras.datasets.cifar10.load_data()

def preprocess_image_input(input_images):
    input_images = input_images.astype('float32')
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims

x_test = preprocess_image_input(x_test)

input_tensor = tf.keras.layers.Input(shape=(32,32,3))

# load multiple models sharing same input tensor
resnet50_3k3e = ResNet50_3k3e(input_tensor=input_tensor)
resnet50_5k3e = ResNet50_5k3e(input_tensor=input_tensor)

print('===== ResNet50_3k3e ======')
print(resnet50_3k3e.summary())
print('===== ResNet50_5k3e ======')
print(resnet50_5k3e.summary())

# init coverage table
resnet50_3k3e_layer_dict, resnet50_5k3e_layer_dict = init_coverage_tables([resnet50_3k3e, resnet50_5k3e])

# ==============================================================================================
# start gen inputs
print('Start gen input. Seed:', args.seeds)
random.seed(args.seeds)
np.random.seed(args.seeds)

for _ in range(args.seeds):
    print("======================= Step ", _, "=======================")
    
    gen_img = random.choice(x_test)
    gen_img = np.expand_dims(gen_img, axis=0)

    orig_img = gen_img.copy()
    
    # first check if input already induces differences
    label1 = np.argmax(resnet50_3k3e.predict(gen_img))
    label2 = np.argmax(resnet50_5k3e.predict(gen_img))

    # Check inconsistent prediction label
    if not label1 == label2:    
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}'.format(label1, label2) + bcolors.ENDC)

        update_coverage(gen_img, resnet50_3k3e, resnet50_3k3e_layer_dict, args.threshold)
        update_coverage(gen_img, resnet50_5k3e, resnet50_5k3e_layer_dict, args.threshold)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f'
              % (len(resnet50_3k3e_layer_dict), neuron_covered(resnet50_3k3e_layer_dict)[2], 
                 len(resnet50_5k3e_layer_dict), neuron_covered(resnet50_5k3e_layer_dict)[2]) + bcolors.ENDC)
        
        averaged_nc = (neuron_covered(resnet50_3k3e_layer_dict)[0] + neuron_covered(resnet50_5k3e_layer_dict)[0]) / float(
            neuron_covered(resnet50_3k3e_layer_dict)[1] + neuron_covered(resnet50_5k3e_layer_dict)[1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)
        
        gen_img_deprocessed = np.squeeze(gen_img, axis=0)
        
        # save the result to disk
        imageio.imwrite('./generated_inputs_cifar10/' + 'assignment2_already_differ_' + str(label1) + '_' + str(
            label2) + '.png', gen_img_deprocessed)
        continue
    
    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(resnet50_3k3e_layer_dict)
    layer_name2, index2 = neuron_to_cover(resnet50_5k3e_layer_dict)
    
    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(resnet50_3k3e.get_layer('before_classification').output[..., orig_label])
        loss2 = K.mean(resnet50_5k3e.get_layer('before_classification').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(resnet50_3k3e.get_layer('before_classification').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(resnet50_5k3e.get_layer('before_classification').output[..., orig_label])
        
    loss1_neuron = K.mean(resnet50_3k3e.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(resnet50_5k3e.get_layer(layer_name2).output[..., index2])
    layer_output = (loss1 + loss2) + args.weight_nc * (loss1_neuron + loss2_neuron)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, grads])
    
    # run gradient ascent for grad_iterations steps
    for iters in range(args.grad_iterations):
        loss_value1, loss_value2, grads_value = iterate([gen_img])
        
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        
        # get the prediction result after perturbation
        predictions1 = np.argmax(resnet50_3k3e.predict(gen_img))
        predictions2 = np.argmax(resnet50_5k3e.predict(gen_img))    
                
        print('Grad iter:',iters, 'predictions1', predictions1, 'predictions2', predictions2)

        if not predictions1 == predictions2:
            print(bcolors.OKGREEN + '===== Different result found!' + bcolors.ENDC)
            update_coverage(gen_img, resnet50_3k3e, resnet50_3k3e_layer_dict, args.threshold)
            update_coverage(gen_img, resnet50_5k3e, resnet50_5k3e_layer_dict, args.threshold)
            
            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f'
                  % (len(resnet50_3k3e_layer_dict), neuron_covered(resnet50_3k3e_layer_dict)[2], 
                     len(resnet50_3k3e_layer_dict), neuron_covered(resnet50_5k3e_layer_dict)[2]) + bcolors.ENDC)
            
            averaged_nc = (neuron_covered(resnet50_3k3e_layer_dict)[0] + neuron_covered(resnet50_5k3e_layer_dict)[0]) / float(
                neuron_covered(resnet50_3k3e_layer_dict)[1] + neuron_covered(resnet50_5k3e_layer_dict)[1])
            
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = np.squeeze(gen_img, axis=0)
            orig_img_deprocessed = np.squeeze(orig_img, axis=0)
            
            # save the result to disk
            imageio.imwrite('./generated_inputs_cifar10/' + 'assignment2_' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '.png',
                   gen_img_deprocessed)
            imageio.imwrite('./generated_inputs_cifar10/' + 'assignment2_' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_orig.png',
                   orig_img_deprocessed)
            print(bcolors.OKGREEN + 'image saved!' + bcolors.ENDC)
            break