import gc
import os
import shutil

import torch
import torch.distributions as distribution
import torch.nn as nn
import torch.optim as optimizers
from scipy.misc import imsave as save_img

from model_prepare import PreModel


class Visualizer(PreModel):
    """
    This should serve as a framework for any sort of models and later some other models
    """

    def __init__(self, model, module_list=None, initial_size=56, model_intake_size=416,
                 batch_size=1, channel_num=3, cuda=True):
        """
        This function initializes the visualizer
        :param module_list: the module list from the model
        :param initial_size: the initial image size
        :param model_intake_size: the model's input image size
        :param batch_size: the batch size of the image to optimize
        :param channel_num: the number of channels
        :param cuda: the device to put the model and data on
        """
        super(Visualizer, self).__init__(model, module_list, cuda, model_intake_size, batch_size, channel_num)

        self.z_image = self.image_init(initial_size)  # initialize the first image
        self.z_image.requires_grad = True
        self.init_size = initial_size
        self.new = True
        # get the scale
        if isinstance(model_intake_size, int):
            self.input_generator = nn.UpsamplingBilinear2d(size=(model_intake_size, model_intake_size))
        else:
            self.input_generator = nn.UpsamplingBilinear2d(size=model_intake_size)

    @staticmethod
    def cast(value, d_type=torch.float32):
        """
        This function takes care of the casting because it is too tedious in Pytorch
        :param value: the number to convert
        :param d_type: the data type
        :return: the cast value
        """
        return torch.tensor(value).type(d_type)

    @staticmethod
    def tanh(x, scaling=0.2):
        """
        This function put boundaries upon the image so that all of its values are of [0-1]
        :param x: the input
        :param scaling: the scaling of the tanh function
        :return: the scaled input
        """
        return torch.nn.Tanh()(scaling * x) / 2 + 0.5

    def random_init(self, image_size, mean=0, std=1, normal=True):
        """
        This function initializes a new original image
        :param image_size: the image size of the model
        :param mean: the mean of the distribution
        :param std: the standard deviation of the distribution
        :param normal: if normal distribution
        :return: the image initialized
        """
        if normal:
            sampler = distribution.Normal(self.cast(mean), self.cast(std))
        else:
            sampler = distribution.Uniform(-1.5, 1.5)

        if not isinstance(image_size, list):
            size = torch.Size([self.batch_size, self.channel_num, image_size, image_size])
        else:
            size = torch.Size([self.batch_size, self.channel_num, image_size[0], image_size[1]])

        return sampler.sample(size)

    def noise_gen(self, image_size, noise_ratio=0.2, mean=0, std=1):
        """
        This function allows to create a mask onto
        :param self: the class itself
        :param image_size: the image size of the model
        :param noise_ratio: the noise ratio imposed onto the mask
        :param mean: the mean of the mask
        :param std: the standard deviation of the distribution
        :return: the random mask
        """
        return self.random_init(image_size, mean, std, normal=False) * noise_ratio

    def image_init(self, image_size):
        """
        This function allows to create a mask onto
        :param self: the class itself
        :param image_size: the image size of the model
        :return: the random mask
        """
        image = self.random_init(image_size, normal=False)
        image.require_grad = True

        return image

    def generate_input_image(self):
        """
        This function creates an input image for the model
        :return: the input image
        """
        return self.tanh(self.input_generator(self.z_image.to(self.device)))

    def upscale_image(self, scale_step, mask=True):
        """
        This function up-scales the current image
        :param mask: if to mask the up-scaled image
        :param scale_step: the number of steps
        """
        scale_ratio = (self.input_size / self.init_size) ** (1 / scale_step)
        up_scaler = nn.UpsamplingNearest2d(scale_factor=scale_ratio)
        z_image = up_scaler(self.z_image)
        if mask:
            z_image = z_image + self.noise_gen(z_image.shape[-1])

        self.z_image = z_image.clone().detach().requires_grad_(True)

    @staticmethod
    def mkdir_single(path):
        """
        This function tries to create a file folder
        :param path: the path to create
        """
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    def mkdir(self, data_path, layer_idx, channel_idx):
        """
        This function creates the folders to store the visualizations
        :param data_path: the data path to store in
        :param layer_idx: the layer number
        :param channel_idx: the channel index
        """
        self.mkdir_single(f"{data_path}")
        self.mkdir_single(f"{data_path}/layer{layer_idx}")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Channel{channel_idx}")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Channel{channel_idx}/Color")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Channel{channel_idx}/Mono0")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Channel{channel_idx}/Mono1")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Channel{channel_idx}/Mono2")

    @staticmethod
    def del_dir(data_path):
        try:
            shutil.rmtree(data_path)
        except FileNotFoundError:
            pass

    def save_image(self, data_path, img_idx, layer_idx, channel_idx, epoch_idx, id_batch=0, step_idx=0):
        """
        This function saves the image
        :param img_idx: the index of the image
        :param data_path: the data path to save to
        :param layer_idx: the layer index of the images
        :param channel_idx: the channel index of the images
        :param epoch_idx: the epoch index this image is from
        :param id_batch: the id in the batch
        :param step_idx: the step index
        """

        self.mkdir(data_path, layer_idx, channel_idx)

        img_to_save = self.generate_input_image()
        save_img(f"{data_path}/layer{layer_idx}/Channel{channel_idx}/Color/{img_idx}S{step_idx}E{epoch_idx}.jpg",
                 img_to_save[id_batch].detach().cpu().permute(1, 2, 0)[:, :, :])
        save_img(f"{data_path}/layer{layer_idx}/Channel{channel_idx}/Mono0/{img_idx}S{step_idx}E{epoch_idx}.jpg",
                 img_to_save[id_batch].detach().cpu().permute(1, 2, 0)[:, :, 0])
        save_img(f"{data_path}/layer{layer_idx}/Channel{channel_idx}/Mono1/{img_idx}S{step_idx}E{epoch_idx}.jpg",
                 img_to_save[id_batch].detach().cpu().permute(1, 2, 0)[:, :, 1])
        save_img(f"{data_path}/layer{layer_idx}/Channel{channel_idx}/Mono2/{img_idx}S{step_idx}E{epoch_idx}.jpg",
                 img_to_save[id_batch].detach().cpu().permute(1, 2, 0)[:, :, 2])

    def clear_cuda_memory(self):
        """
        This function clears the cuda memory
        """
        try:
            z_image = self.z_image.clone().detach().cpu().data
            del self.z_image
            torch.cuda.empty_cache()
            self.z_image = z_image.requires_grad_()
        except AttributeError:
            pass

    def refresh(self, size):
        if not self.new:
            del self.z_image
            gc.collect()
            self.z_image = self.image_init(size)
            self.new = 1

    def one_pass_neuron(self, layer_idx, channel_idx, optimizer, neuron_x=0, neuron_y=0):
        """
        This function does the one pass on a neuron
        :param layer_idx: the layer index of the neuron
        :param channel_idx: the channel index of the neuron
        :param neuron_x: the x coordinate of the neuron
        :param neuron_y: the y coordinate of the neuron
        :param optimizer: the optimizer
        """
        img = self.generate_input_image()
        output = self.forward_pass(img, layer_idx)
        _, _, x, y = output.shape
        neuron_x = x if neuron_x > x else neuron_x
        neuron_y = y if neuron_y > y else neuron_y
        loss = - output.mean(0)[channel_idx, neuron_x, neuron_y]
        self.backward_pass(loss, optimizer)

    def one_pass_channel(self, layer_idx, channel_idx, optimizer):
        """
        This function performs a forward pass
        :param layer_idx: the layer index
        :param channel_idx: the channel index
        :param optimizer: the optimizer
        """
        img = self.generate_input_image()
        output = self.forward_pass(img, layer_idx)
        output_channels = output.mean(-1).mean(-1).mean(0)
        loss = - output_channels[channel_idx]
        self.backward_pass(loss, optimizer)

    @staticmethod
    def backward_pass(loss, optimizer):
        """
        This function performs one backward pass to optimize the parameters
        :param loss: the loss function to optimize for
        :param optimizer: the optimizer
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def vanilla_visualize(self, layer_idx, channel_idx, epochs, optimizer=optimizers.Adam,
                          data_path="vanilla_vis", learning_rate=0.001, weight_decay=0, single_pass=None):
        """
        This function performs the vanilla visualization by optimization
        :param layer_idx: the layer index
        :param channel_idx: the channel index
        :param epochs: the number of epochs
        :param optimizer: the optimizer used
        :param data_path: the data path to store the images
        :param learning_rate: the learning rate
        :param weight_decay: the weight decay, used for regularization
        :param single_pass: the function to forward
        """
        self.refresh(self.input_size)
        print(f"Start to visualize channel {channel_idx} layer {layer_idx}")
        self.del_dir(f"{data_path}/layer{layer_idx}/Channel{channel_idx}")

        if single_pass is None:
            single_pass = self.one_pass_channel

        # prepares the image
        optimizer_instance = optimizer([self.z_image], lr=learning_rate, weight_decay=weight_decay)

        img_idx = 0
        for epoch in range(epochs):
            single_pass(layer_idx, channel_idx, optimizer_instance)

            # save image
            if epoch % 5 == 0:
                self.save_image(data_path, epoch, layer_idx, channel_idx, epoch)
                img_idx += 1

        self.save_image(data_path, epoch, layer_idx, channel_idx, epochs, step_idx=0)
        img_idx += 1

    def multistep_visualize(self, layer_idx, channel_idx, epochs=3, optimizer=optimizers.Adam,
                            data_path="multi_vis", learning_rate=0.001, weight_decay=0, scale_step=12,
                            initial_size=30, forward_pass=None):
        """
        This function does the visualization
        :param layer_idx: the index of the layer to visualize
        :param channel_idx: the channel index of the layer to visualize
        :param epochs: the number of epochs to train
        :param optimizer: the optimizer of the image
        :param data_path: the data path to store the images
        :param learning_rate: the learning rate of the optimizer
        :param weight_decay: the weight decay of the optimizer
        :param initial_size: the initial input size
        :param forward_pass: the function that performs the forward pass
        """

        self.refresh(initial_size)

        print(f"Start to visualize channel {channel_idx} layer {layer_idx} with Multistep")
        self.del_dir(f"{data_path}/layer{layer_idx}/Channel{channel_idx}")

        if forward_pass is None:
            forward_pass = self.one_pass_channel

        img_idx = 0
        for step in range(scale_step):
            # prepares the image
            optimizer_instance = optimizer([self.z_image], lr=learning_rate, weight_decay=weight_decay)

            for epoch in range(epochs // scale_step):
                forward_pass(layer_idx, channel_idx, optimizer_instance)

                # save image
                if epoch * (step + 1) % 5 == 0:
                    self.save_image(data_path, img_idx, layer_idx, channel_idx, epoch, step_idx=step)
                    img_idx += 1

            self.save_image(data_path, img_idx, layer_idx, channel_idx, epochs, step_idx=step)
            img_idx += 1
            if self.cuda:  # this should put self.z_image onto CPU
                self.clear_cuda_memory()
            gc.collect()
            # this should put z_img back to
            self.upscale_image(scale_step)
            # this clears system memory
            gc.collect()

        if self.cuda:
            self.clear_cuda_memory()

        self.new = False

    def visualize_whole_layer(self, layer_idx, epochs=3, optimizer=optimizers.Adam,
                              data_path=".", learning_rate=None, weight_decay=None):
        """
        This function allows to visualize a whole layer at once
        Proceed with caution since Pytorch's GRAM management is not quite as good as the function requires
        """
        for channel_idx in range(self.channel_count[layer_idx]):
            self.multistep_visualize(layer_idx=layer_idx, channel_idx=channel_idx, epochs=epochs, optimizer=optimizer,
                                     data_path=data_path, learning_rate=learning_rate, weight_decay=weight_decay)

    def visualize_all_model(self, epochs=3, optimizer=optimizers.Adam,
                            data_path="visualization", learning_rate=None, weight_decay=None):
        """
        This function allows to visualize all the channels in a model
        Proceed with caution since Pytorch's GRAM management is not quite as good as the function requires
        """
        for layer_idx in range(self.layer_num):
            self.visualize_whole_layer(layer_idx=layer_idx, epochs=epochs, optimizer=optimizer,
                                       data_path=data_path, learning_rate=learning_rate, weight_decay=weight_decay)
