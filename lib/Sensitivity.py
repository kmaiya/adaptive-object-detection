########################################################
# Purpose of This Script:                              #
#   - Define the class of the measurer                 #
#                                                      #
########################################################

import gc

import torch

from model_prepare import PreModel


class SensitivityMeasurer(PreModel):
    """
    This class serves as a sensitivity measurer that has two metrics from the paper
    Sensitivity and Generalization in Neural Networks: an Empirical Study
    """
    def __init__(self, model, module_list=None, cuda=True, model_intake_size=416, batch_size=1, channel_num=3):
        super(SensitivityMeasurer, self).__init__(model, module_list, cuda, model_intake_size, batch_size, channel_num)
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

    def compute_neuron_jacobian(self, inputs, layer_idx, channel_idx, neuron_idx):
        """
        This function computes the Jacobian of one neuron's activation
        :param inputs: the variables of the Jacobian
        :param layer_idx: the index of the layer of interest
        :param channel_idx: the index of the channel of interest
        :param neuron_idx: the place of the neuron
        :return: the Jacobian vector
        """
        inputs.requires_grad = True  # make sure that the input is tracked
        activation = self.get_nth_neuron(inputs, layer_idx, channel_idx, neuron_idx)  # a forward pass
        activation.backward()  # backward pass that computes the gradient
        return inputs.grad

    def compute_channel_jacobian(self, inputs, layer_idx, channel_idx, mode=None):
        """
        This function computes the Jacobian of one channel of designated layer
        :param inputs: the variables of the Jacobian
        :param layer_idx: the index of the layer of interest
        :param channel_idx: the index of the channel of interest
        :return: the Jacobian matrix of that channel
        """
        if mode is None:
            inputs.requires_grad = True
            height, width = self.size_count[layer_idx]
            outputs = self.get_nth_channel(inputs, layer_idx, channel_idx)
            Jacobian = []
            for width_i in range(width):
                for height_i in range(height):
                    outputs[:, height_i, width_i].backward(retain_graph=True)
                    Jacobian.append(inputs.grad)

            return Jacobian

        elif mode == "reduction_mean":
            inputs.requires_grad = False  # make sure that the input is tracked
            inputs.requires_grad = True
            activation = self.get_nth_channel(inputs, layer_idx, channel_idx)  # a forward pass
            if activation is not None:
                activation = activation.mean()
                activation.backward()  # backward pass that computes the gradient
                return inputs.grad
            else:
                return None

        raise ValueError("the mode has only reduction_mean as of now")

    def compute_layer_jacobian(self, inputs, layer_idx):
        """
        This function computes the Jacobian of one layer
        :param inputs: the variables of the Jacobian
        :param layer_idx: the layer of interest
        :return: the Jacobian
        """
        channel_count = self.channel_count[layer_idx]
        height, width = self.size_count[layer_idx]
        outputs = self.get_n_th_layer(inputs, layer_idx)
        Jacobian = []
        for channel_i in range(channel_count):
            for width_i in range(width):
                for height_i in range(height):
                    outputs[:, channel_i, height_i, width_i].backward(retain_graph=True)
                    Jacobian.append(inputs.grad)

        return Jacobian

    def compute_jacobian(self, inputs, outputs=None, mode=None):
        """
        This function computes the jacobian of the output of interest w.r.t. the input of interest
        :param inputs: the input of interest
        :param outputs: the outputs of interest, of type list or tuple,
                                of the format [layer_idx, channel_idx, neuron_idx]
        :return: the jacobian matrix (or vector)
        """
        # TODO: Think about how to efficiently compute the jacobian
        if mode is None:
            if len(outputs) == 3:
                # here implements the jacobian of a neuron
                return self.compute_neuron_jacobian(inputs, *outputs)
            if len(outputs) == 2:
                # here implements the jacobian of a channel
                return self.compute_channel_jacobian(inputs, *outputs)
            if len(outputs) == 1:
                # here implements the jacobian of a layer
                return self.compute_layer_jacobian(inputs, *outputs)

        elif mode == "reduction_mean_channel":
            jacobian = []
            for layer_idx in range(self.layer_num):
                for channel_idx in range(self.channel_count[layer_idx]):
                    jacobian.append(self.compute_channel_jacobian(inputs,
                                                                  layer_idx, channel_idx, mode=mode).clone().cpu())
                    torch.cuda.empty_cache()
            return jacobian

        elif mode == "reduction_mean_layer":
            jacobian = []
            for layer_idx in range(self.layer_num):
                jacobian.append(self.compute_layer_jacobian(inputs,
                                                            layer_idx))

    def gradient_incr(self, inputs):
        """
        This function computes the gradient from adjacent layers
        :param inputs: the input
        :return: the gradients
        """
        inputs = inputs.to(self.device)
        gradients = []
        for i in range(self.layer_num):
            gradients.append(self.forward_pass(inputs, input_index=i, output_index=i + 1))
        return gradients

    def compute_paper(self, img):
        if self.cuda:
            img = img.to(self.device)

        alpha = 1
        output = self.model(input_index=alpha, output_index=self.layer_num + 10)
        output_index = 10  # TODO get the index
        output[output_index].backward()
        score_function = self.module_list[alpha][0].weight.grad  # take the derivative w.r.t. weights
