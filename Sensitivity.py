########################################################
# Purpose of This Script:                              #
#   - Define the class of the measurer                 #
#                                                      #
########################################################

import gc

import torch


class SensitivityMeasurer:
    """
    This class serves as a sensitivity measurer that has two metrics from the paper
    Sensitivity and Generalization in Neural Networks: an Empirical Study
    """
    def __init__(self, model, module_list=None, cuda=True, model_intake_size=416, batch_size=1, channel_num=3):
        """
        This function initializes the object
        :param model: the model of interest
        :param module_list: the module list of the model, if provided, otherwise will be written by list(model.children())
        :param cuda: whether to use cuda, which will be limited by hardware
        :param model_intake_size: the input size of the image, could be a int or a list/tuple
        :param batch_size: the batch size of the input
        :param channel_num: the number of input channel
        """
        # check if use cuda
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # get the models and module list
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        if module_list is not None:
            self.module_list = module_list
        else:
            self.module_list = list(model.children())
        for i, layer in enumerate(self.module_list):
            self.module_list[i] = layer.to(self.device)

        self.layer_num = len(self.module_list)  # the number of layers of the model
        self.channel_count = []  # the number of channels of each layer
        self.size_count = []  # elements of format [[height, width]]
        self.batch_size = batch_size  # the batch size of the input to the model
        self.channel_num = channel_num

        if isinstance(model_intake_size, int):
            self.width = model_intake_size
            self.height = model_intake_size
        else:
            self.width, self.height = model_intake_size

        self._module_list_channel_count()

        gc.collect()
        torch.cuda.empty_cache()

    def _module_list_channel_count(self):
        """
        This function counts the number of channels in each layer
        Note that for any model with residue network, we have to overload this method
        """
        place_holder = torch.randn(self.batch_size, self.channel_num,
                                   self.height, self.width).to(self.device)
        for i, layer in enumerate(self.module_list):
            self.size_count.append(0)
            self.module_list[i] = layer.to(self.device)
            try:
                place_holder = self.module_list[i](place_holder)
                output_shape = place_holder.shape
                self.channel_count.append(output_shape[1])
                for channel_idx in range(output_shape[1]):
                    self.size_count[-1] = output_shape[2:]
            except NotImplementedError:
                self.size_count[-1] = torch.Size([0, 0])

    def forward_pass(self, img, layer_idx):
        """
        This function gets the nth layer output
        :param img: the input image
        :param layer_idx: the layer index of interest
        :return: the output at that layer
        """
        img = img.to(self.device)
        for i, layer in enumerate(self.module_list):
            try:
                img = layer(img)
            except NotImplementedError:
                pass
            if i == layer_idx:
                return img
        raise ValueError(f"Layer {layer_idx} is larger than the maximum layer count {self.layer_num}")

    @staticmethod
    def reduction(reduction_tensor, reduction_type):
        """
        This function does the dimensionality reduction
        :param reduction_tensor: the tensor to reduce dimensionality of
        :param reduction_type: the type of reduction to perform, one of 'mean' or 'sum'
        :return: the reduced tensor
        """
        if reduction_type is None:
            return reduction_tensor
        if reduction_type == 'mean':
            return reduction_tensor.mean()
        if reduction_type == 'sum':
            return reduction_tensor.sum()

    def get_n_th_layer(self, img, layer_idx, reduction=None):
        """
        This function wraps around the get_n_th_layer_act_core function to introduce the ability of reduction
        :param img: the input image
        :param layer_idx: the the index of the layer of interest
        :param reduction: the reduction method, one of 'sum' or 'mean'
        :return: the n-th layer's activation
        """
        return self.reduction(self.forward_pass(img, layer_idx), reduction)

    def get_nth_channel(self, img, layer_idx, channel_idx, reduction=None):
        """
        This function gets the n-th channel of the n-th layer's output
        :param img: the input image
        :param layer_idx: the layer index
        :param channel_idx: the channel index
        :param reduction: the reduction method, one of 'sum' or 'mean'
        :return: the n-th channel output
        """
        try:
            return self.reduction(self.get_n_th_layer(img, layer_idx)[:, channel_idx, :, :], reduction)
        except IndexError:
            try:
                return self.reduction(self.get_n_th_layer(img, layer_idx)[:, channel_idx, :], reduction)
            except IndexError:
                pass

    def get_nth_neuron(self, img, layer_idx, channel_idx, neuron_idx):
        """
        This function gets the n-th neuron of the n-th channel of the n-th layer's output
        :param img: the input image
        :param layer_idx: the index of the layer of interest
        :param channel_idx: the index of the channel of interest
        :param neuron_idx: the index or the coordinate of the neuron of interest
        :return: the activation of that neuron
        """
        if isinstance(neuron_idx, list) or isinstance(neuron_idx, tuple):
            x_idx, y_idx = neuron_idx
        else:
            x_idx = y_idx = neuron_idx
        return self.get_nth_channel(img, layer_idx, channel_idx, reduction=None)[:, x_idx, y_idx]

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

        elif mode == "reduction_mean":
            jacobian = []
            for layer_idx in range(self.layer_num):
                for channel_idx in range(self.channel_count[layer_idx]):
                    jacobian.append(self.compute_channel_jacobian(inputs,
                                                                  layer_idx, channel_idx, mode=mode).clone().cpu())
                    torch.cuda.empty_cache()
            return jacobian
