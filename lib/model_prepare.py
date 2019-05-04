import gc

import torch


class PreModel:
    """
    This class prepares the model of interest for a few useful functions
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

        # This here ta
        if isinstance(model_intake_size, int):
            self.width = model_intake_size
            self.height = model_intake_size
        else:
            self.width, self.height = model_intake_size

        try:
            self._module_list_channel_count()
        except RuntimeError:
            self.size_count = None

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

    def forward_pass(self, img, layer_idx=None, output_index=None, input_index=None):
        """
        This function gets the nth layer output
        :param img: the input image
        :param layer_idx: the layer index of interest
        :param output_index: the index of the output layer
        :param input_index: the index of the input layer
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
        return img

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
