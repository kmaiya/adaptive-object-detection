########################################################
# Purpose of This Script:                              #
#   - An inheritance from the YOLO                     #
#   implementation for better flexibility              #
#                                                      #
########################################################


import torch

from YOLOv3.models import Darknet


class YOLO(Darknet):
    """
    This is a modified version of YOLOv3 from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py
    This modification allows for more flexibility when it comes to investigation into the model
    """

    def __init__(self, config_path, img_size=416):
        """
        Simple inheritance
        :param config_path: the path of the configuration file
        :param img_size: the size of the input image
        """
        super(YOLO, self).__init__(config_path, img_size)

    def forward(self, x, layer_idx=None, output_index=None, input_index=None):
        """
        This function should allow for the following operations:
            1. get output from intermediate layers
            2. get intermediate gradient between layers of index
        :param x: the input
        :param layer_idx: the layer index for the above operations
        :param output_index: the index of the output layer to take derivatives
        :param input_index: the input variable to take derivatives w.r.t.
        :return: depends
        """
        get_layer = layer_idx is not None
        get_grad = output_index is not None and input_index is not None
        activation_dict = dict()
        output = []
        layer_outputs = []
        if not get_grad:
            for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
                if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                    x = module(x)
                elif module_def["type"] == "route":
                    layer_i = [int(x) for x in module_def["layers"].split(",")]
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                elif module_def["type"] == "shortcut":
                    layer_i = int(module_def["from"])
                    x = layer_outputs[-1] + layer_outputs[layer_i]
                elif module_def["type"] == "yolo":
                    # Train phase: get loss
                    x = module(x)
                    output.append(x)
                layer_outputs.append(x)
                if get_layer:
                    if i in layer_idx:
                        activation_dict[i] = x

            if get_layer:
                return activation_dict
            return torch.cat(output, 1)

        else:
            assert input_index < output_index
            for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
                if i == input_index:
                    x = x.detach().clone()  # create a new leaf variable
                    inputs = x  # shares the same address
                    x.requires_grad = True  # this also changes inputs.requires_grad
                    # since they are pointing to the same thing
                if i == output_index:
                    x.mean().backward()
                    return inputs.grad
                if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                    x = module(x)
                elif module_def["type"] == "route":
                    layer_i = [int(x) for x in module_def["layers"].split(",")]
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                elif module_def["type"] == "shortcut":
                    layer_i = int(module_def["from"])
                    x = layer_outputs[-1] + layer_outputs[layer_i]
                elif module_def["type"] == "yolo":
                    # Train phase: get loss
                    x = module(x)
                    output.append(x)
                layer_outputs.append(x)
                if get_layer:
                    if i in layer_idx:
                        activation_dict[i] = x

            if get_layer:
                return activation_dict
            return torch.cat(output, 1)


if __name__ == '__main__':
    config_path = 'YOLOv3/config/yolov3.cfg'
    weight_path = "YOLOv3/weights/yolov3.weights"
    image_folder = "YOLOv3/data/samples"
    class_path = 'YOLOv3/data/coco.names'
    image_size = 416
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    YOLOv3 = YOLO(config_path, image_size)
    YOLOv3.load_weights(weight_path)
    YOLOv3.to(device)
    place_holder = torch.randn(1, 3, image_size, image_size).to(device)

    # Examples
    # to get output from layer 2, 24, and 100
    output = YOLOv3(place_holder, layer_idx=[2, 24, 100])
    # to get gradient from layer 24 back to layer 5
    gradient = YOLOv3(place_holder, output_index=24, input_index=5)
