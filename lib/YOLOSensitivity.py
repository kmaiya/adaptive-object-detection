########################################################
# Purpose of This Script:                              #
#   - An instance of the sensitivity measurer          #
#                                                      #
########################################################

import cv2
import numpy as np
import torch

from Sensitivity import SensitivityMeasurer
from modified_YOLO import YOLO


class YOLOMeasurer(SensitivityMeasurer):
    """
    This is a measurer made for YOLO
    """

    def __init__(self, model, module_list, cuda=True):
        """
        First let's inherit the measurer
        :param model: the YOLOv3 model pre-trained
        :param module_list: the list of modules of the YOLO model
        :param cuda: the device to put on
        """
        super(YOLOMeasurer, self).__init__(model, module_list, cuda=cuda)

    def forward_pass(self, img, layer_idx=None, output_index=None, input_index=None):
        """
        This function gets the n-th layer output of YOLOv3
        :param img: the input image
        :param layer_idx: the index of the layer
        :param output_index: the index of the output to take gradient from
        :param input_index: the index of the input layer whose gradient is calculated
        :return: the output of the specific layer
        """
        get_grad = output_index is not None and input_index is not None
        if not get_grad:
            img = img.to(self.device)
            return self.model(img, layer_idx=[layer_idx])[layer_idx]
        else:
            img = img.to(self.device)
            return self.model(img, output_index=output_index, input_index=input_index)

    def _module_list_channel_count(self):
        """
        This overloads the module list method, which modifies the self.channel_count
        """
        place_holder = torch.randn(self.batch_size, self.channel_num,
                                   self.height, self.width).to(self.device)
        outputs = self.model(place_holder, layer_idx=list(range(self.layer_num)))
        self.layer_num = len(set(outputs.values()))
        self.channel_count = [output.shape[1] for output in outputs.values()]
        self.size_count = [output.shape[2:] for output in outputs.values()]
        self.size_count[-1] = torch.Size([0, 0])
        del place_holder
        if self.cuda:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # the paths
    config_path = 'YOLOv3/config/yolov3.cfg'
    weight_path = "YOLOv3/weights/yolov3.weights"
    image_folder = "YOLOv3/data/samples"
    class_path = 'YOLOv3/data/coco.names'

    image_size = 416
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load the images
    img1 = cv2.imread('pair_compare/00073_000000463_.jpg')
    img1 = cv2.resize(img1, tuple([image_size, image_size]))
    img1 = torch.tensor(img1 / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img2 = cv2.imread('pair_compare/00073_000000465_.jpg')
    img2 = cv2.resize(img2, tuple([image_size, image_size]))
    img2 = torch.tensor(img2 / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    YOLOv3 = YOLO(config_path, image_size)
    YOLOv3.load_weights(weight_path)
    yolo_module_list = list(YOLOv3.children())[0]
    place_holder = torch.randn(1, 3, 416, 416).to(device)
    measurer = YOLOMeasurer(YOLOv3, yolo_module_list)
    # measurer.get_nth_neuron(place_holder, 1, 0, 0).shape
    # Jacob = measurer.compute_channel_jacobian(place_holder, 1, 0)
    # jacobian = measurer.compute_channel_jacobian(img1, [0, 0], "reduction_mean")
    gradients1 = measurer.gradient_incr(img1)
    mean_gradients1 = []
    stacked_grad = []
    for i, gradient in enumerate(gradients1):
        if gradient is not None:
            grad_list = [gradient[0, i, ...] for i in range(gradient.shape[1])]
            stacked_grad += grad_list
    mean_gradients1 = [gradient.norm().detach().cpu().numpy() for gradient in stacked_grad]

    gradients2 = measurer.gradient_incr(img2)
    mean_gradients2 = []
    stacked_grad = []
    for i, gradient in enumerate(gradients2):
        if gradient is not None:
            grad_list = [gradient[0, i, ...] for i in range(gradient.shape[1])]
            stacked_grad += grad_list
    mean_gradients2 = [gradient.norm().detach().cpu().numpy() for gradient in stacked_grad]

    np.savetxt("pair_compare/incr_1.csv", np.array(mean_gradients1), delimiter=",")
    np.savetxt("pair_compare/incr_2.csv", np.array(mean_gradients2), delimiter=",")
    """
    uncomment this part if you want to compute the norm
    
    mean_Jacobian1 = measurer.compute_jacobian(img1, mode="reduction_mean")
    for i, derivative in enumerate(mean_Jacobian1):
        if derivative is not None:
            mean_Jacobian1[i] = derivative.norm().detach().cpu().numpy()
    mean_Jacobian1 = np.array(mean_Jacobian1)
    mean_Jacobian2 = measurer.compute_jacobian(img2, mode="reduction_mean")
    for i, derivative in enumerate(mean_Jacobian2):
        if derivative is not None:
            mean_Jacobian2[i] = derivative.norm().detach().cpu().numpy()
    mean_Jacobian2 = np.array(mean_Jacobian2)

    np.savetxt("pair_compare/1.csv", mean_Jacobian1, delimiter=",")
    np.savetxt("pair_compare/2.csv", mean_Jacobian2, delimiter=",")
    """
