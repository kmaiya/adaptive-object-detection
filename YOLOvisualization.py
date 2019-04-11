import torch

from modified_YOLO import YOLO
from visualizer import Visualizer


class YOLOv3Visualizer(Visualizer):
    """
    This is a class that allows for some modifications for YOLOv3
    """

    def __init__(self, model, module_list, cuda=True):
        """
        First let us inherit the visualizer
        :param model: the YOLOv3 model pre-trained
        :param cuda: the device to work on
        """
        super(YOLOv3Visualizer, self).__init__(model, module_list=module_list, cuda=cuda)

    def forward_pass(self, img, idx):
        """
        This function gets the n-th layer output of YOLOv3
        :param img: the input image
        :param idx: the index of the layer
        :return: the output of the specific layer
        """
        return self.model(img, layer_idx=[idx])[idx]

    def _module_list_channel_count(self):
        """
        This overloads the module list method, which modifies the self.channel_count
        """
        place_holder = self.random_init(self.input_size).to(device)
        outputs = self.model(place_holder, layer_idx=list(range(self.layer_num)))
        self.layer_num = len(set(outputs.values()))
        self.channel_count = [output.shape[1] for output in outputs.values()]
        del place_holder
        if self.cuda:
            self.clear_cuda_memory()


if __name__ == "__main__":
    # layer_idx = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    # channel_idx = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    # epochs = int(sys.argv[3]) if len(sys.argv) >= 4 else 30
    # lr = float(sys.argv[4]) if len(sys.argv) >= 5 else 1e-4
    # weight_decay = lr / 100

    layer_idx = 101
    channel_idx = 3
    epochs = 300

    lr = 1e-2
    weight_decay = 1e-5

    scale_step = 30
    init_size = 416

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_size = 416
    config_path = 'YOLOv3/config/yolov3.cfg'
    weight_path = "YOLOv3/weights/yolov3.weights"
    image_folder = "YOLOv3/data/samples"
    class_path = 'YOLOv3/data/coco.names'
    batch_size = 1
    n_cpu = 8

    YOLOv3 = YOLO(config_path, image_size)
    YOLOv3.load_weights(weight_path)
    yolo_module_list = list(YOLOv3.children())[0]

    visualizer = YOLOv3Visualizer(YOLOv3, module_list=yolo_module_list, cuda=True)
    visualizer.multistep_visualize(layer_idx, channel_idx, data_path="multi_vis", learning_rate=lr,
                                   weight_decay=weight_decay, epochs=epochs, scale_step=scale_step,
                                   initial_size=init_size, forward_pass=visualizer.one_pass_neuron)
    visualizer.vanilla_visualize(layer_idx, channel_idx, data_path="vanilla_vis", learning_rate=lr,
                                 weight_decay=weight_decay, epochs=epochs,
                                 single_pass=visualizer.one_pass_neuron)
    ####
    # visualizer.visualize_whole_layer(10, data_path='visualization', weight_decay=1e-5)
