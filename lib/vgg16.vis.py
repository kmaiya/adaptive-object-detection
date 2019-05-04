import torchvision.models as models

from visualizer import Visualizer


class VggVisualizer(Visualizer):
    """
    Vgg16 visualizer
    """

    def forward_pass(self, img, layer_idx=None, output_index=None, input_index=None):
        if layer_idx is not None:
            output_dict = {}
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                img = layer(img)
                # Only need to forward until the selected layer is reached
                if index in layer_idx:
                    # (forward hook function triggered)
                    output_dict[index] = img

            return output_dict
        else:
            return self.model(img)


if __name__ == "__main__":
    vgg16 = models.vgg16(pretrained=True).features
    vgg_visualizer = VggVisualizer(vgg16, list(vgg16.children()),
                                model_intake_size=224)

    vgg_visualizer.vanilla_visualize(17, 5, data_path="vgg", epochs=50)
