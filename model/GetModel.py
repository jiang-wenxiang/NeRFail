from torch import nn, optim
from torchvision.models.inception import Inception3
from torchvision.models.vgg import vgg16
from torchvision.models.alexnet import alexnet
from torchvision.transforms import Resize
from torchvision.models import vit_b_16, resnet50, densenet121, efficientnet_b0, swin_b, Swin_B_Weights
from torchvision import models
from model.IncResv2 import inceptionresnetv2
from model.MyModel import MyCNN
import timm


def getModel(model_name, num_classes):

    if model_name == "inception":
        # inception v3 model
        model = Inception3(num_classes=num_classes)
    elif model_name == "vgg16":
        model = vgg16(num_classes=num_classes)
    elif model_name == "alexnet":
        model = alexnet(num_classes=num_classes)
    elif model_name == "vit_b_16":
        model = vit_b_16(num_classes=num_classes)
    elif model_name == "resnet50":
        model = resnet50(num_classes=num_classes)
    elif model_name == "densenet121":
        model = densenet121(num_classes=num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(num_classes=num_classes)
    elif model_name == "efficientnet":
        model = efficientnet_b0(num_classes=num_classes)
    elif model_name == "swin_b":
        model = swin_b(num_classes=num_classes)
    elif model_name == "incresv2":
        model = inceptionresnetv2(num_classes=num_classes, pretrained=None)
    elif model_name == "mixer_b":
        model = timm.create_model(model_name="mixer_b16_224", num_classes=num_classes)

    else:
        if model_name != "my_model":
            print("Model Name is Wrong!!!")
        model = MyCNN(num_classes=num_classes)

    if model_name == "my_model":
        torch_resize = None
    if model_name == "vit_b_16" or model_name == "mixer_b":
        torch_resize = Resize([224, 224])
    else:
        torch_resize = Resize([299, 299])

    return model, torch_resize