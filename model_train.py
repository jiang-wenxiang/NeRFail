import copy
import os
import time
from pathlib import Path

import torch
from torch import nn, optim
from torchvision.models.inception import Inception3
from torchvision.models.vgg import vgg16
from torchvision.models.alexnet import alexnet
from torchvision.transforms import Resize
from torchvision.models import vit_b_16, resnet50, densenet121, efficientnet_b0, swin_b, Swin_B_Weights
from torchvision import models
from model.IncResv2 import inceptionresnetv2
from MyDataset import MySimpleDataset
import configargparse
from model.GetModel import getModel

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

parser = configargparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="inception", help='classifier model name')
parser.add_argument('--num_classes', type=int, default=8, help='the number of classification')
parser.add_argument('--data_dir', type=str, default="./data/nerf_synthetic", help='the classification set path')
args = parser.parse_args()

load_pretrain_weights = False
num_classes = args.num_classes
_3_channels = True
resize_frame = True
load_later = False

model_name = args.model_name
# model_name = "resnet50"
# model_name = "densenet121"
# model_name = "mobilenet"
# model_name = "efficientnet"
# model_name = "swin_b"
# model_name = "incresv2"
# model_name = "mixer_b"

model, torch_resize = getModel(model_name, num_classes)

model.to(device)

model.train()

# config
data_dir = Path(args.data_dir)
batch_size = 16
epochs = 200

save_model_epochs = 50

experiment_name = model_name
output_dir = Path("./output") / Path(experiment_name)
model_dir = Path("./model")

model_weights_save = model_dir / Path("weights")
class_names_file = output_dir / Path("class_names.txt")

# create path
for path in [output_dir, model_weights_save]:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

pre_tab_str = ""
# if load_pretrain_weights:
#     # weights_path = model_weights_save / Path("inception_v3_google-0cc3c7bd.pth")
#     # pretrain_model = torch.load(weights_path, map_location=device)
#     pretrain_model = swin_b(weights=Swin_B_Weights).state_dict()
#     jump_load_layer = ['fc.weight', 'fc.bias', 'AuxLogits.fc.weight', 'AuxLogits.fc.bias',
#                        'classifier.weight', 'classifier.bias', "classifier.1.weight", "classifier.1.bias",
#                        "head.weight", "head.bias"]
#     model_dict = model.state_dict()
#     state_dict = {k:v for k,v in pretrain_model.items() if ((k in model_dict.keys()) and (k not in jump_load_layer))}
#     model_dict.update(state_dict)
#     model.load_state_dict(model_dict)
#     pre_tab_str = "pre_"

# train
since = time.time()
val_acc_history = []

print("Start train "+model_name+"!")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: MySimpleDataset(data_dir, x, class_names_file,
                                     _3_channels=_3_channels, resize_frame=resize_frame,
                                     resize_frame_size=torch_resize, device=device, load_later=load_later) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True) for x in
    ['train', 'val']
}


for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    print('*' * 10)
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # trans 4 channels to 3 channels with a write background
            if _3_channels:
                pass
            else:
                inputs_size = inputs.size()
                inputs_alpha = inputs[:, 3, :, :].unsqueeze(1).broadcast_to([inputs_size[0], 3,
                                                                             inputs_size[2], inputs_size[3]])
                inputs_rgb = inputs[:, :3, :, :]
                tensor_white = torch.ones_like(inputs_rgb) * 255
                inputs = torch.where(inputs_alpha > 0, inputs_rgb, tensor_white)

            if torch_resize is not None:
                # resize image from 800x800 to other
                inputs = torch_resize(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                if phase == 'train':
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    if model_name == "inception":
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val':
            if (epoch + 1) % save_model_epochs == 0:
                # print('Saving model......')
                torch.save(model.state_dict(), model_weights_save / Path(model_name+'_'+pre_tab_str+str(num_classes)+'_%03d.pth' % (epoch + 1)))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if phase == 'val':
            val_acc_history.append(epoch_acc)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
# load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), model_weights_save / Path(model_name+'_'+pre_tab_str+str(num_classes)+'_best.pth'))


