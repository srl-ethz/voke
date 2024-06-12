"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

# import torch
import torch.nn as nn
import math
# from torchvision import models
from models import efficientnet, resnet, vgg

class ImagePredictionModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg["model"]["in_channels"]
        self.num_classes = cfg["model"]["num_classes"]
        # self.n_nodes = cfg["model"]["n_nodes"]
        self.dropout = cfg["model"]["dropout"]
        self.image_size = cfg["data"]["size"]
        self.model = self.get_model(cfg["model"]["type_model"])

    def forward(self, x):
        x = self.model(x)
        return x

    def classifier_head(self, output_dim):
        return nn.Sequential(nn.Flatten(),
                            #  nn.Linear(output_dim, self.n_nodes),
                            #  nn.ReLU(),
                             nn.Dropout(self.dropout),
                             nn.Linear(output_dim, self.num_classes))

    def get_model(self, model_type):
        # VGG
        if model_type == 'vgg_s_bn':
            base_model = vgg.Net_bn_short(self.dropout, self.num_classes)
        
        elif model_type == 'vgg11':
            base_model = vgg.vgg11(dropout=self.dropout, 
                                   in_channels=self.in_channels, 
                                   num_classes=self.num_classes)
            
        elif model_type == 'vgg11_bn':
            base_model = vgg.vgg11_bn(dropout=self.dropout,
                                      in_channels=self.in_channels, 
                                      num_classes=self.num_classes)
            
        elif model_type == 'vgg13':
            base_model = vgg.vgg13(dropout=self.dropout, 
                                   in_channels=self.in_channels, 
                                   num_classes=self.num_classes)
            
        elif model_type == 'vgg13_bn':
            base_model = vgg.vgg13_bn(dropout=self.dropout,
                                      in_channels=self.in_channels, 
                                      num_classes=self.num_classes)

        # ResNet
        elif model_type == 'resnet18':
            base_model = resnet.resnet18(in_channels=self.in_channels, 
                                         num_classes=self.num_classes)
            base_model.fc = self.classifier_head(base_model.fc.in_features)

        elif model_type == 'resnet34':
            base_model = resnet.resnet34(in_channels=self.in_channels, 
                                         num_classes=self.num_classes)
            base_model.fc = self.classifier_head(base_model.fc.in_features)

        elif model_type == 'resnet50':
            base_model = resnet.resnet50(in_channels=self.in_channels, 
                                         num_classes=self.num_classes)
            base_model.fc = self.classifier_head(base_model.fc.in_features)

        # EfficientNet
        elif model_type == 'efficientnet_b0':
            base_model = efficientnet.efficientnet_b0(in_channels=self.in_channels, num_classes=self.num_classes)
            # base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            # if self.freeze_layers:
            #     print("Freeze layers of pretrained model")
            #     self.freeze_layers_base_model(base_model)

            # base_model.classifier = self.classifier_head(1280)
        elif model_type == 'efficientnet_b1':
            base_model = efficientnet.efficientnet_b1(dropout=self.dropout, 
                                                      in_channels=self.in_channels, 
                                                      num_classes=self.num_classes)

        elif model_type == 'efficientnet_b2':
            base_model = efficientnet.efficientnet_b2(dropout=self.dropout, 
                                                      in_channels=self.in_channels, 
                                                      num_classes=self.num_classes)
        elif model_type == 'efficientnet_b3':
            base_model = efficientnet.efficientnet_b3(dropout=self.dropout, 
                                                      in_channels=self.in_channels, 
                                                      num_classes=self.num_classes)
        elif model_type == 'efficientnet_b4':
            base_model = efficientnet.efficientnet_b4(dropout=self.dropout, 
                                                      in_channels=self.in_channels, 
                                                      num_classes=self.num_classes)
        elif model_type == 'efficientnet_b5':
            base_model = efficientnet.efficientnet_b5(dropout=self.dropout, 
                                                      in_channels=self.in_channels, 
                                                      num_classes=self.num_classes)
        elif model_type == 'efficientnet_b6':
            base_model = efficientnet.efficientnet_b6(dropout=self.dropout, 
                                                      in_channels=self.in_channels, 
                                                      num_classes=self.num_classes)
        elif model_type == 'efficientnet_b7':
            base_model = efficientnet.efficientnet_b7(dropout=self.dropout, 
                                                      in_channels=self.in_channels, 
                                                      num_classes=self.num_classes)
        
        # EfficientNetV2
        elif model_type == 'efficientnet_v2_s':
            base_model = efficientnet.efficientnet_v2_s(dropout=self.dropout, 
                                                        in_channels=self.in_channels, 
                                                        num_classes=self.num_classes)
        elif model_type == 'efficientnet_v2_m':
            base_model = efficientnet.efficientnet_v2_m(dropout=self.dropout, 
                                                        in_channels=self.in_channels, 
                                                        num_classes=self.num_classes)
        elif model_type == 'efficientnet_v2_l':
            base_model = efficientnet.efficientnet_v2_l(dropout=self.dropout, 
                                                        in_channels=self.in_channels, 
                                                        num_classes=self.num_classes)
        # elif model_type == 'cct_14_7x2_224':
        #     base_model = cct_14_7x2_224(pretrained=self.pretrained, progress=self.pretrained, num_classes=self.num_classes, img_size=self.image_size)
        #     if self.freeze_layers:
        #         print("Freeze layers of pretrained model")
        #         self.freeze_layers_base_model(base_model)

        # elif model_type == 'cct_14_7x2_384':
        #     base_model = cct_14_7x2_384(pretrained=self.pretrained, progress=self.pretrained, num_classes=self.num_classes, img_size=self.image_size)
        #     if self.freeze_layers:
        #         print("Freeze layers of pretrained model")
        #         self.freeze_layers_base_model(base_model)

        # elif model_type == 'cct_7_7x2_224_sine':
        #     base_model = cct_7_7x2_224_sine(pretrained=self.pretrained, progress=self.pretrained, num_classes=self.num_classes, img_size=self.image_size)
        #     if self.freeze_layers:
        #         print("Freeze layers of pretrained model")
        #         self.freeze_layers_base_model(base_model)

        return base_model