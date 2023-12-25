import torch
from torch import nn
from torchvision import models
from transformers import AutoModelForSequenceClassification

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, hidden_size, text_pretrained, img_pretrained, img_weights):
        super(MultimodalModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Init pretrained image model
        self.image_feature_extractor = models.get_model(name=img_pretrained, weights=img_weights)

        # Freeze params
        for param in self.image_feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze params of last layer
        for param in self.image_feature_extractor.fc.parameters():
            param.requires_grad = True

        self.fc1 = nn.Linear(1000, hidden_size)

        # Init pretrained text model
        self.text_feature_extractor = AutoModelForSequenceClassification.from_pretrained(text_pretrained, num_labels=hidden_size)

        # Freeze params
        for param in self.text_feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze params of last layer
        for param in self.text_feature_extractor.classifier.parameters():
            param.requires_grad = True

        self.fc2 = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, img_tensor, text_input):
        # Process image
        img_feat = self.image_feature_extractor(img_tensor)
        img_feat = self.fc1(img_feat)

        # Process text
        text_feat = self.text_feature_extractor(**text_input).logits

        # Concat and pass through fully-connected layer
        out = self.fc2(torch.concat([img_feat, text_feat], dim=1))
        return out