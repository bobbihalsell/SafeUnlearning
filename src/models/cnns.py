import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 filters=[64, 64, 128, 128, 256, 256], 
                 num_classes=10, 
                 dropout_rate=0.0,
                 use_batch_norm=True, 
                 downsample_every=3
                 ):
        """
        A CNN with configurable filter counts for each layer.
        
        Args:
            filters: List of filter counts for each convolutional layer
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(CNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Create convolutional layers based on filters list
        for i, out_channels in enumerate(filters):
            # For every third layer, use stride=2 for downsampling
            stride = 2 if i > 0 and i % downsample_every == 0 else 1
            
            # Create convolutional layer
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            self.layers.append(conv)
            
            # Add batch normalization
            if use_batch_norm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            
            # Add dropout after each downsampling layer
            if stride == 2:
                self.layers.append(nn.Dropout(dropout_rate))
            
            # Update input channels for next layer
            in_channels = out_channels
        
        # Output layer - classifier
        self.classifier = nn.Linear(filters[-1], num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                if isinstance(layer, nn.Conv2d):
                    x = layer(x)
                else:  # BatchNorm
                    x = F.relu(layer(x))
            else:  # Dropout
                x = layer(x)
                
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.classifier(x)
        return x


class AllCNN(nn.Module):
    def __init__(self, 
                 filters=[32, 32, 64, 64, 128], 
                 num_classes=10, 
                 dropout_rates=[0.1, 0.1, 0.1],
                 use_batchnorm=True,
                 downsample_every=2
                 ):
        """
        A flexible All-CNN model with configurable filter counts.
        
        Args:
            filters: List of filter counts for each convolutional layer
            num_classes: Number of output classes
            dropout_rates: List of dropout rates to apply after each block
        """
        super(AllCNN, self).__init__()
        
        self.layers = nn.ModuleList()
        in_channels = 3  # Starting with RGB input
        dropout_idx = 0
        
        # Create convolutional layers based on filters list
        for i, out_channels in enumerate(filters):
            # For every second layer, use stride=2 for downsampling
            stride = 2 if i > 0 and i % downsample_every == 0 else 1
            
            # Create convolutional layer
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            self.layers.append(conv)

            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            
            # Add dropout after each downsampling layer
            if stride == 2 and dropout_idx < len(dropout_rates):
                self.layers.append(nn.Dropout(dropout_rates[dropout_idx]))
                dropout_idx += 1
            
            # Update input channels for next layer
            in_channels = out_channels
        
        # Add final dropout before classification
        if dropout_idx < len(dropout_rates):
            self.layers.append(nn.Dropout(dropout_rates[dropout_idx]))
            
        # Output layer - classifier
        self.classifier = nn.Linear(filters[-1], num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Process all layers
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm2d):
                x = F.relu(layer(x))
            else:  # Dropout
                x = layer(x)
                
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.classifier(x)
        return x
        