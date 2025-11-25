"""
ConvNeXt-Tiny with ACMix Integration
Based on the paper implementation:
- ConvNeXt-Tiny as backbone
- ACMix modules inserted at specific layers
- 22-layer architecture (optimal depth from paper)
- Network width: 768
- GELU activation
"""

import torch
import torch.nn as nn
from timm.models.convnext import ConvNeXt
from timm.models.layers import trunc_normal_, DropPath
import timm

from acmix import StackedACMix, ACMixBlock


class ConvNeXtACMix(nn.Module):
    """
    ConvNeXt-Tiny with ACMix modules
    
    Based on paper specifications:
    - Adds Stacked ACMix to 2nd Block (low-dimensional features)
    - Adds Stacked ACMix to last Block (high-dimensional features)
    - Total 22 layers
    """
    
    def __init__(
        self,
        num_classes=100,
        num_acmix_blocks=2,
        drop_path_rate=0.1,
        pretrained=True,
        img_size=224
    ):
        super(ConvNeXtACMix, self).__init__()
        
        # Load pretrained ConvNeXt-Tiny as backbone
        if pretrained:
            self.backbone = timm.create_model(
                'convnext_tiny',
                pretrained=True,
                num_classes=0,  # Remove classification head
                drop_path_rate=drop_path_rate
            )
        else:
            self.backbone = timm.create_model(
                'convnext_tiny',
                pretrained=False,
                num_classes=0,
                drop_path_rate=drop_path_rate
            )
        
        # ConvNeXt-Tiny feature dimensions: [96, 192, 384, 768]
        self.feature_dims = [96, 192, 384, 768]
        
        # Insert ACMix at 2nd stage (low-dimensional features)
        # Stage index 1 has 192 channels
        self.acmix_low = StackedACMix(
            in_channels=self.feature_dims[1],
            out_channels=self.feature_dims[1],
            num_blocks=num_acmix_blocks,
            num_heads=8,
            drop_path=drop_path_rate
        )
        
        # Insert ACMix at last stage (high-dimensional features)
        # Stage index 3 has 768 channels
        self.acmix_high = StackedACMix(
            in_channels=self.feature_dims[3],
            out_channels=self.feature_dims[3],
            num_blocks=num_acmix_blocks,
            num_heads=8,
            drop_path=drop_path_rate
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Stacked FFN Head (as mentioned in paper)
        self.head = StackedFFNHead(
            in_features=self.feature_dims[3],
            hidden_features=self.feature_dims[3] * 2,
            out_features=num_classes,
            num_layers=2,
            drop=0.1
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        """Extract features through the network"""
        
        # Stage 0: Stem
        x = self.backbone.stem(x)
        
        # Stage 1: First stage
        x = self.backbone.stages[0](x)
        
        # Stage 2: Second stage + ACMix (low-dimensional)
        x = self.backbone.stages[1](x)
        x = self.acmix_low(x)
        
        # Stage 3: Third stage
        x = self.backbone.stages[2](x)
        
        # Stage 4: Fourth stage + ACMix (high-dimensional)
        x = self.backbone.stages[3](x)
        x = self.acmix_high(x)
        
        return x
    
    def forward(self, x):
        """Forward pass"""
        # Extract features
        x = self.forward_features(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = x.flatten(1)
        
        # Classification head
        x = self.head(x)
        
        return x


class StackedFFNHead(nn.Module):
    """
    Stacked Feed-Forward Network Head
    As mentioned in the paper for better classification
    """
    
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        num_layers=2,
        act_layer=nn.GELU,
        drop=0.0
    ):
        super(StackedFFNHead, self).__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        layers = []
        
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_features
            out_dim = hidden_features if i < num_layers - 1 else out_features
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(act_layer())
                layers.append(nn.Dropout(drop))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


def create_model(config):
    """
    Create model based on configuration
    """
    model = ConvNeXtACMix(
        num_classes=config['data']['num_classes'],
        num_acmix_blocks=config['model']['num_acmix_blocks'],
        drop_path_rate=config['model']['drop_path_rate'],
        pretrained=config['model']['pretrained'],
        img_size=config['data']['image_size']
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing ConvNeXt-Tiny with ACMix...")
    
    # Create a sample config
    config = {
        'data': {
            'num_classes': 100,
            'image_size': 224
        },
        'model': {
            'num_acmix_blocks': 2,
            'drop_path_rate': 0.1,
            'pretrained': False  # Set to False for testing
        }
    }
    
    model = create_model(config)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ Model test passed!")
