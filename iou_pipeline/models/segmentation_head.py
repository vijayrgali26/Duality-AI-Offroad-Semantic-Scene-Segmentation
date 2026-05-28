"""
Segmentation Head Architectures

Provides ConvNeXt-style decoder heads for semantic segmentation on top of
DINOv2 backbone features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SegmentationHeadConvNeXt(nn.Module):
    """
    ConvNeXt-style segmentation head for semantic segmentation.
    
    This decoder takes patch tokens from DINOv2 backbone and produces
    per-pixel class predictions using ConvNeXt-inspired architecture with
    depthwise separable convolutions.
    
    Architecture:
    1. Stem: 7x7 conv to project embeddings to intermediate channels
    2. ConvNeXt Block: Depthwise 7x7 conv + pointwise 1x1 conv with GELU
    3. Classifier: 1x1 conv to produce class logits
    
    Args:
        in_channels: Input embedding dimension from backbone
        out_channels: Number of output classes
        token_h: Number of patches in height dimension
        token_w: Number of patches in width dimension
        hidden_channels: Number of intermediate channels (default: 128)
        
    Example:
        >>> head = SegmentationHeadConvNeXt(384, 11, 34, 19)
        >>> tokens = torch.randn(2, 646, 384)  # (B, N, D)
        >>> logits = head(tokens)
        >>> print(logits.shape)
        torch.Size([2, 11, 34, 19])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_h: int,
        token_w: int,
        hidden_channels: int = 128
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Stem: Project embeddings to hidden dimension
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GELU()
        )
        
        # ConvNeXt-style block
        # Depthwise conv (groups=channels) followed by pointwise conv
        self.block = nn.Sequential(
            # Depthwise 7x7 convolution
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                groups=hidden_channels  # Depthwise
            ),
            nn.GELU(),
            # Pointwise 1x1 convolution
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        
        # Classifier head
        self.classifier = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through segmentation head.
        
        Args:
            x: Patch tokens from backbone of shape (B, N, D) where:
               - B is batch size
               - N is number of patches (H * W)
               - D is embedding dimension
               
        Returns:
            Class logits of shape (B, C, H, W) where:
            - B is batch size
            - C is number of classes
            - H is number of patches in height
            - W is number of patches in width
        """
        B, N, C = x.shape
        
        # Reshape from (B, N, C) to (B, H, W, C)
        x = x.reshape(B, self.H, self.W, C)
        
        # Permute to (B, C, H, W) for convolutions
        x = x.permute(0, 3, 1, 2)
        
        # Apply stem
        x = self.stem(x)
        
        # Apply ConvNeXt block
        x = self.block(x)
        
        # Apply classifier
        logits = self.classifier(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SegmentationHeadConvNeXtDeep(nn.Module):
    """
    Deeper ConvNeXt-style segmentation head with multiple blocks.
    
    This variant uses multiple ConvNeXt blocks for increased capacity,
    useful for more complex segmentation tasks.
    
    Args:
        in_channels: Input embedding dimension from backbone
        out_channels: Number of output classes
        token_h: Number of patches in height dimension
        token_w: Number of patches in width dimension
        hidden_channels: Number of intermediate channels (default: 128)
        num_blocks: Number of ConvNeXt blocks (default: 2)
        
    Example:
        >>> head = SegmentationHeadConvNeXtDeep(384, 11, 34, 19, num_blocks=3)
        >>> tokens = torch.randn(2, 646, 384)
        >>> logits = head(tokens)
        >>> print(logits.shape)
        torch.Size([2, 11, 34, 19])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_h: int,
        token_w: int,
        hidden_channels: int = 128,
        num_blocks: int = 2
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GELU()
        )
        
        # Multiple ConvNeXt blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=7,
                    padding=3,
                    groups=hidden_channels
                ),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                nn.GELU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        
        # Classifier
        self.classifier = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through deep segmentation head."""
        B, N, C = x.shape
        
        # Reshape and permute
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        
        # Apply stem
        x = self.stem(x)
        
        # Apply multiple blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply classifier
        logits = self.classifier(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SegmentationHeadWithDeepSupervision(nn.Module):
    """
    ConvNeXt-style segmentation head with deep supervision support.
    
    This head can output auxiliary predictions from intermediate layers
    for deep supervision training, which helps gradient flow and improves
    training stability.
    
    Args:
        in_channels: Input embedding dimension from backbone
        out_channels: Number of output classes
        token_h: Number of patches in height dimension
        token_w: Number of patches in width dimension
        hidden_channels: Number of intermediate channels (default: 128)
        
    Example:
        >>> head = SegmentationHeadWithDeepSupervision(384, 11, 34, 19)
        >>> tokens = torch.randn(2, 646, 384)
        >>> main_out, aux_out = head(tokens, return_aux=True)
        >>> print(main_out.shape, aux_out.shape)
        torch.Size([2, 11, 34, 19]) torch.Size([2, 11, 34, 19])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_h: int,
        token_w: int,
        hidden_channels: int = 128
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GELU()
        )
        
        # Auxiliary classifier (after stem)
        self.aux_classifier = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
        # Main ConvNeXt block
        self.block = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                groups=hidden_channels
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        
        # Main classifier
        self.classifier = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional auxiliary output.
        
        Args:
            x: Patch tokens from backbone of shape (B, N, D)
            return_aux: If True, return auxiliary predictions
            
        Returns:
            If return_aux is False: main_logits of shape (B, C, H, W)
            If return_aux is True: (main_logits, aux_logits) both of shape (B, C, H, W)
        """
        B, N, C = x.shape
        
        # Reshape and permute
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        
        # Apply stem
        x = self.stem(x)
        
        # Get auxiliary output if requested
        if return_aux:
            aux_out = self.aux_classifier(x)
        
        # Apply main block
        x = self.block(x)
        
        # Get main output
        main_out = self.classifier(x)
        
        if return_aux:
            return main_out, aux_out
        return main_out
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_segmentation_head(
    head_type: str,
    in_channels: int,
    out_channels: int,
    token_h: int,
    token_w: int,
    hidden_channels: int = 128,
    **kwargs
) -> nn.Module:
    """
    Factory function to create segmentation heads.
    
    Args:
        head_type: Type of head ('convnext', 'convnext_deep', 'deep_supervision')
        in_channels: Input embedding dimension
        out_channels: Number of output classes
        token_h: Number of patches in height
        token_w: Number of patches in width
        hidden_channels: Number of intermediate channels
        **kwargs: Additional arguments for specific head types
                 - num_blocks: For 'convnext_deep'
        
    Returns:
        Segmentation head module
        
    Raises:
        ValueError: If head_type is not supported
        
    Example:
        >>> head = create_segmentation_head('convnext', 384, 11, 34, 19)
        >>> print(type(head).__name__)
        SegmentationHeadConvNeXt
    """
    if head_type == 'convnext':
        return SegmentationHeadConvNeXt(
            in_channels, out_channels, token_h, token_w, hidden_channels
        )
    elif head_type == 'convnext_deep':
        num_blocks = kwargs.get('num_blocks', 2)
        return SegmentationHeadConvNeXtDeep(
            in_channels, out_channels, token_h, token_w, hidden_channels, num_blocks
        )
    elif head_type == 'deep_supervision':
        return SegmentationHeadWithDeepSupervision(
            in_channels, out_channels, token_h, token_w, hidden_channels
        )
    else:
        raise ValueError(
            f"Unsupported head_type '{head_type}'. "
            f"Supported types: 'convnext', 'convnext_deep', 'deep_supervision'"
        )


if __name__ == "__main__":
    # Demo usage
    print("Segmentation Head Demo\n")
    
    # Create a standard ConvNeXt head
    print("Creating SegmentationHeadConvNeXt...")
    head = SegmentationHeadConvNeXt(
        in_channels=384,  # DINOv2-ViT-Small
        out_channels=11,  # 11 terrain classes
        token_h=34,       # 476 / 14
        token_w=19        # 266 / 14
    )
    print(f"  Parameters: {head.get_num_params():,}")
    
    # Test forward pass
    batch_size = 2
    num_patches = 34 * 19
    embed_dim = 384
    tokens = torch.randn(batch_size, num_patches, embed_dim)
    
    print(f"\nInput shape: {tokens.shape}")
    logits = head(tokens)
    print(f"Output shape: {logits.shape}")
    
    # Test deep supervision head
    print("\n\nCreating SegmentationHeadWithDeepSupervision...")
    head_ds = SegmentationHeadWithDeepSupervision(384, 11, 34, 19)
    print(f"  Parameters: {head_ds.get_num_params():,}")
    
    main_out, aux_out = head_ds(tokens, return_aux=True)
    print(f"\nMain output shape: {main_out.shape}")
    print(f"Auxiliary output shape: {aux_out.shape}")
    
    # Test factory function
    print("\n\nUsing factory function...")
    head_factory = create_segmentation_head('convnext_deep', 384, 11, 34, 19, num_blocks=3)
    print(f"Created: {type(head_factory).__name__}")
    print(f"  Parameters: {head_factory.get_num_params():,}")
