#!/usr/bin/env python3
"""
Test script to verify checkpoint loading functionality
"""

import os
import torch
from arc_neural_network import ARCNeuralNetwork, check_existing_checkpoint, load_best_model

def test_checkpoint_loading():
    """Test the checkpoint loading functionality"""
    print("Testing checkpoint loading functionality...")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ARCNeuralNetwork().to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Check for existing checkpoint
    checkpoint_path = '/kaggle/input/suyambhoo/pytorch/arc-agi-2/1/best_model.chkpt'
    start_epoch = check_existing_checkpoint(checkpoint_path)
    
    if start_epoch > 0:
        print(f"\nResuming from epoch: {start_epoch}")
        # Load the model
        model, best_loss = load_best_model(model, checkpoint_path, device=device)
        if best_loss is not None:
            print(f"✓ Model successfully loaded with best loss: {best_loss:.4f}")
        else:
            print("✓ Model successfully loaded (no previous loss available)")
        print("✓ Ready for training continuation")
    else:
        print("\nNo checkpoint found - will start training from scratch")
    
    print("\n" + "=" * 50)
    print("Checkpoint loading test completed!")

if __name__ == "__main__":
    test_checkpoint_loading()
