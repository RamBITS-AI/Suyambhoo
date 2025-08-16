# Suyambhoo
This is my version of the solution for ARC-AGI

## Features

### Checkpoint Loading and Training Continuation
The neural network can now automatically load the best model from a previous training session and continue training from where it left off.

- **Automatic Checkpoint Detection**: The system checks for existing checkpoints at `/kaggle/input/suyambhoo/pytorch/arc-agi-2/1/best_model.chkpt`
- **Resume Training**: If a checkpoint exists, training continues from the next epoch with the saved model state
- **Clean Architecture**: The `train_model` function works with any pre-loaded model state without hardcoded paths
- **Error Handling**: Graceful fallback to fresh training if checkpoint loading fails
- **Progress Tracking**: Clear feedback about checkpoint status and training continuation

## Usage

### Running the Main Training Script
```bash
python arc_neural_network.py
```

The script will automatically:
1. Check for existing checkpoints at the specified path
2. Load the model and optimizer state if found
3. Continue training from the appropriate epoch
4. Save new checkpoints during training

### Testing Checkpoint Loading
```bash
python test_checkpoint_loading.py
```

This test script verifies that the checkpoint loading functionality works correctly.

## Model Architecture

The ARC Neural Network features:
- **Encoder-Decoder Architecture**: Convolutional layers for processing 30x30 grids
- **Attention Mechanism**: Multi-head attention for pattern recognition
- **One-Hot Encoding**: 10-channel representation for grid values 0-9
- **Residual Connections**: Improved gradient flow and training stability

## Training Configuration

- **Epochs**: 1000 (configurable via `NUM_OF_EPOCHS`)
- **Batch Size**: 4
- **Learning Rate**: 0.001 with step decay
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam

## Output Files

- `best_model.chkpt`: Best model checkpoint during training
- `training_loss.png`: Training loss visualization
- `arc_predictions.pt`: Generated predictions for test cases
- `arc_results_{problem_id}.png`: Visualization of results for each problem
