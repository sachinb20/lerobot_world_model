"""
Inference script for the Image World Model.
Loads a trained model and generates predictions on random samples.
"""
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from lerobot_world_model.dataloader import get_train_val_dataloaders, compute_delta_action
from lerobot_world_model.model import ImageWorldModel


def load_checkpoint(checkpoint_path):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"\nLoaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    return checkpoint


def visualize_predictions(
    checkpoint_path="checkpoints/best_checkpoint.pth",
    num_samples=5,
    batch_size=16,
    val_split=0.1,
    save_path="inference_results.png",
):
    """
    Load model and visualize predictions on random samples.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of random samples to visualize from each split
        batch_size: Batch size for dataloaders
        val_split: Validation split ratio
        save_path: Path to save visualization
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("IMAGE WORLD MODEL INFERENCE")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples per split: {num_samples}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Extract config
    image_size = checkpoint['image_size']
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    latent_dim = checkpoint['latent_dim']
    hidden_dim = checkpoint['hidden_dim']
    in_channels = checkpoint['in_channels']
    frame_skip = checkpoint['frame_skip']
    use_delta_action = checkpoint['use_delta_action']
    config = checkpoint['config']
    normalization_stats = checkpoint['normalization_stats']
    
    # Initialize model
    model = ImageWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        in_channels=in_channels,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        image_size=image_size
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load normalization stats
    state_mean = normalization_stats['state_mean'].to(DEVICE)
    state_std = normalization_stats['state_std'].to(DEVICE)
    action_mean = normalization_stats['action_mean'].to(DEVICE)
    action_std = normalization_stats['action_std'].to(DEVICE)
    
    # Get dataloaders
    print("\nLoading data...")
    train_loader, val_loader, image_key, _ = get_train_val_dataloaders(
        batch_size=batch_size,
        image_key=config['image_key'],
        frame_skip=frame_skip,
        use_delta_action=use_delta_action,
        val_split=val_split
    )
    
    # Collect random samples from train and val
    def get_random_samples(dataloader, num_samples):
        """Get random samples from dataloader."""
        all_batches = list(dataloader)
        samples = []
        
        for _ in range(num_samples):
            batch_idx = np.random.randint(0, len(all_batches))
            batch = all_batches[batch_idx]
            sample_idx = np.random.randint(0, len(batch[image_key]))
            
            sample = {
                'image': batch[image_key][sample_idx],
                'state': batch['observation.state'][sample_idx],
            }
            
            if use_delta_action:
                delta_action = compute_delta_action(batch, config)[sample_idx]
                sample['action'] = delta_action
            else:
                sample['action'] = batch['action'][sample_idx]
            
            samples.append(sample)
        
        return samples
    
    print(f"Collecting {num_samples} random samples from train set...")
    train_samples = get_random_samples(train_loader, num_samples)
    
    print(f"Collecting {num_samples} random samples from val set...")
    val_samples = get_random_samples(val_loader, num_samples)
    
    # Generate predictions
    print("\nGenerating predictions...")
    
    def predict_sample(sample):
        """Generate prediction for a single sample."""
        with torch.no_grad():
            # Get current and next images
            curr_image = sample['image'][0].unsqueeze(0).to(DEVICE)  # (1, C, H, W)
            next_image_gt = sample['image'][1].unsqueeze(0).to(DEVICE)
            
            # Normalize images
            if curr_image.max() > 1.0:
                curr_image = curr_image / 255.0
                next_image_gt = next_image_gt / 255.0
            
            # Get state and action
            curr_state = sample['state'][0].unsqueeze(0).to(DEVICE)
            delta_action = sample['action'].unsqueeze(0).to(DEVICE)
            
            # Normalize
            curr_state_norm = (curr_state - state_mean) / state_std
            delta_action_norm = (delta_action - action_mean) / action_std
            
            # Predict
            next_image_pred = model(curr_image, curr_state_norm, delta_action_norm)
            
            return curr_image, next_image_gt, next_image_pred
    
    # Create visualization
    total_samples = num_samples * 2  # train + val
    fig, axes = plt.subplots(total_samples, 3, figsize=(12, 4 * total_samples))
    
    if total_samples == 1:
        axes = axes.reshape(1, -1)
    
    row = 0
    
    # Process train samples
    for i, sample in enumerate(train_samples):
        curr_img, gt_img, pred_img = predict_sample(sample)
        
        # Convert to numpy and denormalize
        curr_img = curr_img.cpu().squeeze().permute(1, 2, 0).numpy()
        gt_img = gt_img.cpu().squeeze().permute(1, 2, 0).numpy()
        pred_img = pred_img.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Clip to valid range
        curr_img = np.clip(curr_img, 0, 1)
        gt_img = np.clip(gt_img, 0, 1)
        pred_img = np.clip(pred_img, 0, 1)
        
        axes[row, 0].imshow(curr_img)
        axes[row, 0].set_title(f"Train {i+1}: Current Frame")
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(gt_img)
        axes[row, 1].set_title("Ground Truth Next")
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(pred_img)
        axes[row, 2].set_title("Predicted Next")
        axes[row, 2].axis('off')
        
        row += 1
    
    # Process val samples
    for i, sample in enumerate(val_samples):
        curr_img, gt_img, pred_img = predict_sample(sample)
        
        curr_img = curr_img.cpu().squeeze().permute(1, 2, 0).numpy()
        gt_img = gt_img.cpu().squeeze().permute(1, 2, 0).numpy()
        pred_img = pred_img.cpu().squeeze().permute(1, 2, 0).numpy()
        
        curr_img = np.clip(curr_img, 0, 1)
        gt_img = np.clip(gt_img, 0, 1)
        pred_img = np.clip(pred_img, 0, 1)
        
        axes[row, 0].imshow(curr_img)
        axes[row, 0].set_title(f"Val {i+1}: Current Frame")
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(gt_img)
        axes[row, 1].set_title("Ground Truth Next")
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(pred_img)
        axes[row, 2].set_title("Predicted Next")
        axes[row, 2].axis('off')
        
        row += 1
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(os.path.dirname(__file__), save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Visualization saved to: {save_path}")
    print(f"{'='*60}\n")
    
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on trained world model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_checkpoint.pth", 
                        help="Path to checkpoint file")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of samples per split (train/val)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for dataloaders")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--save_path", type=str, default="inference_results.png", 
                        help="Path to save visualization")
    args = parser.parse_args()
    
    visualize_predictions(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        val_split=args.val_split,
        save_path=args.save_path,
    )
