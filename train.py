import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os

from lerobot_world_model.dataloader import get_dataloader, get_train_val_dataloaders, compute_delta_action
from lerobot_world_model.model import ImageWorldModel


def suppress_warnings():
    """Suppress common warnings during training for cleaner output."""
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["JAX_PLATFORMS"] = ""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass
    
    print("Warnings suppressed for cleaner training output.")


def train(
    batch_size=16,
    lr=1e-4,
    epochs=10,
    latent_dim=256,
    hidden_dim=512,
    image_key="observation.images.side",
    frame_skip=5,
    use_delta_action=True,
    val_split=0.1,
):
    """
    Train the Image World Model.
    
    Args:
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of training epochs
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        image_key: Key for image observations
        frame_skip: Number of frames to skip between current and next
        use_delta_action: If True, use delta (relative) actions
        val_split: Fraction of episodes for validation (default 0.1 = 10%)
    """
    suppress_warnings()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print("IMAGE WORLD MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Frame skip: {frame_skip}")
    print(f"Use delta action: {use_delta_action}")
    print(f"Image key: {image_key}")
    print(f"Val split: {val_split}")
    print(f"{'='*60}\n")
    
    # Setup Train/Val Dataloaders
    train_loader, val_loader, image_key, config = get_train_val_dataloaders(
        batch_size=batch_size,
        image_key=image_key,
        frame_skip=frame_skip,
        use_delta_action=use_delta_action,
        val_split=val_split
    )
    
    # Get a sample batch to infer dimensions
    sample_batch = next(iter(train_loader))
    print(f"\nBatch keys: {sample_batch.keys()}")
    
    # Get image shape
    image_sample = sample_batch[image_key]
    print(f"Image shape: {image_sample.shape}")  # (B, T, C, H, W)
    
    # Parse dimensions
    if image_sample.ndim == 5:
        B, T, C, H, W = image_sample.shape
        image_size = (H, W)
    else:
        raise ValueError(f"Expected 5D image tensor (B, T, C, H, W), got {image_sample.ndim}D")
    
    # Get state dimension from observation.state
    state_sample = sample_batch["observation.state"]
    if state_sample.ndim == 3:
        state_dim = state_sample.shape[-1]
    else:
        state_dim = state_sample.shape[-1]
    print(f"State shape: {state_sample.shape}")
    
    # Get action dimension from delta action
    if use_delta_action:
        delta_action_sample = compute_delta_action(sample_batch, config)
        action_dim = delta_action_sample.shape[-1]
        print(f"Delta action shape: {delta_action_sample.shape}")
    else:
        action_sample = sample_batch["action"]
        if action_sample.ndim == 3:
            action_dim = action_sample.shape[-1]
        else:
            action_dim = action_sample.shape[-1]
    
    print(f"\nModel Configuration:")
    print(f"  Image size: {image_size}")
    print(f"  Channels: {C}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    
    # Init Model
    model = ImageWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        in_channels=C,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        image_size=image_size
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # ==========================================================================
    # Load normalization statistics (pre-computed or compute on-the-fly)
    # ==========================================================================
    stats_path = os.path.join(os.path.dirname(__file__), "normalization_stats.pt")
    
    if os.path.exists(stats_path):
        print(f"\n{'='*60}")
        print("Loading pre-computed normalization statistics...")
        print(f"{'='*60}")
        
        from lerobot_world_model.compute_stats import load_stats
        normalization_stats = load_stats(stats_path)
        
        state_mean = normalization_stats['state_mean'].to(DEVICE)
        state_std = normalization_stats['state_std'].to(DEVICE)
        action_mean = normalization_stats['action_mean'].to(DEVICE)
        action_std = normalization_stats['action_std'].to(DEVICE)
        
        print(f"  State mean: {state_mean}")
        print(f"  State std: {state_std}")
        print(f"  Action mean: {action_mean}")
        print(f"  Action std: {action_std}")
    else:
        print(f"\n{'='*60}")
        print("Stats file not found. Sampling to compute stats...")
        print(f"(Run `python -m lerobot_world_model.compute_stats` for full dataset stats)")
        print(f"{'='*60}")
        
        all_states = []
        all_delta_actions = []
        
        max_samples = 100
        for i, batch in enumerate(train_loader):
            if i >= max_samples:
                break
            states = batch["observation.state"]
            curr_state = states[:, 0, :] if states.ndim == 3 else states
            all_states.append(curr_state)
            
            if use_delta_action:
                delta_action = compute_delta_action(batch, config)
            else:
                actions = batch["action"]
                delta_action = actions[:, 0, :] if actions.ndim == 3 else actions
            all_delta_actions.append(delta_action)
        
        print(f"  Sampled {min(max_samples, len(train_loader))} batches")
        
        all_states = torch.cat(all_states, dim=0)
        all_delta_actions = torch.cat(all_delta_actions, dim=0)
        
        state_mean = all_states.mean(dim=0).to(DEVICE)
        state_std = (all_states.std(dim=0) + 1e-8).to(DEVICE)
        action_mean = all_delta_actions.mean(dim=0).to(DEVICE)
        action_std = (all_delta_actions.std(dim=0) + 1e-8).to(DEVICE)
        
        normalization_stats = {
            'state_mean': state_mean.cpu(),
            'state_std': state_std.cpu(),
            'action_mean': action_mean.cpu(),
            'action_std': action_std.cpu(),
            'image_scale': 255.0,
        }
        
        print(f"  State mean: {state_mean}")
        print(f"  State std: {state_std}")
        print(f"  Action mean: {action_mean}")
        print(f"  Action std: {action_std}")
    
    print("Normalization statistics ready!")
    
    # ==========================================================================
    # Training loop
    # ==========================================================================
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    
    print(f"\n{'='*60}")
    print(f"Starting Training with Train/Val Split...")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # ======================================================================
        # Training Phase
        # ======================================================================
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_loader):
            # Get images: (B, T, C, H, W) where T=2 (current, next)
            images = batch[image_key].to(DEVICE)
            
            # Extract current and next images
            curr_image = images[:, 0, :, :, :]  # (B, C, H, W)
            next_image_target = images[:, 1, :, :, :]  # (B, C, H, W)
            
            # Normalize images to [0, 1]
            if curr_image.max() > 1.0:
                curr_image = curr_image / 255.0
                next_image_target = next_image_target / 255.0
            
            # Get current joint state (B, T, state_dim) -> (B, state_dim)
            states = batch["observation.state"].to(DEVICE)
            curr_state = states[:, 0, :] if states.ndim == 3 else states
            
            # Normalize state
            curr_state_normalized = (curr_state - state_mean) / state_std
            
            # Get delta action
            if use_delta_action:
                delta_action = compute_delta_action(batch, config).to(DEVICE)
            else:
                actions = batch["action"].to(DEVICE)
                delta_action = actions[:, 0, :] if actions.ndim == 3 else actions
            
            # Normalize delta action
            delta_action_normalized = (delta_action - action_mean) / action_std
            
            # Forward pass: (current_image, current_state, delta_action) -> next_image_pred
            next_image_pred = model(curr_image, curr_state_normalized, delta_action_normalized)
            
            # Compute loss
            loss = criterion(next_image_pred, next_image_target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i}/{len(train_loader)} | Train Loss: {loss.item():.6f} | Last Val Loss: {best_val_loss:.6f}")
                
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ======================================================================
        # Validation Phase
        # ======================================================================
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch[image_key].to(DEVICE)
                curr_image = images[:, 0, :, :, :]
                next_image_target = images[:, 1, :, :, :]
                
                if curr_image.max() > 1.0:
                    curr_image = curr_image / 255.0
                    next_image_target = next_image_target / 255.0
                
                states = batch["observation.state"].to(DEVICE)
                curr_state = states[:, 0, :] if states.ndim == 3 else states
                curr_state_normalized = (curr_state - state_mean) / state_std
                
                if use_delta_action:
                    delta_action = compute_delta_action(batch, config).to(DEVICE)
                else:
                    actions = batch["action"].to(DEVICE)
                    delta_action = actions[:, 0, :] if actions.ndim == 3 else actions
                
                delta_action_normalized = (delta_action - action_mean) / action_std
                
                next_image_pred = model(curr_image, curr_state_normalized, delta_action_normalized)
                loss = criterion(next_image_pred, next_image_target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Save checkpoint after every epoch
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'image_size': image_size,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
            'in_channels': C,
            'frame_skip': frame_skip,
            'use_delta_action': use_delta_action,
            'config': config,
            'normalization_stats': normalization_stats,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        
        # Update best validation loss and save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save(checkpoint, best_path)
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | NEW BEST! ✓")
        else:
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
    # Save model with normalization statistics
    save_path = "image_world_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'image_size': image_size,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'in_channels': C,
        'frame_skip': frame_skip,
        'use_delta_action': use_delta_action,
        'config': config,
        # Normalization statistics for inference
        'normalization_stats': normalization_stats,
    }, save_path)
    print(f"\nModel saved to {save_path}")
    print("Normalization statistics saved with model for inference!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Image World Model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--frame_skip", type=int, default=5, help="Frame skip between current and next")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--no_delta_action", action="store_true", help="Use absolute action instead of delta")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    args = parser.parse_args()
    
    train(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        frame_skip=args.frame_skip,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        use_delta_action=not args.no_delta_action,
        val_split=args.val_split,
    )
