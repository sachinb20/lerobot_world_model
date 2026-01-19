import torch
from torch.utils.data import DataLoader, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np


def get_dataloader(
    repo_id="sach088/topple_the_dino",
    batch_size=32,
    num_workers=4,
    shuffle=True,
    image_key="observation.images.side",
    frame_skip=5,  # Number of frames to skip between current and next (1 = consecutive frames)
    use_delta_action=True,  # If True, compute action as difference between states
):
    """
    Creates a DataLoader for the World Model training with image observations.
    
    Args:
        repo_id (str): HuggingFace dataset ID.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        image_key (str): Key for image observations (e.g., 'observation.images.side').
        frame_skip (int): Number of frames to skip. 1 = consecutive, 2 = every other, etc.
        use_delta_action (bool): If True, action = next_state - current_state (relative).
                                 If False, use absolute action from dataset.
        
    Returns:
        tuple: (DataLoader, image_key, config_dict)
    """
    
    # First load dataset to get fps and available keys
    dataset = LeRobotDataset(repo_id=repo_id, video_backend="pyav")
    fps = dataset.fps
    
    # Calculate delta time based on frame_skip
    # frame_skip=1 means consecutive frames (dt = 1/fps)
    # frame_skip=2 means skip 1 frame (dt = 2/fps)
    dt = frame_skip / fps
    
    # Get available keys
    available_keys = list(dataset.features.keys())
    print(f"Available keys in dataset: {available_keys}")
    print(f"Dataset FPS: {fps}")
    print(f"Frame skip: {frame_skip} (dt = {dt:.4f}s)")
    
    # Build delta_timestamps dict
    final_deltas = {}
    
    # Add the specified image key
    if image_key in available_keys:
        final_deltas[image_key] = [0, dt]  # Current and next frame
        print(f"Using image key: {image_key}")
    else:
        # Try to find alternative image keys
        image_keys = [k for k in available_keys if "image" in k.lower()]
        if image_keys:
            image_key = image_keys[0]
            final_deltas[image_key] = [0, dt]
            print(f"Warning: Specified image key not found. Using '{image_key}' instead.")
        else:
            raise ValueError(f"No image keys found in dataset! Available: {available_keys}")
    
    # Always add observation.state for delta action computation
    if "observation.state" in available_keys:
        final_deltas["observation.state"] = [0, dt]  # Need both for delta computation
    
    # Add action (even if we compute delta, we might want to compare)
    if "action" in available_keys:
        final_deltas["action"] = [0]
    
    print(f"Delta timestamps config: {final_deltas}")
    print(f"Use delta action: {use_delta_action}")
    
    # Re-init with deltas
    dataset = LeRobotDataset(
        repo_id=repo_id,
        delta_timestamps=final_deltas,
        video_backend="pyav"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True
    )
    
    config = {
        "fps": fps,
        "frame_skip": frame_skip,
        "dt": dt,
        "use_delta_action": use_delta_action,
        "image_key": image_key,
    }
    
    return dataloader, image_key, config


def get_train_val_dataloaders(
    repo_id="sach088/topple_the_dino",
    batch_size=32,
    num_workers=4,
    image_key="observation.images.side",
    frame_skip=5,
    use_delta_action=True,
    val_split=0.1,  # 10% for validation
    seed=42,
):
    """
    Creates train and validation DataLoaders with random sample-based splitting.
    
    Args:
        repo_id (str): HuggingFace dataset ID.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        image_key (str): Key for image observations.
        frame_skip (int): Number of frames to skip.
        use_delta_action (bool): If True, action = next_state - current_state.
        val_split (float): Fraction of samples for validation (default 0.1 = 10%).
        seed (int): Random seed for reproducible splits.
        
    Returns:
        tuple: (train_loader, val_loader, image_key, config)
    """
    from torch.utils.data import random_split
    
    # First load dataset to get fps and available keys
    dataset = LeRobotDataset(repo_id=repo_id, video_backend="pyav")
    fps = dataset.fps
    dt = frame_skip / fps
    
    # Get available keys
    available_keys = list(dataset.features.keys())
    print(f"Available keys in dataset: {available_keys}")
    print(f"Dataset FPS: {fps}")
    print(f"Frame skip: {frame_skip} (dt = {dt:.4f}s)")
    
    # Build delta_timestamps dict
    final_deltas = {}
    
    if image_key in available_keys:
        final_deltas[image_key] = [0, dt]
        print(f"Using image key: {image_key}")
    else:
        image_keys = [k for k in available_keys if "image" in k.lower()]
        if image_keys:
            image_key = image_keys[0]
            final_deltas[image_key] = [0, dt]
            print(f"Warning: Specified image key not found. Using '{image_key}' instead.")
        else:
            raise ValueError(f"No image keys found in dataset! Available: {available_keys}")
    
    if "observation.state" in available_keys:
        final_deltas["observation.state"] = [0, dt]
    
    if "action" in available_keys:
        final_deltas["action"] = [0]
    
    print(f"Delta timestamps config: {final_deltas}")
    print(f"Use delta action: {use_delta_action}")
    
    # Re-init dataset with deltas
    dataset = LeRobotDataset(
        repo_id=repo_id,
        delta_timestamps=final_deltas,
        video_backend="pyav"
    )
    
    # Fast random split using torch.utils.data.random_split
    total_samples = len(dataset)
    val_size = max(1, int(total_samples * val_split))
    train_size = total_samples - val_size
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Train samples: {train_size} ({100*(1-val_split):.0f}%)")
    print(f"Val samples: {val_size} ({100*val_split:.0f}%)")
    
    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    config = {
        "fps": fps,
        "frame_skip": frame_skip,
        "dt": dt,
        "use_delta_action": use_delta_action,
        "image_key": image_key,
    }
    
    return train_loader, val_loader, image_key, config


def compute_delta_action(batch, config):
    """
    Compute delta action from observation states.
    
    Delta action = next_state - current_state (relative joint positions)
    
    Args:
        batch: Batch from dataloader
        config: Config dict from get_dataloader
        
    Returns:
        delta_action: (B, action_dim) tensor of relative joint positions
    """
    if "observation.state" not in batch:
        raise ValueError("Cannot compute delta action: 'observation.state' not in batch")
    
    states = batch["observation.state"]  # (B, T, state_dim) where T=2
    
    if states.ndim == 3 and states.shape[1] >= 2:
        curr_state = states[:, 0, :]  # (B, state_dim)
        next_state = states[:, 1, :]  # (B, state_dim)
        delta_action = next_state - curr_state  # Relative change
        return delta_action
    else:
        raise ValueError(f"Expected states with T>=2, got shape {states.shape}")


if __name__ == "__main__":
    # Test with different frame skips
    print("\n" + "=" * 60)
    print("Testing frame_skip=1 (consecutive frames)")
    print("=" * 60)
    dl, img_key, config = get_dataloader(frame_skip=1, batch_size=2)
    batch = next(iter(dl))
    print("\nBatch keys:", batch.keys())
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}, dtype={v.dtype}")
    
    # Compute delta action
    delta = compute_delta_action(batch, config)
    print(f"\nDelta action shape: {delta.shape}")
    print(f"Delta action sample: {delta[0]}")
    
    print("\n" + "=" * 60)
    print("Testing frame_skip=5 (skip 4 frames)")
    print("=" * 60)
    dl2, _, config2 = get_dataloader(frame_skip=5, batch_size=2)
    batch2 = next(iter(dl2))
    delta2 = compute_delta_action(batch2, config2)
    print(f"Delta action with skip=5: {delta2[0]}")
    print(f"Note: Larger deltas expected with larger frame_skip")
