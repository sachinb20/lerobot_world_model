"""
Compute and save normalization statistics for the world model.
Run this once before training to pre-compute stats.
"""
import torch
import os
from lerobot_world_model.dataloader import get_dataloader, compute_delta_action


def compute_and_save_stats(
    repo_id="sach088/topple_the_dino",
    image_key="observation.images.side",
    frame_skip=5,
    use_delta_action=True,
    save_path="normalization_stats.pt",
    batch_size=32,
):
    """
    Compute normalization statistics from the full dataset and save to file.
    
    Args:
        repo_id: HuggingFace dataset ID
        image_key: Key for image observations
        frame_skip: Number of frames to skip
        use_delta_action: If True, compute delta action stats
        save_path: Path to save the stats file
        batch_size: Batch size for loading data
    """
    print(f"\n{'='*60}")
    print("COMPUTING NORMALIZATION STATISTICS")
    print(f"{'='*60}")
    print(f"Dataset: {repo_id}")
    print(f"Frame skip: {frame_skip}")
    print(f"Use delta action: {use_delta_action}")
    print(f"Save path: {save_path}")
    print(f"{'='*60}\n")
    
    # Load full dataset
    dataloader, image_key, config = get_dataloader(
        repo_id=repo_id,
        batch_size=batch_size,
        image_key=image_key,
        frame_skip=frame_skip,
        use_delta_action=use_delta_action,
        shuffle=False,  # No need to shuffle for stats
    )
    
    print(f"Total batches: {len(dataloader)}")
    print("Computing statistics from full dataset...")
    
    all_states = []
    all_delta_actions = []
    
    for i, batch in enumerate(dataloader):
        states = batch["observation.state"]
        curr_state = states[:, 0, :] if states.ndim == 3 else states
        all_states.append(curr_state)
        
        if use_delta_action:
            delta_action = compute_delta_action(batch, config)
        else:
            actions = batch["action"]
            delta_action = actions[:, 0, :] if actions.ndim == 3 else actions
        all_delta_actions.append(delta_action)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(dataloader)} batches...")
    
    all_states = torch.cat(all_states, dim=0)
    all_delta_actions = torch.cat(all_delta_actions, dim=0)
    
    print(f"\nTotal samples: {len(all_states)}")
    
    # Compute mean and std
    state_mean = all_states.mean(dim=0)
    state_std = all_states.std(dim=0) + 1e-8
    
    action_mean = all_delta_actions.mean(dim=0)
    action_std = all_delta_actions.std(dim=0) + 1e-8
    
    # Create stats dict
    normalization_stats = {
        'state_mean': state_mean,
        'state_std': state_std,
        'action_mean': action_mean,
        'action_std': action_std,
        'image_scale': 255.0,
        # Metadata
        'repo_id': repo_id,
        'frame_skip': frame_skip,
        'use_delta_action': use_delta_action,
        'num_samples': len(all_states),
    }
    
    # Save
    torch.save(normalization_stats, save_path)
    
    print(f"\n{'='*60}")
    print("STATISTICS COMPUTED AND SAVED")
    print(f"{'='*60}")
    print(f"  State mean: {state_mean}")
    print(f"  State std: {state_std}")
    print(f"  Action mean: {action_mean}")
    print(f"  Action std: {action_std}")
    print(f"\nSaved to: {save_path}")
    print(f"{'='*60}\n")
    
    return normalization_stats


def load_stats(stats_path="normalization_stats.pt"):
    """Load pre-computed normalization statistics."""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stats file not found: {stats_path}\n"
            f"Run `python -m lerobot_world_model.compute_stats` first to compute stats."
        )
    
    stats = torch.load(stats_path, weights_only=True)
    print(f"Loaded normalization stats from {stats_path}")
    print(f"  Computed from {stats.get('num_samples', 'unknown')} samples")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument("--repo_id", type=str, default="sach088/topple_the_dino", help="Dataset repo ID")
    parser.add_argument("--frame_skip", type=int, default=5, help="Frame skip")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for loading")
    parser.add_argument("--save_path", type=str, default="normalization_stats.pt", help="Output path")
    parser.add_argument("--no_delta_action", action="store_true", help="Use absolute action")
    args = parser.parse_args()
    
    compute_and_save_stats(
        repo_id=args.repo_id,
        frame_skip=args.frame_skip,
        use_delta_action=not args.no_delta_action,
        save_path=args.save_path,
        batch_size=args.batch_size,
    )
