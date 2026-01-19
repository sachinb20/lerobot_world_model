"""
Sanity Check Script for World Model Dataloader

Verifies:
1. Batch contains both time steps (current and next)
2. Frame skip parameter works correctly
3. Delta action computation is correct
4. Images are properly formatted for training
"""

import torch
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from world_model.dataloader import get_dataloader, compute_delta_action


def sanity_check(frame_skip=1):
    repo_id = "sach088/topple_the_dino"
    image_key = "observation.images.side"
    
    print("=" * 60)
    print("SANITY CHECK: World Model Dataloader")
    print("=" * 60)
    
    # Step 1: Load dataset and check available keys
    print("\n[1] Loading dataset...")
    dataset = LeRobotDataset(repo_id=repo_id, video_backend="pyav")
    fps = dataset.fps
    dt = frame_skip / fps
    
    print(f"    Dataset: {repo_id}")
    print(f"    FPS: {fps}")
    print(f"    Frame skip: {frame_skip}")
    print(f"    Delta time (dt): {dt:.4f}s")
    print(f"    Total episodes: {dataset.num_episodes}")
    print(f"    Total frames: {len(dataset)}")
    
    # Step 2: Check available keys
    print("\n[2] Available keys in dataset:")
    available_keys = list(dataset.features.keys())
    for k in available_keys:
        print(f"    - {k}")
    
    # Step 3: Use the dataloader with frame_skip
    print(f"\n[3] Testing dataloader with frame_skip={frame_skip}...")
    dataloader, img_key, config = get_dataloader(
        repo_id=repo_id,
        batch_size=2,
        frame_skip=frame_skip,
        use_delta_action=True,
        image_key=image_key
    )
    
    print(f"    Config: {config}")
    
    # Step 4: Get a sample batch
    print("\n[4] Checking batch data shapes...")
    batch = next(iter(dataloader))
    
    print(f"    Batch keys: {batch.keys()}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    
    # Step 5: Verify image has 2 time steps
    images = batch[img_key]
    print(f"\n[5] Verifying image temporal dimension...")
    
    if images.ndim == 5:
        B, T, C, H, W = images.shape
        print(f"    Image tensor shape: (B={B}, T={T}, C={C}, H={H}, W={W})")
        
        if T == 2:
            print(f"    [✓] Correct! Found 2 time steps (current and next)")
        else:
            print(f"    [!] WARNING: Expected T=2, got T={T}")
    else:
        print(f"    [!] ERROR: Expected 5D tensor (B, T, C, H, W), got {images.ndim}D")
        return
    
    # Step 6: Verify delta action computation
    print("\n[6] Verifying delta action computation...")
    
    if "observation.state" in batch:
        states = batch["observation.state"]
        print(f"    State shape: {states.shape}")
        
        curr_state = states[:, 0, :]
        next_state = states[:, 1, :]
        
        print(f"    Current state (sample 0): {curr_state[0]}")
        print(f"    Next state (sample 0): {next_state[0]}")
        
        # Compute delta action
        delta_action = compute_delta_action(batch, config)
        manual_delta = next_state - curr_state
        
        print(f"    Delta action (next - curr): {delta_action[0]}")
        
        # Verify computation
        if torch.allclose(delta_action, manual_delta):
            print(f"    [✓] Delta action computation verified!")
        else:
            print(f"    [!] Delta action mismatch!")
        
        # Verify: current_state + delta_action = next_state
        print(f"\n    Verifying: current_state + delta_action = next_state")
        reconstructed_next_state = curr_state + delta_action
        print(f"    Reconstructed next state (sample 0): {reconstructed_next_state[0]}")
        print(f"    Actual next state (sample 0):        {next_state[0]}")
        
        if torch.allclose(reconstructed_next_state, next_state, atol=1e-6):
            print(f"    [✓] Verified: current_state + delta_action = next_state")
        else:
            diff = (reconstructed_next_state - next_state).abs().max()
            print(f"    [!] Mismatch! Max difference: {diff:.8f}")
    else:
        print(f"    [!] No observation.state in batch, cannot verify delta action")
    
    # Step 7: Verify frame skip works by comparing direct access
    print(f"\n[7] Verifying frame skip={frame_skip}...")
    
    sample_idx = 50
    
    # Get frame directly at sample_idx
    direct_curr = dataset[sample_idx]
    direct_next = dataset[sample_idx + frame_skip]
    
    print(f"    Direct access frame {sample_idx}: state = {direct_curr['observation.state'][:3]}...")
    print(f"    Direct access frame {sample_idx + frame_skip}: state = {direct_next['observation.state'][:3]}...")
    
    expected_delta = direct_next['observation.state'] - direct_curr['observation.state']
    print(f"    Expected delta (manual): {expected_delta[:3]}...")
    
    # Step 8: Visualize the frames
    print("\n[8] Saving visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Current frame
    curr_img = images[0, 0].permute(1, 2, 0).numpy()
    if curr_img.max() > 1.0:
        curr_img = curr_img / 255.0
    axes[0, 0].imshow(curr_img)
    axes[0, 0].set_title(f"Current Frame (t=0)")
    axes[0, 0].axis('off')
    
    # Next frame
    next_img = images[0, 1].permute(1, 2, 0).numpy()
    if next_img.max() > 1.0:
        next_img = next_img / 255.0
    axes[0, 1].imshow(next_img)
    axes[0, 1].set_title(f"Next Frame (t={frame_skip}/fps)")
    axes[0, 1].axis('off')
    
    # Difference
    diff_img = (next_img - curr_img)
    diff_img = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min() + 1e-8)
    axes[0, 2].imshow(diff_img)
    axes[0, 2].set_title("Image Difference (normalized)")
    axes[0, 2].axis('off')
    
    # Plot state comparison
    if "observation.state" in batch:
        state_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        x = range(len(state_names))
        
        axes[1, 0].bar(x, curr_state[0].numpy(), alpha=0.7, label='Current')
        axes[1, 0].bar(x, next_state[0].numpy(), alpha=0.7, label='Next')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(state_names, rotation=45, ha='right')
        axes[1, 0].set_title("Joint States")
        axes[1, 0].legend()
        
        # Delta action
        axes[1, 1].bar(x, delta_action[0].numpy(), color='green', alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(state_names, rotation=45, ha='right')
        axes[1, 1].set_title(f"Delta Action (frame_skip={frame_skip})")
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    save_path = f"sanity_check_frames_skip{frame_skip}.png"
    plt.savefig(save_path, dpi=150)
    print(f"    Saved to: {save_path}")
    
    # Step 9: Summary
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)
    print(f"  Dataset: {repo_id}")
    print(f"  Image key: {img_key}")
    print(f"  Image shape: (B, T={T}, C={C}, H={H}, W={W})")
    print(f"  Frame skip: {frame_skip} (dt = {dt:.4f}s)")
    print(f"  Action type: Delta (relative joint positions)")
    print(f"  Action dim: {delta_action.shape[-1] if 'observation.state' in batch else 'N/A'}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_skip", type=int, default=10, help="Frame skip value to test")
    args = parser.parse_args()
    
    sanity_check(frame_skip=args.frame_skip)
