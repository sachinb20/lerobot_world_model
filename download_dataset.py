from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    repo_id = "sach088/topple_the_dino"
    print(f"Downloading dataset: {repo_id}")
    
    # Instantiating LeRobotDataset triggers the download/caching if not already present
    dataset = LeRobotDataset(repo_id=repo_id)
    
    print(f"Dataset {repo_id} loaded successfully.")
    print(f"Total episodes: {dataset.meta.total_episodes}")
    print(f"Total frames: {dataset.meta.total_frames}")
    print(f"FPS: {dataset.meta.fps}")
    print(f"Features: {dataset.features}")
    print("Download and verification complete.")

if __name__ == "__main__":
    main()
