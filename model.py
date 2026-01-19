import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """CNN Encoder for image observations."""
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        # Input: (B, C, H, W) - typical image sizes 64x64, 128x128, 224x224
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # /8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # /16
            nn.ReLU(),
        )
        # After conv: (B, 256, H/16, W/16)
        # We'll use adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class ImageDecoder(nn.Module):
    """CNN Decoder to reconstruct images from latent space."""
    def __init__(self, latent_dim=256, out_channels=3, output_size=(128, 128)):
        super().__init__()
        self.output_size = output_size
        
        # Calculate initial feature map size
        self.init_h = output_size[0] // 16
        self.init_w = output_size[1] // 16
        
        self.fc = nn.Linear(latent_dim, 256 * self.init_h * self.init_w)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # x2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # x4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # x16
            nn.Sigmoid(),  # Output in [0, 1] range for images
        )
        
    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.init_h, self.init_w)
        x = self.deconv(x)
        # Ensure output matches expected size
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x


class DynamicsModel(nn.Module):
    """Predicts next latent state from current latent state, joint state, and delta action."""
    def __init__(self, latent_dim, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        # Input: image latent + joint state + delta action
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, z, state, delta_action):
        """
        Args:
            z: Image latent (B, latent_dim)
            state: Current joint state (B, state_dim)
            delta_action: Delta action (B, action_dim)
        """
        x = torch.cat([z, state, delta_action], dim=-1)
        return self.net(x)


class ImageWorldModel(nn.Module):
    """
    World Model for image-based state prediction.
    
    Inputs:
        - Current joint state
        - Current image
        - Delta action
    
    Output:
        - Predicted next frame (image)
    """
    def __init__(self, state_dim, action_dim, in_channels=3, latent_dim=256, hidden_dim=512, image_size=(128, 128)):
        super().__init__()
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.encoder = ImageEncoder(in_channels, latent_dim)
        self.dynamics = DynamicsModel(latent_dim, state_dim, action_dim, hidden_dim)
        self.decoder = ImageDecoder(latent_dim, in_channels, image_size)
        
    def forward(self, image, state, delta_action):
        """
        Predicts next image state given current image, joint state, and delta action.
        
        Args:
            image: Current image observation (B, C, H, W)
            state: Current joint state (B, state_dim)
            delta_action: Delta action (B, action_dim)
            
        Returns:
            next_image_pred: Predicted next image (B, C, H, W)
        """
        # Encode current image to latent space
        z_curr = self.encoder(image)
        
        # Predict next latent using image latent + joint state + delta action
        z_next_pred = self.dynamics(z_curr, state, delta_action)
        
        # Decode to predicted next image
        next_image_pred = self.decoder(z_next_pred)
        
        return next_image_pred
    
    def encode(self, image):
        """Encode image to latent space."""
        return self.encoder(image)
    
    def decode(self, z):
        """Decode latent to image."""
        return self.decoder(z)
    
    def predict_latent(self, z, state, delta_action):
        """Predict next latent from current latent, joint state, and delta action."""
        return self.dynamics(z, state, delta_action)


# Keep the original WorldModel for backward compatibility
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.net(z)


class WorldModel(nn.Module):
    """Original MLP-based world model for state vectors."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(state_dim, hidden_dim, latent_dim)
        self.dynamics = DynamicsModel(latent_dim, action_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, state_dim)
        
    def forward(self, state, action):
        z_curr = self.encoder(state)
        z_next_pred = self.dynamics(z_curr, action)
        next_state_pred = self.decoder(z_next_pred)
        return next_state_pred, z_curr, z_next_pred
