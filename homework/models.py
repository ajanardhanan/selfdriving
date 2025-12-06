from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input dimension calculation:
        # Left track: n_track points * 2 coordinates (x, y)
        # Right track: n_track points * 2 coordinates (x, y)
        input_dim = (n_track * 2) + (n_track * 2)
        
        # Output dimension calculation:
        # n_waypoints * 2 coordinates (x, y)
        output_dim = n_waypoints * 2
        
        # Hidden layer size (Hyperparameter)
        # 64 or 128 is usually sufficient for this complexity
        hidden_dim = 128

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Batch size
        B = track_left.shape[0]

        # 1. Flatten inputs
        # Reshape track_left from (B, n_track, 2) -> (B, n_track * 2)
        left_flat = track_left.view(B, -1)
        # Reshape track_right from (B, n_track, 2) -> (B, n_track * 2)
        right_flat = track_right.view(B, -1)

        # 2. Concatenate features
        # Shape becomes (B, input_dim)
        x = torch.cat([left_flat, right_flat], dim=1)

        # 3. Pass through MLP
        # Shape becomes (B, output_dim)
        out = self.model(x)

        # 4. Reshape output to desired signature
        # (B, n_waypoints * 2) -> (B, n_waypoints, 2)
        return out.view(B, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            d_model (int): latent dimension size
            nhead (int): number of attention heads
            num_layers (int): number of transformer decoder layers
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # 1. Input Projection:
        # Encodes the "Byte Array" (Lane boundaries) from (x, y) -> d_model
        self.input_proj = nn.Linear(2, d_model)

        # 2. Learnable Query Embeddings:
        # The "Latent Array" that will query the lane features
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # 3. Transformer Decoder:
        # We use batch_first=True so input tensors are (Batch, Seq, Feature)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. Output Projection:
        # Decodes the processed queries back to (x, y) coordinates
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints using cross-attention between learned queries and track boundaries.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.shape[0]

        # --- Prepare Keys and Values (The Lane Boundaries) ---
        # Concatenate left and right tracks: (B, n_track, 2) -> (B, 2 * n_track, 2)
        tracks = torch.cat([track_left, track_right], dim=1)
        
        # Project to embedding dimension: (B, 20, d_model)
        # This acts as the 'memory' for the transformer
        memory = self.input_proj(tracks)

        # --- Prepare Queries (The Learned Waypoints) ---
        # Get learned embeddings: (n_waypoints, d_model)
        queries = self.query_embed.weight
        
        # Expand queries for the batch: (B, n_waypoints, d_model)
        tgt = queries.unsqueeze(0).expand(B, -1, -1)

        # --- Cross Attention ---
        # Pass through Transformer Decoder
        # tgt (Queries) attends to memory (Keys/Values)
        # Output shape: (B, n_waypoints, d_model)
        out = self.transformer_decoder(tgt=tgt, memory=memory)

        # --- Final Prediction ---
        # Project back to 2D coordinates: (B, n_waypoints, 2)
        waypoints = self.output_proj(out)

        return waypoints


# Constants assumed to be defined in the environment; 
# providing standard ImageNet values here for completeness.
INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # 5-Layer Strided CNN Backbone
        # Input shape: (B, 3, 96, 128)
        self.net = nn.Sequential(
            # Layer 1: 96x128 -> 48x64
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2: 48x64 -> 24x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3: 24x32 -> 12x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 4: 12x16 -> 6x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 5: 6x8 -> 3x4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Output Head
        # Flattened size calculation: 
        # Channels (256) * Height (3) * Width (4) = 3072
        self.fc = nn.Linear(3072, n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        # 1. Normalize input
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # 2. Extract Features
        # Shape: (B, 256, 3, 4)
        features = self.net(x)

        # 3. Flatten
        # Shape: (B, 3072)
        flat_features = features.flatten(1)

        # 4. Predict linear output
        # Shape: (B, n_waypoints * 2)
        out = self.fc(flat_features)

        # 5. Reshape to required output format
        # Shape: (B, n_waypoints, 2)
        return out.view(-1, self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
