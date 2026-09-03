import math
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

EARTH_RADIUS_METERS = 6_371_000.0

def _local_xy_to_latlon(x_m: torch.Tensor, y_m: torch.Tensor,
                         origin_lat_deg: float, origin_lon_deg: float):
    origin_lat = torch.tensor(origin_lat_deg, dtype=x_m.dtype, device=x_m.device)
    origin_lon = torch.tensor(origin_lon_deg, dtype=x_m.dtype, device=x_m.device)
    lat = origin_lat + torch.rad2deg(y_m / EARTH_RADIUS_METERS)
    lon = origin_lon + torch.rad2deg(
        x_m / (EARTH_RADIUS_METERS * math.cos(math.radians(origin_lat_deg)))
    )
    return lat, lon


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.model = backbone

        self._origin_lat = 39.951633240621874
        self._origin_lon = -75.19156205248895
        self._x_mean     = -4.2876779234493496e-10
        self._x_std      = 49.6173502588834
        self._y_mean     = 7.318392430363547e-10
        self._y_std      = 75.69032697639024


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        if isinstance(batch, list):
            batch = torch.stack(batch)

        self.eval()
        with torch.no_grad():
            pred_norm = self.forward(batch)

        pred_x_m = pred_norm[:, 0] * self._x_std + self._x_mean
        pred_y_m = pred_norm[:, 1] * self._y_std + self._y_mean

        pred_lat, pred_lon = _local_xy_to_latlon(
            pred_x_m, pred_y_m, self._origin_lat, self._origin_lon
        )

        return torch.stack([pred_lat, pred_lon], dim=1)