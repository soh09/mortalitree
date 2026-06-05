"""Clay metadata encodings for the `time` and `latlon` inputs.

Clay's `add_encodings` builds its positional encoding at width `dim - 8` and
reserves the trailing 8 dims for `hstack((time, latlon))`, so each of `time`
and `latlon` must be a 4-element cyclic (sin/cos) encoding (see spec §3.1).
Passing raw 2-dim values makes the metadata 4 wide and the encoding fails to
add to the `dim`-wide patch tokens.
"""
import math

import torch


def encode_time(week: float, hour: float) -> torch.Tensor:
    """Cyclic sin/cos encoding of acquisition time.

    week in 1..52, hour in 0..24. Returns a (4,) float32 tensor:
        [sin(week_a), cos(week_a), sin(hour_a), cos(hour_a)]
    """
    week_a = week * 2.0 * math.pi / 52.0
    hour_a = hour * 2.0 * math.pi / 24.0
    return torch.tensor(
        [math.sin(week_a), math.cos(week_a), math.sin(hour_a), math.cos(hour_a)],
        dtype=torch.float32,
    )


def encode_latlon(lat: float, lon: float) -> torch.Tensor:
    """Cyclic sin/cos encoding of latitude/longitude (degrees).

    Returns a (4,) float32 tensor:
        [sin(lat_r), cos(lat_r), sin(lon_r), cos(lon_r)]
    """
    lat_r = lat * math.pi / 180.0
    lon_r = lon * math.pi / 180.0
    return torch.tensor(
        [math.sin(lat_r), math.cos(lat_r), math.sin(lon_r), math.cos(lon_r)],
        dtype=torch.float32,
    )
