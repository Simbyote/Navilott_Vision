
from dataclasses import dataclass

@dataclass
class EncoderFrame:
    """Aggregated encoder values for one pipeline frame window."""
    wheel_speed: float | None = None        # m/s
    distance_traveled: float | None = None  # m, since last frame
    sample_count: int = 0

    @property
    def valid(self) -> bool:
        return self.sample_count > 0

@dataclass
class EncoderReader:
    """
    Initializes the N20 encoders and manages a background sampling thread.
    """

    def snapshot(self) -> EncoderFrame:
        raise NotImplementedError