# rain-paradox.py
from dataclasses import dataclass
from typing import List, Iterable
import math
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Person:
    """Simple box-model person with rectangular projections."""
    height_feet: float
    shoulder_width_feet: float
    depth_feet: float

    @property
    def top_area_sqft(self) -> float:
        """Horizontal projection (head/shoulders/back)."""
        return self.shoulder_width_feet * self.depth_feet

    @property
    def front_area_sqft(self) -> float:
        """Vertical projection against the direction of travel."""
        return self.height_feet * self.shoulder_width_feet


@dataclass(frozen=True)
class Rain:
    """
    Rain characterization.

    intensity_drops_per_sqft_s: raindrop flux through a horizontal plane
    (drops per square foot per second).
    fall_speed_ft_s: approximate terminal velocity of drops (ft/s).
    """
    intensity_drops_per_sqft_s: float
    fall_speed_ft_s: float


def intensity_from_inches_per_hour(
    rainfall_inches_per_hour: float,
    drop_diameter_mm: float = 2.0
) -> float:
    """
    Convert rainfall rate (inches/hour) to raindrop intensity
    (drops/ft^2/s), assuming spherical drops with a mean diameter.

    Steps:
    - Convert in/hr -> ft/s volumetric flux (ft^3 / (ft^2 * s)) = ft/s
    - Drop volume V = 4/3 * pi * r^3 (ft^3)
    - Intensity = flux / drop_volume (drops / ft^2 / s)
    """
    # Rain depth flux (ft/s)
    depth_flux_ft_per_s = (rainfall_inches_per_hour / 12.0) / 3600.0

    # Drop diameter in feet
    drop_diameter_ft = (drop_diameter_mm / 1000.0) * 3.28084
    drop_radius_ft = drop_diameter_ft / 2.0
    drop_volume_cuft = (
        (4.0 / 3.0) * math.pi * (drop_radius_ft ** 3)
    )

    if drop_volume_cuft <= 0:
        raise ValueError("Drop diameter must be > 0")

    intensity = depth_flux_ft_per_s / drop_volume_cuft
    return intensity


def simulate_wetness(
    speed_ft_s: float,
    distance_feet: float,
    person: Person,
    rain: Rain,
) -> float:
    """
    Total expected raindrop collisions ("wetness") while
    moving to shelter.

    Parameters:
      - speed_ft_s: movement speed (ft/s)
      - distance_feet: distance to shelter (ft)
      - person: body dimensions
      - rain: rain parameters

    Model:
      - From above:
        intensity * top_area * (distance / speed)
      - From front:
        (intensity / fall_speed) * front_area * distance
    """
    if speed_ft_s <= 0:
        raise ValueError("Speed must be positive")

    intensity = rain.intensity_drops_per_sqft_s
    fall_speed = rain.fall_speed_ft_s
    top_area = person.top_area_sqft
    front_area = person.front_area_sqft

    rain_from_above = intensity * top_area * (distance_feet / speed_ft_s)
    rain_from_front = (intensity / fall_speed) * front_area * distance_feet

    return rain_from_above + rain_from_front


def plot_wetness_vs_speed(
    speeds_ft_s: Iterable[float],
    distance_feet: float,
    person: Person,
    rain: Rain,
    title: str = "Wetness vs. Speed in Rain",
) -> None:
    """Plot wetness vs speed with asymptote line included."""
    speeds = list(speeds_ft_s)
    wetness_values = [
        simulate_wetness(speed, distance_feet, person, rain)
        for speed in speeds
    ]

    # Asymptote = frontal-only term (independent of speed)
    intensity = rain.intensity_drops_per_sqft_s
    fall_speed = rain.fall_speed_ft_s
    frontal_limit = (
        (intensity / fall_speed) *
        person.front_area_sqft *
        distance_feet
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        speeds,
        wetness_values,
        marker="o",
        linestyle="-",
        color="blue",
        label="Total wetness"
    )
    plt.axhline(
        frontal_limit,
        color="crimson",
        linestyle="--",
        label="Frontal-only limit (as speed→∞)"
    )

    plt.title(title)
    plt.xlabel("Speed (ft/s)")
    plt.ylabel("Total Raindrops (expected)")
    plt.grid(True)
    plt.xticks(speeds)
    plt.legend()
    plt.tight_layout()
    plt.show()


def mph_to_ft_s(mph: float) -> float:
    """Convert miles per hour to feet per second."""
    return mph * 5280.0 / 3600.0


if __name__ == "__main__":
    # Default person dimensions (~5'10" tall;
    # ~20" shoulder width; ~12" depth)
    person = Person(
        height_feet=70 / 12,
        shoulder_width_feet=20 / 12,
        depth_feet=12 / 12
    )

    # Example rain: 0.2 in/hr with ~2 mm drops,
    # fall speed ~20 ft/s
    intensity = intensity_from_inches_per_hour(
        0.2, drop_diameter_mm=2.0
    )
    rain = Rain(
        intensity_drops_per_sqft_s=intensity,
        fall_speed_ft_s=20.0
    )

    # Speeds (ft/s): walk -> sprint
    speeds_ft_s: List[float] = [3.3, 5.5, 8.8, 13.2, 30.0]
    distance_feet = 500.0

    # Compute and print wetness values
    wetness_values: List[float] = [
        simulate_wetness(speed, distance_feet, person, rain)
        for speed in speeds_ft_s
    ]
    for speed, wetness in zip(speeds_ft_s, wetness_values):
        print(
            f"Speed: {speed:.1f} ft/s -> "
            f"Wetness: {int(wetness):,} drops"
        )

    # Plot
    plot_wetness_vs_speed(speeds_ft_s, distance_feet, person, rain)
