+++
title = "Modeling Heat Capacity and Evaporation with Python: Why Water Warms Slowly but Cools Fast"
subtitle = "Using thermodynamics and Python to explain why pools resist daytime warming but lose heat rapidly at night"
date = "2025-10-12T09:00:00-07:00"
draft = false
categories = ["Applied Modeling and Simulation"]
tags = ["energy balance", "Python", "thermodynamics",]
author = "Tom Archer"
listThumb = "thermo-water-energy-balance-thumb.png"

hero = "thermo-water-energy-balance.png"
heroAlt = "Educational illustration of a swimming pool highlighting heat capacity and evaporation, with sun and moon icons and heat arrows"
heroLabel = "Open full-size thermo water energy balance"
heroCaption = "Water warms slowly but cools fast because high heat capacity resists change while evaporation accelerates night losses."

whatYoullLearn = [
    "Why water's high specific heat makes a swimming pool warm much more slowly than the surrounding air",
    "Why evaporation can remove heat from water faster than sunlight replaces it",
    "How convection, radiation, solar absorption, and evaporation combine in a surface energy balance",
    "How to translate conservation of energy into a Python model of water temperature",
    "Why covering a pool can dramatically reduce nighttime heat loss",
    "How simulation can reveal the effects of weather, geometry, and surface conditions on water temperature"
]
+++

Every summer, it feels like a small miracle when the pool finally warms up enough to swim. In Nevada, where the air temperature can sit above 100°F (38°C) for weeks, you'd expect the water to keep pace. Yet, somehow, it takes forever to warm, and only a few cool nights can undo all that progress.

The same phenomenon shows up in a stick of butter. Butter melts quickly, while margarine stays stubbornly firm even under the same heat. That's not coincidence; it's {{< term "thermodynamics" "thermodynamics" >}}.

The butter versus margarine comparison is a staple example in nutrition science. It shows how the proportions of fat, water, and solids affect how much energy it takes to change temperature. Butter, with more fat and less water, heats up and melts quickly. Margarine, full of water and unsaturated oils, absorbs more energy before softening because water's *{{< term "specific-heat" "specific heat" >}}* is much higher.


A swimming pool works the same way, just scaled up thousands of times. Its massive water content means it has enormous {{< term "heat-capacity" "heat capacity" >}} where warming takes a long time because every degree requires tremendous energy. Cooling happens faster, though, because {{< term "evaporation" "evaporation" >}} and night radiation pull energy out far more efficiently than the sun can replace it.

In this post, we will turn that intuition into a simple Python model that explains exactly why water warms so slowly but cools so fast.

---

## The Thought Experiment

Imagine a shallow pool sitting in the desert sun. During the day, it absorbs energy from sunlight, and at night it loses energy to the air and sky. The surprising part is not that both happen, but that the rates are so uneven.

The pool's heat capacity acts like a huge thermal battery; it takes time to charge. The cooling process, on the other hand, is driven by evaporation. Each gram of water that evaporates carries away about 2.45 kilojoules of energy, and with enough dry air and a light breeze, those losses pile up quickly.

The same logic explains why a pat of margarine melts more slowly than butter. Water is a powerful {{< term "heat-sink" "heat sink" >}}, and evaporation is a powerful energy thief. Put them together, and you have a recipe for long warmups and quick cooldowns.

---

## Modeling the Problem

We will treat the water as a well mixed {{< term "control-volume" "control volume" >}} with a single temperature. The surface exchanges heat with the air through {{< term "convection" "convection" >}}, with the sky through {{< term "longwave-radiation" "longwave radiation" >}}, and with the sun through {{< term "shortwave-radiation" "shortwave absorption" >}}. Evaporation removes energy proportional to the rate at which vapor escapes.

**Assumptions:**

- The water is well mixed with no {{< term "thermal-stratification" "stratification" >}}. This means that the water's temperature is assumed to be uniform throughout, rather than layered (or stratified). In real life, large bodies of water often stratify, meaning warmer, lighter water floats on top while cooler, denser water sinks.
- Conduction through walls and floor is small compared with surface fluxes. In other words, most heat exchange happens through the surface where the water meets the air, not through the pool's sides or bottom. This fact allows us to ignore the smaller conductive losses into the ground or walls.
- Shortwave absorption is a constant fraction of incident {{< term "irradiance" "irradiance" >}}. The sunlight that hits the surface is partly reflected and partly absorbed. Here, we assume that a fixed portion (often 85–90%) is absorbed as heat, rather than modeling how absorption varies with sun angle or surface color.
- Longwave exchange follows the {{< term "stefan-boltzmann-law" "Stefan–Boltzmann law" >}} with an effective sky temperature. The pool radiates heat upward to the sky, and the sky radiates a smaller amount back. We model this as a simple radiation balance using the Stefan–Boltzmann law, where the "sky temperature" represents the average radiative temperature of the atmosphere above.
- Evaporation follows a bulk aerodynamic relationship that depends on humidity gradient and wind. The rate of evaporation depends on how dry the air is and how quickly air moves over the surface. Wind helps carry away moist air, increasing evaporation, while higher humidity slows it down.
- Weather varies smoothly over the day and repeats during the simulation window. Instead of using random or hourly data, we assume repeating daily cycles for temperature, humidity, sunlight, and wind. This simplifies the simulation while still capturing the main {{< term "diurnal-cycle" "diurnal" >}} pattern of heating and cooling.

The **energy balance** is defined as follows using the standard form of the surface energy balance used throughout thermodynamics, climatology, and environmental engineering. In academic terms, this is the first-law ({{< term "conservation-of-energy" "conservation of energy" >}}) equation for a control volume, applied to a water surface.

\\[
C \frac{dT}{dt} = Q_{\text{solar}} + Q_{\text{convective}} + Q_{\text{radiative}} + Q_{\text{evaporative}}
\\]

where:

\\[
C = m c_p
\\]

\\[
Q_{\text{solar}} = \alpha A I(t)
\\]

\\[
Q_{\text{convective}} = h_c A [T_{\text{air}}(t) - T(t)]
\\]

\\[
Q_{\text{radiative}} = \varepsilon \sigma A [T_{\text{sky}}(t)^4 - T(t)^4]
\\]

\\[
Q_{\text{evaporative}} = -L_v \dot{m}_{\text{evap}}
\\]

---

## Using AI to Scaffold the Code

We'll begin by using AI to generate the helper functions and constants needed to support our simulation.

**Prompt example:** When you ask an AI assistant to provide scaffolding, be specific about inputs, outputs, and units. Request pure functions for each physical term, expressed in SI units, and constants defined once. Ask for no plotting and no input or output inside the helpers so that you can test them easily.

> Write a single Python module that defines helper functions for a simple water-surface energy balance, and include only valid Python code in the output.
>
> Requirements:
>
> 1. Import NumPy as `np`.
> 2. Define these constants (SI units):
>
>    * `SIGMA = 5.670374419e-8`  # Stefan–Boltzmann W·m^-2·K^-4
>    * `LV = 2.45e6`             # {{< term "latent-heat-of-vaporization" "latent heat of vaporization" >}} J·kg^-1
>    * `CP = 4186.0`             # specific heat of water J·kg^-1·K^-1
>    * `RHO = 1000.0`            # density of water kg·m^-3
> 3. Define temperature conversion helpers:
>
>    * `c_to_f(c)` returns Fahrenheit from Celsius
>    * `f_to_c(f)` returns Celsius from Fahrenheit
>    * `c_to_k(C)` returns Kelvin from Celsius
> 4. Define saturation vapor pressure over water (Pa) as a function of Celsius using Magnus:
>
>    * `esat(C) = 610.94 * exp(17.625*C / (C + 243.04))`
> 5. Define an evaporation mass flux helper (kg·s^-1) using a bulk aerodynamic form:
>
>    * `evap_flux(TwC, TairC, RH, wind, A, k_evap=2.5e-7, beta_wind=0.1)`
>      where `RH` is fractional 0..1, `wind` is m·s^-1, `A` is m^2.
>    * Compute `es = esat(TwC)`, `ea = RH * esat(TairC)`, `grad = max(es - ea, 0)`,
>      and return `k_evap * A * grad * (1.0 + beta_wind * wind)`.
> 6. Define heat flux terms that return Watts (W):
>
>    * `q_solar(I, A, alpha)` returns `alpha * A * I` where `I` is W·m^-2
>    * `q_conv(TwK, TairK, A, h_c)` returns `h_c * A * (TairK - TwK)`
>    * `q_rad(TwK, TskyK, A, epsilon)` returns `epsilon * SIGMA * A * (TskyK**4 - TwK**4)`
>    * `q_evap(TwC, TairC, RH, wind, A, k_evap=2.5e-7, beta_wind=0.1)` computes `mdot = evap_flux(...)` and returns `-LV * mdot`
> 7. Define a diurnal helper `diurnal(base, amp, hours, peak_shift=7.0)` that returns `base + amp * sin(2*pi*(hours - peak_shift)/24.0)`.
> 8. Do not include any print statements, main guards, or plotting.
> 9. Output only the code with the exact function names and signatures specified.

**Python:** Your LLM should respond with code similar to the following:

```python
import numpy as np

SIGMA = 5.670374419e-8
LV = 2.45e6
CP = 4186.0
RHO = 1000.0

def c_to_f(c):
    return c * 9.0/5.0 + 32.0

def f_to_c(f):
    return (f - 32.0) * 5.0/9.0

def c_to_k(C):
    return C + 273.15

def esat(C):
    return 610.94 * np.exp(17.625 * C / (C + 243.04))

def evap_flux(TwC, TairC, RH, wind, A, k_evap=2.5e-7, beta_wind=0.1):
    es = esat(TwC)
    ea = RH * esat(TairC)
    grad = np.maximum(es - ea, 0.0)
    return k_evap * A * grad * (1.0 + beta_wind * wind)

def q_solar(I, A, alpha):
    return alpha * A * I

def q_conv(TwK, TairK, A, h_c):
    return h_c * A * (TairK - TwK)

def q_rad(TwK, TskyK, A, epsilon):
    return epsilon * SIGMA * A * (TskyK**4 - TwK**4)

def q_evap(TwC, TairC, RH, wind, A, k_evap=2.5e-7, beta_wind=0.1):
    mdot = evap_flux(TwC, TairC, RH, wind, A, k_evap, beta_wind)
    return -LV * mdot

def diurnal(base, amp, hours, peak_shift=7.0):
    return base + amp * np.sin(2*np.pi * (hours - peak_shift) / 24.0)
```

---

## Simulation: Diurnal Heating and Night Cooling

**Prompt example:** Now we tell AI to generate a fairly tractable function to simulate the diurnal heating and cooling of a body of water over a specified number of days. Note how specific I have to be in order to get the code I need.

> Write a single Python function `simulate(days=14, dt=3600.0, cover=False, seed_temp_C=24.0)` that performs a {{< term "forward-euler" "forward Euler" >}} simulation of a well mixed water body under diurnal forcing. Output only valid Python code.
>
> Requirements:
>
> 1. The function must include a NumPy import as `np` at top scope.
> 2. Use these already defined helpers and constants, do not redefine them:
>
>    * helpers: `diurnal(base, amp, hours, peak_shift)`, `c_to_k(C)`, `q_solar(I, A, alpha)`, `q_conv(TwK, TairK, A, h_c)`, `q_rad(TwK, TskyK, A, epsilon)`, `q_evap(TwC, TairC, RH, wind, A, k_evap, beta_wind)`
>    * constants: `RHO` and `CP`
> 3. Geometry and bulk properties inside the function:
>
>    * `L, W, depth = 8.0, 4.0, 1.4`
>    * `A = L * W`
>    * `V = A * depth`
>    * `m = RHO * V`
>    * `C = m * CP`
> 4. Surface exchange coefficients and optical properties:
>
>    * `alpha = 0.85`
>    * `epsilon = 0.95`
>    * `h_c = 8.0`
>    * `k_evap = 2.5e-7`
>    * `beta_wind = 0.1`
>    * If `cover` is True, set `k_evap = 0.3e-7`, `epsilon = 0.75`, `h_c = 6.0`
> 5. Time discretization:
>
>    * `N = int(days * 24)`
>    * `t = np.arange(N) * dt`
>    * `hours = (t / 3600.0) % 24`
> 6. Synthetic diurnal forcing:
>
>    * `T_air_C = diurnal(37.0, 10.0, hours)`
>    * `RH = np.clip(diurnal(0.18, 0.10, hours), 0.05, 0.6)`
>    * `wind = np.clip(diurnal(2.0, 1.5, hours), 0.2, 6.0)`
>    * `I = np.maximum(0.0, 900.0 * np.sin(np.pi * (hours - 6.0) / 12.0))`
>    * `T_sky_K = c_to_k(T_air_C - 15.0)`
> 7. State initialization:
>
>    * `Tw = np.zeros(N)`
>    * `Tw[0] = seed_temp_C`
> 8. Time stepping loop for `i` from 1 to `N-1`:
>
>    * `TwK = c_to_k(Tw[i-1])`
>    * Compute fluxes (Watts):
>      `Qs = q_solar(I[i-1], A, alpha)`
>      `Qc = q_conv(TwK, c_to_k(T_air_C[i-1]), A, h_c)`
>      `Qr = q_rad(TwK, T_sky_K[i-1], A, epsilon)`
>      `Qe = q_evap(Tw[i-1], T_air_C[i-1], RH[i-1], wind[i-1], A, k_evap, beta_wind)`
>    * `Qnet = Qs + Qc + Qr + Qe`
>    * `Tw[i] = Tw[i-1] + (Qnet * dt) / C`
> 9. Return a tuple `(t / 3600.0, Tw)` where time is in hours and temperature in Celsius.
> 10. After the function, include a main guard that runs a 14 day open and covered simulation, prints the number of simulated hours, and prints the final temperatures with one decimal place labeled in Celsius.
> 11. Generously comment the code, but do not exceed 80 chars on any line.

**Python:** Below is the`simulate` function and a compact `main` with a simple run that compares open water with a covered surface.

```python
def simulate(days=14, dt=3600.0, cover=False, seed_temp_C=24.0):
    # Geometry and bulk properties
    L, W, depth = 8.0, 4.0, 1.4  # meters
    A = L * W                    # surface area, m^2
    V = A * depth                # volume, m^3
    m = RHO * V                  # mass of water, kg
    C = m * CP                   # heat capacity, J/K

    # Surface exchange coefficients and optical properties
    alpha = 0.85                 # shortwave absorptivity (fraction)
    epsilon = 0.95               # longwave emissivity (dimensionless)
    h_c = 8.0                    # convective HTC, W/(m^2 K)
    k_evap = 2.5e-7              # bulk evaporation coefficient
    beta_wind = 0.1              # wind amplification factor

    # Apply cover effects: reduce evaporation and radiation, slightly 
    # lower convection
    if cover:
        k_evap = 0.3e-7          # strong reduction in evaporation losses
        epsilon = 0.75           # less longwave emission with cover
        h_c = 6.0                # modestly lower convection

    # Time discretization
    N = int(days * 24)           # number of hourly steps
    t = np.arange(N) * dt        # seconds
    hours = (t / 3600.0) % 24    # local hour of day for diurnal cycles

    # Synthetic diurnal weather forcing

    # hot mean with daily swing
    T_air_C = diurnal(37.0, 10.0, hours)                     

    # very dry afternoons
    RH = np.clip(diurnal(0.18, 0.10, hours), 0.05, 0.6)      

    # m/s
    wind = np.clip(diurnal(2.0, 1.5, hours), 0.2, 6.0)       

    # W/m^2
    I = np.maximum(0.0, 900.0 * np.sin(np.pi * (hours - 6.0) / 12.0))  

    # effective sky temp in Kelvin
    T_sky_K = c_to_k(T_air_C - 15.0)                         

    # State initialization
    Tw = np.zeros(N)
    Tw[0] = seed_temp_C

    # Time integration of energy balance
    for i in range(1, N):
        # Convert current water temp to Kelvin for radiation and 
        # convection calcs
        TwK = c_to_k(Tw[i-1])

        # Compute flux terms (Watts). Sign convention:
        # Positive adds heat to water. Negative removes heat 
        # from water.

        # shortwave absorbed
        Qs = q_solar(I[i-1], A, alpha)

        # convection with air
        Qc = q_conv(TwK, c_to_k(T_air_C[i-1]), A, h_c)

        # longwave net radiation
        Qr = q_rad(TwK, T_sky_K[i-1], A, epsilon)

        # evaporation
        Qe = q_evap(Tw[i-1], T_air_C[i-1], RH[i-1], 
                    wind[i-1], A, k_evap, beta_wind)

        # Net heat rate and temperature update

        # Watts
        Qnet = Qs + Qc + Qr + Qe

        # Celsius update
        Tw[i] = Tw[i-1] + (Qnet * dt) / C

    # Return time in hours and temperature in Celsius
    return t / 3600.0, Tw

if __name__ == "__main__":
    hours_open, Tw_open = simulate(days=14, cover=False)
    hours_cov, Tw_cov = simulate(days=14, cover=True)
    print(f"Simulated {len(hours_open)} hours")
    print(f"Final temperature open: {Tw_open[-1]:.1f} C")
    print(f"Final temperature cover: {Tw_cov[-1]:.1f} C")
```

**Sample Output:** The following simulation shows a ~25 °F (~14 °C) difference between a covered body of water and an uncovered one.

```yaml
Simulated 336 hours
Final temperature open:  61.4 °F (16.3 °C)
Final temperature cover: 86.8 °F (30.5 °C)
```

---

## Making the Results Visual

{{< lightbox
    src="plot.png"
    alt="Line plot showing how germs accumulate on food over 10 seconds with a steep rise at the start and then leveling off"
    label="Open full-size plot"
    caption="Water temperature over 14 days under diurnal forcing for open vs covered conditions (left axis °F, right axis °C). The cover reduces nighttime losses from evaporation and longwave radiation, raising the average temperature and narrowing the daily swing compared with the open surface."
>}}

**Prompt example:** A picture makes the mechanism obvious. Here I ask AI to generate the Python code to plot the two temperature curves on the same axes to compare warming with and without a cover.

> Write a Python function named plot_runs_dual_axis(run_open, run_cov) that uses Matplotlib to plot two temperature time series with Fahrenheit on the left (primary) y-axis and Celsius on the right (secondary) y-axis.
>
> Requirements:
>
> 1. Assume these helpers already exist in scope and must be used: `c_to_f(array_like)` and `f_to_c(array_like)`.
> 2. Inputs are tuples:
>
>    * `run_open = (hours_open, Tw_open_C, label_open)`
>    * `run_cov  = (hours_cov,  Tw_cov_C,  label_cov)`
>      where `hours_*` are hour vectors, and `Tw_*_C` are water temperatures in Celsius.
> 3. Convert the Celsius series to Fahrenheit for plotting on the primary axis.
> 4. Create the plot with `fig, ax = plt.subplots(figsize=(10, 4))`.
> 5. Plot both runs with labels, set x-label to `"Hour"`, primary y-label to `"Water temperature (°F)"`, title to `"Two week warming under diurnal forcing"`, and show a legend.
> 6. Add a **secondary y-axis on the right** in Celsius using `ax.secondary_yaxis('right', functions=(f_to_c, c_to_f))` and set its label to `"Water temperature (°C)"`.
> 7. Call `fig.tight_layout()` and `plt.show()` at the end.
> 8. Include a short docstring explaining inputs and the dual-axis behavior.
> 9. Return only valid Python code with the single function definition and necessary `import matplotlib.pyplot as plt`.

**Python:** Here is a minimal plotting script for the temperature trajectories.

```python
def plot_runs_dual_axis(run_open, run_cov):
    import matplotlib.pyplot as plt

    # Unpack the arguments for open vs. covered run
    hours_open, Tw_open_C, label_open = run_open
    hours_cov,  Tw_cov_C,  label_cov  = run_cov

    # Convert series to °F for the primary y-axis
    Tw_open_F = c_to_f(Tw_open_C)
    Tw_cov_F  = c_to_f(Tw_cov_C)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(hours_open, Tw_open_F, label=label_open)
    ax.plot(hours_cov,  Tw_cov_F,  label=label_cov)

    ax.set_xlabel("Hour")
    ax.set_ylabel("Water temperature (°F)")
    ax.set_title("Two week warming under diurnal forcing")
    ax.legend(loc="best")

    # Secondary y-axis in °C mapped from the primary °F axis
    # functions = (forward, inverse) where forward maps 
    #   primary->secondary
    # Primary is °F, secondary is °C, so use (f_to_c, c_to_f)
    ax2 = ax.secondary_yaxis('right', functions=(f_to_c, c_to_f))
    ax2.set_ylabel("Water temperature (°C)")

    fig.tight_layout()
    plt.show()

```

---

## What We Learned

Water behaves like a massive thermal reservoir. High specific heat spreads warming across many days. Evaporation and longwave radiation remove energy efficiently at night, especially in dry air. A simple cover changes the balance by cutting evaporation and reducing radiative loss, which keeps more of the daytime energy in the water.

---

## Exercises for the Reader

### Beginner Level: Quick Fixes and Calibration

1. Calibrate the evaporation coefficient so the model matches a week of measured temperatures from your own setup.

1. Change depth while holding surface area fixed and observe how geometry changes heating and cooling time scales.

1. Replace the synthetic solar curve with measured irradiance from a local weather source.

### Intermediate Level: Weather and Geometry

1. Add a thin surface layer on top of the bulk water and give it its own temperature for faster daytime response.

2. Drive the model with hourly air temperature, humidity, and wind from a nearby station and compare seasons.

3. Add an approximate ground heat flux term using a constant heat transfer coefficient and soil temperature.

### Advanced Level: Stochasticity and Control

1. Add random gusts in wind and random humidity excursions at night and examine the spread of outcomes.

1. Implement a simple controller that applies a cover based on forecasted night conditions and a target minimum temperature.

1. Couple the model to a small heat pump with a {{< term "coefficient-of-performance" "coefficient of performance" >}} and schedule it when losses are smallest.

---

## Closing Thoughts

Butter and pools may seem like strange lab partners, until you **turn up the heat**. In butter, fat gives way quickly, while the water in margarine plays the long game. Pools behave like margarine on a grand scale, warming slowly, then giving it all back after just a few dry nights. Next time you're poolside on a broiling summer day, you'll know exactly why the water can feel like early spring.

## Try It Yourself

[Download the full code on GitHub](https://github.com/TomArcher/technical-blog-examples/tree/main/thermo-water-energy-balance)