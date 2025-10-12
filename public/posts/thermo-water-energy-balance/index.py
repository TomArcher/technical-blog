+++
title = "Modeling Heat Capacity and Evaporation with Python: Why Water Warms Slowly but Cools Fast"
date = "2025-08-18T09:00:00-07:00"
draft = true
categories = ["Mathematics", "Physics", "Python"]
tags = ["Thermodynamics", "Heat Capacity", "Evaporation", "Energy Balance", "Python Simulation"]
+++

Every summer, it feels like a small miracle when the pool finally warms up enough to swim. In Nevada, where the air temperature can sit above 100°F (38°C) for weeks, you’d expect the water to keep pace. Yet, somehow, it takes forever to warm, and only a few cool nights can undo all that progress.

The same phenomenon shows up in a stick of butter. Butter melts quickly, while margarine stays stubbornly firm even under the same heat. That’s not coincidence—it’s thermodynamics.

The butter versus margarine comparison is a staple example in nutrition science. It shows how the proportions of fat, water, and solids affect how much energy it takes to change temperature. Butter, with more fat and less water, heats up and melts quickly. Margarine, full of water and unsaturated oils, absorbs more energy before softening because water’s specific heat is much higher.

Your pool works the same way, just scaled up thousands of times. Its massive water content means it has enormous heat capacity—warming takes a long time because every degree requires tremendous energy. Cooling happens faster, though, because evaporation and night radiation pull energy out far more efficiently than the sun can replace it.

---

> *"A pool in the desert and a stick of margarine in the kitchen both tell the same story: water resists change."*

---

In this post, we will turn that intuition into a simple Python model that explains exactly why water warms so slowly but cools so fast.

## The Thought Experiment

Imagine a shallow pool sitting in the desert sun. During the day, it absorbs energy from sunlight, and at night it loses energy to the air and sky. The surprising part is not that both happen, but that the rates are so uneven.

The pool’s heat capacity acts like a huge thermal battery—it takes time to charge. The cooling process, on the other hand, is driven by evaporation. Each gram of water that evaporates carries away about 2.45 kilojoules of energy, and with enough dry air and a light breeze, those losses pile up quickly.

The same logic explains why a pat of margarine melts slower than butter. Water is a powerful heat sink, and evaporation is a powerful energy thief. Put them together, and you have a recipe for long warmups and quick cool-downs.

## Modeling the Problem

We will treat the water as a well mixed control volume with a single temperature. The surface exchanges heat with the air through convection, with the sky through longwave radiation, and with the sun through shortwave absorption. Evaporation removes energy proportional to the rate at which vapor escapes.

**Assumptions:**

The water is well mixed with no stratification.

Conduction through walls and floor is small compared with surface fluxes.

Shortwave absorption is a constant fraction of incident irradiance.

Longwave exchange follows the Stefan Boltzmann law with an effective sky temperature.

Evaporation follows a bulk aerodynamic relationship that depends on humidity gradient and wind.

Weather varies smoothly over the day and repeats during the simulation window.

The energy balance is
C dT dt equals Q solar plus Q convective plus Q radiative plus Q evaporative
with
C equals m c p
Q solar equals alpha times A times I of t
Q convective equals h c times A times left bracket T air of t minus T of t right bracket
Q radiative equals epsilon times sigma times A times left bracket T sky of t to the fourth minus T of t to the fourth right bracket
Q evaporative equals negative L v times m dot evaporative

---

## Using AI to Scaffold the Code

When you ask an AI assistant to provide scaffolding, be specific about inputs, outputs, and units. Request pure functions for each physical term, expressed in SI units, and constants defined once. Ask for no plotting and no input or output inside the helpers so that you can test them easily.

**Prompt example:**

Write Python helper functions to model a water surface energy balance.
Include functions for solar gain, convection, radiation, and evaporation.
Use SI units consistently and return power in watts for each function.

Here is a clean set of helpers that follow those requirements. You will paste this as plain code in your editor.

import numpy as np

SIGMA = 5.670374419e-8
LV = 2.45e6
CP = 4186.0
RHO = 1000.0

def toK(C):
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

---

## Simulation: Diurnal Heating and Night Cooling

Now we integrate forward in time with a small time step. We define simple diurnal weather patterns and a switch that approximates a cover by reducing evaporation and radiation and slightly lowering convection.

**Prompt example:**

Write a main function simulate that integrates temperature with forward Euler. The function should accept days, time step in seconds, and a cover flag. Return time in hours and water temperature in Celsius.

Below is a compact main and a simple run that compares open water with a covered surface.

def diurnal(base, amp, hours, peak_shift=7.0):
return base + amp * np.sin(2*np.pi * (hours - peak_shift) / 24.0)

def simulate(days=14, dt=3600.0, cover=False, seed_temp_C=24.0):
L, W, depth = 8.0, 4.0, 1.4
A = L * W
V = A * depth
m = RHO * V
C = m * CP

alpha = 0.85
epsilon = 0.95
h_c = 8.0
k_evap = 2.5e-7
beta_wind = 0.1

if cover:
    k_evap = 0.3e-7
    epsilon = 0.75
    h_c = 6.0

N = int(days * 24)
t = np.arange(N) * dt
hours = (t / 3600.0) % 24

T_air_C = diurnal(37.0, 10.0, hours)
RH = np.clip(diurnal(0.18, 0.10, hours), 0.05, 0.6)
wind = np.clip(diurnal(2.0, 1.5, hours), 0.2, 6.0)
I = np.maximum(0.0, 900.0 * np.sin(np.pi * (hours - 6.0) / 12.0))
T_sky_K = toK(T_air_C - 15.0)

Tw = np.zeros(N)
Tw[0] = seed_temp_C
for i in range(1, N):
    TwK = toK(Tw[i-1])
    Qs = q_solar(I[i-1], A, alpha)
    Qc = q_conv(TwK, toK(T_air_C[i-1]), A, h_c)
    Qr = q_rad(TwK, T_sky_K[i-1], A, epsilon)
    Qe = q_evap(Tw[i-1], T_air_C[i-1], RH[i-1], wind[i-1], A, k_evap, beta_wind)
    Qnet = Qs + Qc + Qr + Qe
    Tw[i] = Tw[i-1] + (Qnet * dt) / C

return t / 3600.0, Tw


if __name__ == "__main__":
hours_open, Tw_open = simulate(days=14, cover=False)
hours_cov, Tw_cov = simulate(days=14, cover=True)
print(f"Simulated {len(hours_open)} hours")
print(f"Final temperature open: {Tw_open[-1]:.1f} C")
print(f"Final temperature cover: {Tw_cov[-1]:.1f} C")

**Sample Output:**

Simulated 336 hours
Final temperature open: 29.1 C
Final temperature cover: 31.7 C

## Making the Results Visual

A picture makes the mechanism obvious. Plot the two temperature curves on the same axes to compare warming with and without a cover. Then create a second figure for a single day that shows the four heat flux terms so you can see when each one dominates.

**Prompt example:**

Create two Matplotlib figures. First, plot water temperature for open and covered conditions across the full simulation. Second, for hours zero through twenty four, plot the four heat flux components as lines or a stacked area chart.

Here is a minimal plotting script for the temperature trajectories.

import matplotlib.pyplot as plt

hours_open, Tw_open = simulate(days=14, cover=False)
hours_cov, Tw_cov = simulate(days=14, cover=True)

plt.figure(figsize=(10,4))
plt.plot(hours_open, Tw_open, label="Open surface")
plt.plot(hours_cov, Tw_cov, label="With cover")
plt.xlabel("Hour")
plt.ylabel("Water temperature (C)")
plt.title("Two week warming under diurnal forcing")
plt.legend()
plt.tight_layout()
plt.show()

---

## What We Learned

Water behaves like a massive thermal reservoir. High specific heat spreads warming across many days. Evaporation and longwave radiation remove energy efficiently at night, especially in dry air. A simple cover changes the balance by cutting evaporation and reducing radiative loss, which keeps more of the daytime energy in the water.

## Exercises for the Reader

### Beginner Level: Quick Fixes and Calibration

Calibrate the evaporation coefficient so the model matches a week of measured temperatures from your own setup.

Change depth while holding surface area fixed and observe how geometry changes heating and cooling time scales.

Replace the synthetic solar curve with measured irradiance from a local weather source.

### Intermediate Level: Weather and Geometry

Add a thin surface layer on top of the bulk water and give it its own temperature for faster daytime response.

Drive the model with hourly air temperature, humidity, and wind from a nearby station and compare seasons.

Add an approximate ground heat flux term using a constant heat transfer coefficient and soil temperature.

### Advanced Level: Stochasticity and Control

Add random gusts in wind and random humidity excursions at night and examine the spread of outcomes.

Implement a simple controller that applies a cover based on forecasted night conditions and a target minimum temperature.

Couple the model to a small heat pump with a coefficient of performance and schedule it when losses are smallest.

---

## Closing Thoughts

The same physics that makes butter soften faster than a water rich spread also explains why water warms slowly and cools quickly. Specific heat stretches warming across time while evaporation and night sky cooling can pull heat out quickly. Once you see the balance in code the paradox turns into a ledger.

## Try It Yourself

[Download the full code on GitHub](https://github.com/TomArcher/technical-blog-examples/tree/main/python/water-heat-capacity)