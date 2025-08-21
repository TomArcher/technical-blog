+++
title = "From Ice Shows to Algorithms: Cracking the Truck-Packing Puzzle"
date = '2025-08-20T09:00:00-07:00'
draft = true
categories = ["Programming", "Optimization"]
tags = [
"Python",
"3D Bin Packing",
"Algorithms",
"AI",
"Modeling",
"Heuristics",
"Matplotlib"
]
listThumb = "three-d-packing.png"
+++

<figure style="float: right; margin: 0 20px 10px 20px; width: 250px; text-align: center;"> <img src="./three-d-packing.png" alt="3D Bin Packing" width="250" style="display: block; margin: 0 auto;"> <figcaption style="font-size: 0.9em; color: #555; margin-top: 5px;"> <em>How do you fit it all in?</em> </figcaption> </figure>

---

My first full-time programming job was for **Holiday on Ice**, an international ice show. While I focused mainly on back office systems such as accounting, itinerary, and box office reporting, I knew that one of the biggest technical challenges faced by the show's crew was efficiently loading trucks for the next city.  

One day, the controller asked me if I could code a system that took, as input, the trucks’ **3D dimensions** and the **3D dimensions (and weight)** of every object to be packed. Back in the **Turbo Pascal** era, exploring 3D packing was painful. Today, with **Python** and **AI-assisted scaffolding**, it’s surprisingly approachable.  

This same problem shows up anywhere people need to fit things into finite 3D spaces—think **concert tours** (road cases into trailers), **e-commerce fulfillment** (cartons into pallets/containers), **furniture moves**, **air cargo** (ULDs), **shipping & logistics** (LTL/FTL), and **warehouse slotting**.  

---

> *"Given the dimensions of a truck and a list of containers (with their dimensions and weight), in what order, position, and orientation should you pack the truck?"*  

---

## The Thought Experiment

Imagine a truck interior sized **H × W × D** (height, width, depth) and a set of boxes with **height, width, depth, and weight**. The challenge:  

- **Maximize space usage** while avoiding overlap.  
- **Respect the truck bounds** (no sticking out).  
- Optionally **consider weight** and stacking rules.  

Real-world simple? Not really. Computationally, it’s a classic **3D bin-packing** problem. On the scale of computational complexity, this problem is extremely difficult (**NP-hard**), meaning there’s no known fast way to always find the perfect solution. Instead, we rely on heuristics, which are clever shortcuts that deliver good (though not guaranteed optimal) results quickly.  

It’s like trying to figure out every possible way to load groceries into your car trunk. You could test every arrangement, but that would take forever, so instead you look for smart tricks to get a good result quickly. Tricks such as loading heavy items first, stacking boxes neatly, and filling gaps with small bags.  

# Modeling the Packing Problem

To simulate the problem, we make the following simplifications:

**Assumptions:**

- The truck is a perfect cuboid with fixed dimensions.
- Each box is also a cuboid with (height, width, depth, weight).
- Boxes may only be rotated in **axis-aligned orientations** (6 total).  
- Free space is tracked as rectangular subspaces (split after each placement).
- Goal: maximize volume utilization, while optionally considering stacking rules.

## Using AI to Scaffold the Code

I start by prompting an AI model to sketch out a basic packing algorithm:

```prompt
Write a Python script that, given the dimensions of a truck 
(height, width, depth) and a dictionary of objects with 
height, width, depth, and weight, produces a packing order. 
The script should output 3D positions and orientations for 
each object.
```

The model produces a working draft. As with most AI-generated code, it needs refinement so blocks don’t float or extend beyond the truck’s limits. These edits include better variable and function naming, clearer placement logic, and fixing layout issues. Still, this first generation saves me hours of scaffolding and gives me a strong foundation to improve upon. What follows is the refined version.

```python
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Dimensions:
    height: float
    width: float
    depth: float

@dataclass
class Placement:
    name: str
    size: Dimensions
    weight: float
    position: Tuple[float, float, float]

def pack_truck(truck: Dimensions, items: Dict[str, Dict]) -> Tuple[List[Placement], List[str], List[str]]:
    placements = []
    skipped = []
    notes = []

    # Track current "cursor" position
    x, y, z = 0.0, 0.0, 0.0
    current_layer_height = 0.0
    row_depth = 0.0

    for name, item in items.items():
        h, w, d, wt = item["height"], item["width"], item["depth"], item["weight"]

        # If it doesn't fit in the current row, move along width
        if x + w > truck.width:
            x = 0
            z += row_depth
            row_depth = 0

        # If it doesn't fit in current layer, move up
        if z + d > truck.depth:
            z = 0
            y += current_layer_height
            current_layer_height = 0

        # If it doesn't fit in truck, skip
        if y + h > truck.height:
            skipped.append(name)
            notes.append(f"{name} skipped: too tall for remaining truck height.")
            continue

        placements.append(
            Placement(
                name=name,
                size=Dimensions(h, w, d),
                weight=wt,
                position=(x, y, z),
            )
        )

        # Update cursor positions
        x += w
        row_depth = max(row_depth, d)
        current_layer_height = max(current_layer_height, h)

    return placements, skipped, notes
```

## Simulation: Packing a Truck

Once the function exists, I test it with a sample truck definition and a dictionary of sample items:

```prompt
Write example code that defines a truck, provides a dictionary 
of items with height, width, depth, and weight, and prints 
the resulting packing plan.
```

The model produces usable code that I edit for clarity and readability.

```python
if __name__ == "__main__":
    truck = Dimensions(height=100.0, width=100.0, depth=240.0)

    items_input = {
        "crateA": {"height": 50, "width": 40, "depth": 60, "weight": 80},
        "crateB": {"height": 30, "width": 30, "depth": 30, "weight": 40},
        "crateC": {"height": 20, "width": 60, "depth": 40, "weight": 50},
        "crateD": {"height": 80, "width": 40, "depth": 50, "weight": 90},
        "crateE": {"height": 10, "width": 90, "depth": 30, "weight": 20},
        "crateF": {"height": 40, "width": 40, "depth": 40, "weight": 60},
    }

    placements, skipped, notes = pack_truck(truck, items_input)

    print("== PACK PLAN ==")
    for i, p in enumerate(placements, 1):
        x, y, z = p.position
        print(
            f"{i:02d}. {p.name}: pos=({x:.1f},{y:.1f},{z:.1f}) "
            f"size(HxWxD)=({p.size.height:.1f}x{p.size.width:.1f}x{p.size.depth:.1f}) "
            f"weight={p.weight:.1f}"
        )

    if skipped:
        print("\nSkipped:", skipped)
    if notes:
        print("\nNotes:")
        for n in notes:
            print("-", n)
```

## Making the Results Visual

It’s one thing to print coordinates; it’s another to see them. AI can scaffold a plotting function too.

```prompt
Write a Python function that takes a truck’s dimensions and 
the placements list, and plots the packed truck in 3D using 
Matplotlib. Each box should have a unique color and label.
```

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

def plot_truck_packing(truck: Dimensions, placements: List[Placement]):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Truck wireframe
    ax.plot([0, truck.width, truck.width, 0, 0],
            [0, 0, truck.depth, truck.depth, 0],
            [0, 0, 0, 0, 0], "k-")
    ax.plot([0, truck.width, truck.width, 0, 0],
            [0, 0, truck.depth, truck.depth, 0],
            [truck.height]*5, "k-")

    for p in placements:
        x, y, z = p.position
        h, w, d = p.size.height, p.size.width, p.size.depth

        # Vertices of cuboid
        vertices = [
            [x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],
            [x, y, z+d], [x+w, y, z+d], [x+w, y+h, z+d], [x, y+h, z+d],
        ]
        faces = [
            [vertices[j] for j in [0,1,2,3]],
            [vertices[j] for j in [4,5,6,7]],
            [vertices[j] for j in [0,1,5,4]],
            [vertices[j] for j in [2,3,7,6]],
            [vertices[j] for j in [1,2,6,5]],
            [vertices[j] for j in [0,3,7,4]],
        ]
        color = [random.random(), random.random(), random.random()]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors="k", alpha=0.7))
        ax.text(x + w/2, y + h/2, z + d/2, p.name, color="k")

    ax.set_xlabel("Width (x)")
    ax.set_ylabel("Height (y)")
    ax.set_zlabel("Depth (z)")
    ax.set_xlim(0, truck.width)
    ax.set_ylim(0, truck.height)
    ax.set_zlim(0, truck.depth)
    plt.show()
```

If you are following along, remember to call the XYZ function at the end of the main function.

```python
if __name__ == "__main__":

    ...

    # Visualize (optional)
    try:
        plot_truck_packing(truck, placements)
    except Exception as e:
        print("\n(Visualization skipped; matplotlib may not be installed.)")
        print(e)
```

## The Verdict

What did we learn?

Prompt engineering matters: clear, constrained prompts help AI generate scaffolds closer to usable.

Algorithmic complexity: this is an NP-hard problem, so we don’t expect perfect solutions—just good heuristics.

Heuristics like greedy placement + guillotine splits can deliver solid results in practical time, even for large item sets.

Visualization bridges the gap between abstract coordinates and human intuition.

## Exercises for the Reader

###  Beginner Level: Quick Fixes & Calibration

1. Add weight limits: don’t allow heavy crates to stack on fragile ones.

1. Track utilization percentage after each placement.

### Intermediate Level: Geometry & Body Modeling

1. Allow full box rotations (beyond axis-aligned).

1. Experiment with different scoring heuristics (e.g., minimize height first).

### Advanced Level: Environment & Stochasticity

1. Implement simulated annealing or genetic algorithms to search better packings.

1. Add real-world rules like axle load balancing or fragile stacking.

1. Scale up to thousands of boxes with Monte Carlo heuristics.

## Last words

Packing trucks may seem like grunt work, but it’s really a microcosm of **optimization problems**: finite resources, constraints, tradeoffs. The thrill of solving them, especially with today’s AI scaffolding, is the same thrill that first drew me to programming back at Holiday on Ice.

## Try It Yourself

[Download the full code on GitHub](https://github.com/TomArcher/technical-blog-examples/tree/main/python/three-d-packing)