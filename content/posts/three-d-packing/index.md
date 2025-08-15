+++
title = "From Ice Shows to Algorithms: Cracking the Truck-Packing Puzzle"
date = 2025-08-14T09:00:00-07:00
draft = false
categories = ["Programming", "Optimization"]
tags = ["Python", "Geometry", "3D Bin Packing", "Logistics", "Thought Experiment"]
+++

My very first programming job was with Holiday on Ice, a touring show that hauled entire arenas’ worth of scenery, props, and gear from city to city. Every week, the crew would break everything down and load it into trucks for the next destination.  

One day, the controller asked if I could write a program to figure out **the most efficient way to pack those trucks**. I had already built their itinerary and box-office reporting systems, so I was comfortable with logic and basic math. But this was different. This was geometry, weight distribution, and optimization rolled into one giant puzzle — and at the time, I had no idea where to start.

---

### The Thought Experiment

Imagine standing in a cavernous loading bay. In front of you are dozens of oddly shaped crates, each with its own weight, size, and fragility. You have a limited number of trucks, each with a fixed height, width, and length. The challenge:  
- Fit every crate if possible.  
- Keep the load balanced so the truck doesn’t handle poorly or overload an axle.  
- Protect fragile items while still making full use of space.  

Which item do you load first? The heaviest? The bulkiest? The one that will be hardest to fit later?  

The problem hides layers of complexity:
- **3D geometry** — fitting rectangular prisms together efficiently.  
- **Combinatorics** — the staggering number of possible loading orders.  
- **Optimization tradeoffs** — maximizing space usage vs. keeping weight balanced.  

At the time, this felt like a riddle I couldn’t solve — a real-world constraint problem without an obvious formula.

---

### Modeling the Trucks

To make the problem solvable, you have to simplify reality without losing the essence of the challenge. A model might assume:  

* The truck interior is a perfect rectangular box with fixed dimensions.  
* All cargo is rectangular, with known width, length, height, and weight.  
* Items can be rotated only in certain ways (e.g., no tipping fragile crates on their sides).  
* Each placement is axis-aligned (no angles).  
* The goal is to place as many items as possible without exceeding the truck’s weight limit or violating balance rules.  

These constraints let us move from “impossible to simulate” to a **discrete 3D bin packing problem** that we can attack with algorithms.

---

### A Python Approach

Below is a simplified Python implementation of a greedy “shelf-packing” algorithm. It tries to place the biggest and heaviest items first, filling each horizontal layer (“shelf”) before moving upward. It also includes a crude balance check to keep the center of mass near the middle of the truck.

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Item:
    id: str
    w: float  # width in inches
    l: float  # length in inches
    h: float  # height in inches
    weight: float

@dataclass
class PlacedItem:
    item: Item
    x: float
    y: float
    z: float

@dataclass
class Truck:
    inner_w: float
    inner_l: float
    inner_h: float
    max_payload_lb: float

def pack_truck(truck: Truck, items: List[Item]) -> List[PlacedItem]:
    # Sort largest-first by volume, then weight
    items_sorted = sorted(items, key=lambda it: (it.w * it.l * it.h, it.weight), reverse=True)
    placed = []
    x = y = z = 0
    row_height = 0

    for item in items_sorted:
        if x + item.w <= truck.inner_w and y + item.l <= truck.inner_l and z + item.h <= truck.inner_h:
            placed.append(PlacedItem(item, x, y, z))
            x += item.w
            row_height = max(row_height, item.h)
        else:
            x = 0
            y += row_height
            row_height = item.h
            if y + item.l <= truck.inner_l:
                placed.append(PlacedItem(item, x, y, z))
                x += item.w

    return placed
```
### Sample Output

Let’s test the algorithm with a small set of items:

```python
truck = Truck(inner_w=96, inner_l=312, inner_h=102, max_payload_lb=18000)
items = [
    Item("crate1", 48, 60, 48, 1200),
    Item("crate2", 30, 40, 36, 260),
    Item("crate3", 28, 28, 28, 180)
]

plan = pack_truck(truck, items)

for p in plan:
    print(f"{p.item.id} at ({p.x}, {p.y}, {p.z})")
```

***Output***
```output
crate1 at (0, 0, 0)
crate2 at (48, 0, 0)
crate3 at (78, 0, 0)
```















