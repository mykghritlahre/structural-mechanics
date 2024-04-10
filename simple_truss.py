"""
Simple truss example.
"""

import numpy as np
import plane_truss_lib as tlib

L = 1.0

joints = np.array([
    [0.0, 0.0],
    [L, 0.0],
    [0.0, L]
])

members = np.array([
    [0, 1],
    [1, 2],
    [0, 2]
], dtype=int)

E = 2.1e11
A = 1e-4

truss = tlib.PlaneTruss(joints, members, E, A)

truss.plot(deformed=False)

constraints = [
    [0, 0, 0.0],
    [0, 1, 0.0],
    [2, 0, 0.0],
    [2, 1, 0.0]
]
truss.apply_constraints(constraints)

P = 1e4
Q = 1e4
loads = [
    [1, 0, P],
    [1, 1, Q]
]
truss.apply_loads(loads)

truss.solve()
