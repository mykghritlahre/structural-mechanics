"""
Pratt Truss

AE 238: Aerospace Structural Mechanics
Spring 2023
"""

import numpy as np
import plane_truss_lib as tlib

L = 1.0
r32 = np.sqrt(3)/2

joints = np.array([
    [0.0, 0.0],
    [0.5*L, 0.0],
    [L, 0.0],
    [1.5*L, 0.0],
    [2*L, 0.0],
    [2.5*L, 0.0],
    [3*L, 0.0],
    [3.5*L, 0.0],
    [4*L, 0.0],
    [3.5*L, r32*L],
    [3*L, r32*L],
    [2.5*L, r32*L],
    [2*L, r32*L],
    [1.5*L, r32*L],
    [L, r32*L],
    [0.5*L, r32*L]
])

members = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [7, 9],
    [6, 9],
    [6, 10],
    [5, 10],
    [5, 11],
    [4, 11],
    [4, 12],
    [4, 13],
    [3, 13],
    [3, 14],
    [2, 14],
    [2, 15],
    [1, 15],
    [0, 15],
    [15, 14],
    [14, 13],
    [13, 12],
    [12, 11],
    [11, 10],
    [10, 9]

], dtype=int)

E = 2.1e11
A = 1e-4

truss = tlib.PlaneTruss(joints, members, E, A)

constraints = [
    [0, 0, 0.0],
    [0, 1, 0.0],
    [8, 1, 0.0]
]
truss.apply_constraints(constraints)

P = -1e4
loads = [
    [1, 1, P],
    [2, 1, P],
    [3, 1, P],
    [4, 1, P],
    [5, 1, P],
    [6, 1, P],
    [7, 1, P]
]
truss.apply_loads(loads)

truss.solve()

print('Reactions: ')
print(truss.reactions)

print('Internal forces: ')
print(truss.member_forces)

truss.plot(deformed=True, mag=5)
