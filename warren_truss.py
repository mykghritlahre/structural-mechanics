"""
Warren Truss


AE 238: Aerospace Structural Mechanics
Spring 2023
"""

import numpy as np
import plane_truss_lib as tlib

L = 1.0
r32 = np.sqrt(3)/2

joints = np.array([
    [0.0, 0.0],
    [L, 0.0],
    [2*L, 0.0],
    [3*L, 0.0],
    [4*L, 0.0],
    [5*L, 0.0],
    [4.5*L, r32*L],
    [3.5*L, r32*L],
    [2.5*L, r32*L],
    [1.5*L, r32*L],
    [0.5*L, r32*L]
])

members = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [4, 6],
    [4, 7],
    [3, 7],
    [3, 8],
    [2, 8],
    [2, 9],
    [1, 9],
    [1, 10],
    [0, 10],
    [10, 9],
    [9, 8],
    [8, 7],
    [7, 6]
], dtype=int)

E = 2.1e11
A = 1e-2

truss = tlib.PlaneTruss(joints, members, E, A)

constraints = [
    [0, 0, 0.0],
    [0, 1, 0.0],
    [5, 0, 0.0],
    [5, 1, 0.0]
]
truss.apply_constraints(constraints)

P = 1e4
loads = [
    [1, 1, P],
    [2, 1, P],
    [3, 1, P],
    [4, 1, P]
]
truss.apply_loads(loads)

truss.solve()

print('Reactions: ')
print(truss.reactions)

print('Internal forces: ')
print(truss.member_forces)

truss.plot(deformed=True)
