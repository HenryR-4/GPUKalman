#!/usr/bin/env python3

import numpy as np
import sys
from time import perf_counter

if len(sys.argv) < 2:
    print("Usage: kalman <file>")
    print("     File is csv with x,y,z headers")
    sys.exit(1)

def KalmanFilter(F, Q, x0, P0, R, H):
    x = x0
    P = P0
    def run(z):
        nonlocal x, P
        # predict
        x = F.dot(x)
        P = (F.dot(P)).dot(F.T) + Q

        # update
        y = z - H.dot(x)
        K = (P.dot(H.T)).dot(np.linalg.inv((H.dot(P)).dot(H.T) + R))
        x = x + K.dot(y)
        P = (np.eye(len(x)) - K.dot(H)).dot(P)

        return x
    return run

data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=np.float32)
zss = np.hsplit(data, data.shape[1]/3)

# oh boy :) I love big friendly matricies
F = np.array([
    [1, 1, 1/2, 1/6, 0, 0,   0,   0, 0, 0,   0,   0],
    [0, 1,   1, 1/2, 0, 0,   0,   0, 0, 0,   0,   0],
    [0, 0,   1,   1, 0, 0,   0,   0, 0, 0,   0,   0],
    [0, 0,   0,   1, 0, 0,   0,   0, 0, 0,   0,   0],
    [0, 0,   0,   0, 1, 1, 1/2, 1/6, 0, 0,   0,   0],
    [0, 0,   0,   0, 0, 1,   1, 1/2, 0, 0,   0,   0],
    [0, 0,   0,   0, 0, 0,   1,   1, 0, 0,   0,   0],
    [0, 0,   0,   0, 0, 0,   0,   1, 0, 0,   0,   0],
    [0, 0,   0,   0, 0, 0,   0,   0, 1, 1, 1/2, 1/6],
    [0, 0,   0,   0, 0, 0,   0,   0, 0, 1,   1, 1/2],
    [0, 0,   0,   0, 0, 0,   0,   0, 0, 0,   1,   1],
    [0, 0,   0,   0, 0, 0,   0,   0, 0, 0,   0,   1]
])

Q = np.array([
    [1/36, 1/12, 1/6, 1/6,    0,    0,   0,   0,    0,    0,   0,   0],
    [1/12,  1/4, 1/2, 1/2,    0,    0,   0,   0,    0,    0,   0,   0],
    [ 1/6,  1/2,   1,   1,    0,    0,   0,   0,    0,    0,   0,   0],
    [ 1/6,  1/2,   1,   1,    0,    0,   0,   0,    0,    0,   0,   0],
    [   0,    0,   0,   0, 1/36, 1/12, 1/6, 1/6,    0,    0,   0,   0],
    [   0,    0,   0,   0, 1/12,  1/4, 1/2, 1/2,    0,    0,   0,   0],
    [   0,    0,   0,   0,  1/6,  1/2,   1,   1,    0,    0,   0,   0],
    [   0,    0,   0,   0,  1/6,  1/2,   1,   1,    0,    0,   0,   0],
    [   0,    0,   0,   0,    0,    0,   0,   0, 1/36, 1/12, 1/6, 1/6],
    [   0,    0,   0,   0,    0,    0,   0,   0, 1/12,  1/4, 1/2, 1/2],
    [   0,    0,   0,   0,    0,    0,   0,   0,  1/6,  1/2,   1,   1],
    [   0,    0,   0,   0,    0,    0,   0,   0,  1/6,  1/2,   1,   1]
])

P0 = np.array([
    [2000000,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
    [      0, 2000000,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
    [      0,       0, 2000000,       0,       0,       0,       0,       0,       0,       0,       0,       0],
    [      0,       0,       0, 2000000,       0,       0,       0,       0,       0,       0,       0,       0],
    [      0,       0,       0,       0, 2000000,       0,       0,       0,       0,       0,       0,       0],
    [      0,       0,       0,       0,       0, 2000000,       0,       0,       0,       0,       0,       0],
    [      0,       0,       0,       0,       0,       0, 2000000,       0,       0,       0,       0,       0],
    [      0,       0,       0,       0,       0,       0,       0, 2000000,       0,       0,       0,       0],
    [      0,       0,       0,       0,       0,       0,       0,       0, 2000000,       0,       0,       0],
    [      0,       0,       0,       0,       0,       0,       0,       0,       0, 2000000,       0,       0],
    [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0, 2000000,       0],
    [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, 2000000]
])

H = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
])

res = []

start = perf_counter()
for zs in zss:
    f = KalmanFilter(
        F=F,
        Q=Q,
        x0=np.array([0,0,0,0,0,0,0,0,0,0,0,0]),
        P0=P0,
        R=np.array([[10000,0,0],[0,10000,0],[0,0,10000]]),
        H=H
    )
    xs = np.empty(0)
    for z in zs:
        x = f(z)
        xs = np.append(xs, np.array([x[0], x[4], x[8]]))
    res.append(xs.reshape(-1,3))
end = perf_counter()
print(end-start, file=sys.stderr) # put to stderr so you can see performance and still output result

# for i in range(len(res)):
    # res[i] = res[i].reshape(-1, 3)

for i in range(len(res[0])):
    if i == 0:
        for j in range(len(res)):
            e = "\n" if (j==len(res)-1) else ","
            print(f'x{j},y{j},z{j}', end=e)
    for j in range(len(res)):
        e = "\n" if (j==len(res)-1) else ","
        print(f'{res[j][i][0]},{res[j][i][1]},{res[j][i][2]}', end=e)

# for j in range(len(res[0])):
#     for i,xs in enumerate(res):
#         if j == 0:
#             if i == len(res)-1:
#                 print(f'x{i},y{i},z{i}')
#             else:
#                 print(f'x{i},y{i},z{i},')
# 
#         xs = xs.reshape(-1,3)
#         x = xs[j]
#         if i == len(res)-1:
#             print(f'{x[0]},{x[1]},{x[2]}')
#         else:
#             print(f'{x[0]},{x[1]},{x[2]},')
