#!/usr/bin/env python3

import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: kalman <file>")
    print("     Files are csv with x0,y0,x1,y1,... headers")
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
        print(f'P = {P}')
        print(f'H = {H}')
        print(f'P H^T = {P.dot(H.T)}')
        print(f'HPH^t + R = {(H.dot(P)).dot(H.T) + R}')
        print(f'(HPH^t + R)^-1 = {np.linalg.inv((H.dot(P)).dot(H.T) + R)}')
        print(f'K = {K}')
        x = x + K.dot(y)
        P = (np.eye(len(x)) - K.dot(H)).dot(P)

        return x
    return run

zs1 = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)

f = KalmanFilter(
    F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]),
    Q=np.array([[25,0,0,0],[0,25,0,0],[0,0,49,0],[0,0,0,49]]),
    x0=np.array([0,0,0,0]),
    P0=np.array([[160000,0,0,0],[0,160000,0,0],[0,0,40000,0],[0,0,0,40000]]),
    R=np.array([[10000,0],[0,10000]]),
    H=np.array([[1,0,0,0],[0,1,0,0]]),
)

print('x,y')
x = f(zs1[0])
for z in zs1:
    x = f(z)
    print(f'{x[0]},{x[1]}')
