#!env python3

import numpy as np
from sympy.functions.special.delta_functions import Heaviside
from sympy import Min, Symbol, refine
from sympy.utilities.lambdify import lambdify
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def run(k1=0.3, k2=0.3,k3=0.3,k4=0.6,A1=150,A2=25,A3=10,K=200,P3=1.2,P2=1, plot=False):
    A = Symbol('A', positive = True)
    L21 = L22 = L23 = L24 = 0.25
    L31 = L32 = L33 = 0.3
    L34 = 0.1

    a33 = Min(k4 * L34 * A3 * (k1*L31+k2*L32+k3*L33), K*A)
    a23 = Min(k4 * L24 * A2 * (k1*L21+k2*L22+k3*L23), K*A-a33)
    a13 = Min(A*A1, K*A-a33-a23);

    a32 = Min((1-k4) * L34 * A3 * (k1*L31+k2*L32+k3*L33), K*(1-A))
    a22 = Min((1-k4) * L24 * A2 * (k1*L21+k2*L22+k3*L23), K*(1-A)-a32)
    a12 = Min((1-A)*A1, K*(1-A)-a32-a22);

    X = (a13+a23+a33) * P3 + (a12 + a22 + a32)*P2
    X = -X

    Xl = lambdify((A,), X, modules=['numpy','sympy'])
    Xj = [X.diff(A)]
    Xlj = [lambdify((A,), f, modules=['numpy','sympy']) for f in Xj]

    def jac(zz):
        return np.array([jf(zz[0]) for jf in Xlj])

    def Xf(zz):
        return Xl(zz[0])

    m = 0
    guess = 0
    for j in np.arange(0.001,0.999,0.001):
        v = Xf((j, ))
        if(v < m):
            m = v
            guess = j

    bnd = ((0.001, 0.999),)
    result = minimize(Xf,np.array([j]), bounds=bnd, method='SLSQP', jac=jac)
    if plot:
        plt.plot(np.arange(0.001,0.999,0.001), [-Xl(i) for i in np.arange(0.001,0.999,0.001)])
        plt.savefig('r.png')
        plt.clf()
        print(result)
    return result.x[0]

run(plot=True, k4=0.2)

# With different K
print('Generating: k')
kx = np.arange(0, 300, 10)
plt.plot(
        kx, [run(P2=1, P3=1.01, K=k) for k in kx], 'y-',
        kx, [run(P2=1, P3=1.05, K=k) for k in kx], 'g-',
        kx, [run(P2=1, P3=1.1, K=k) for k in kx], 'b-',
        kx, [run(P2=1, P3=1.2, K=k) for k in kx], 'r-',
        )
plt.savefig('k.png')
plt.clf()

# With different P3
print('Generating: P3')
px = np.arange(1,1.6, 0.03)
plt.plot(
        px, [run(P3=p, K=150) for p in px], 'g-',
        px, [run(P3=p, K=180) for p in px], 'b-',
        px, [run(P3=p, K=210) for p in px], 'r-',
        )
plt.savefig('p.png')
plt.clf()

# With different A1
print('Generating: A1')
ax = np.arange(50,300, 10)
plt.plot(
        ax, [run(A1=a, P3=1.05) for a in ax], 'g-',
        ax, [run(A1=a, P3=1.1) for a in ax], 'b-',
        ax, [run(A1=a, P3=1.2) for a in ax], 'r-',
        )
plt.savefig('a.png')
plt.clf()

# With different k4
print('Generating: k4')
kx = np.arange(0,1,0.05)
plt.plot(
        kx, [run(k4=k, A2=10) for k in kx], 'g-',
        kx, [run(k4=k, A2=25) for k in kx], 'b-',
        kx, [run(k4=k, A2=50) for k in kx], 'r-',
        )
plt.savefig('k4.png')
plt.clf()
