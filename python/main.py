import numpy as np
import mpmath
import matplotlib.pyplot as plt
from matplotlib import cm
import os

_N = 0
def _lerch(z,k,beta):
    global _N
    _N+=1
    if not _N%50:
        print(f'calculating phi({z=},{k=},{beta=})')
    try:
        return mpmath.lerchphi(z,k,beta)
    except:
        return np.nan
def subplot_indeces(N):
    cols = max(int(np.ceil(N/1.9)), 1)
    rows = int(N//cols)
    if N%cols: rows +=1
    pos = range(1,N+1)
    return rows,cols,pos


def posneg(A):
    neg = np.zeros_like(A)
    pos = np.zeros_like(A)
    neg[A>0] = np.nan
    pos[A<0] = np.nan
    return pos, neg


lerch = np.frompyfunc(_lerch,3,1)

z_from, z_to, z_by = 0, 1, 101
b_from, b_to, b_by = 0, 2, 101
# b_from, b_to, b_by = 0, 10, 101

filename = f"{z_from} {z_to} {z_by} {b_from} {b_to} {b_by}.npz"

z = np.linspace(z_from,z_to,z_by)[1:-1]
b = np.linspace(b_from,b_to,b_by)[1:]
Z,B = np.meshgrid(z,b)
phi_0 = 1/(1-Z)

if os.path.exists(filename):
    print('loading data')
    with np.load(filename) as f:
        phi_1=f['phi_1']
        phi_2=f['phi_2']
        phi_3=f['phi_3']
    print('data loaded')
else:
    print('calculating data')
    phi_1 = lerch(Z,1,B).astype(np.float64)
    print('phi_1 done')
    phi_2 = lerch(Z,2,B).astype(np.float64)
    print('phi_2 done')
    phi_3 = lerch(Z,3,B).astype(np.float64)
    print('phi_3 done')
    np.savez(filename, phi_1=phi_1, phi_2=phi_2, phi_3=phi_3)
    print('data saved')

ZtoB = np.pow(Z,B)
lnZ = np.log(Z)
K = ((B*phi_1-phi_0)*lnZ-B*phi_2)
Omega=B**2*(1-Z)*ZtoB
dyda = K*Omega
Omega_z=Omega/Z*(B-Z*phi_0)
Omega_b=Omega*(2/B+lnZ)
K_z=-B/Z*K-phi_0*(1/Z+phi_0*lnZ)
K_b=K/B+phi_0*lnZ/B-B*lnZ*phi_2+2*B*phi_3
dzda = B*Z*(lnZ+K/phi_0)

DyDa = -B**2*(Omega_b*K+Omega*K_b)+dzda*(Omega_z*K+Omega*K_z)
C2=-B*phi_0+B**2*phi_1+Z*B**2*phi_1**2
C0=B*(Z*B*phi_2**2-4*phi_2+2*B*phi_3)
C1=-2*phi_0+4*B*phi_1-2*B**2*phi_2*(1+Z*phi_1)
DyDaL=-Omega*B*(C2*lnZ**2+C1*lnZ+C0)
Dpos, Dneg= posneg(DyDa)

Psi = C2*lnZ**2+C1*lnZ+C0
Psipos,Psineg = posneg(Psi)
Psiclip=np.clip(Psi,-10,10)


Q2=B**2/Z*(phi_0-B*phi_1)-B*phi_0**2+2*B**2*phi_0*phi_1+B**2*(1-2*B)*phi_1**2
Q1=2*B/Z*(phi_0-2*B*phi_1+B**2*phi_2)-2*phi_0**2-2*B**2*phi_2*(phi_0+(1-2*B)*phi_1)
Q0=2/Z*(-phi_0+2*B**2*phi_2-B**3*phi_3)+B**2*(1-2*B)*phi_2**2
Psiz=Q2*lnZ**2+Q1*lnZ+Q0
Psizclip=np.clip(Psiz,-5,10)
Psizpos,Psizneg = posneg(Psizclip)

a = (phi_0-B*phi_1)
b = (phi_0-2*B*phi_1+B**2*phi_2)
c = (-phi_0+2*B**2*phi_2-B**2*phi_3)
disc = Q1**2-4*Q0*Q2

r1=phi_1/phi_0
r2=phi_2/phi_1
r3=phi_3/phi_2

Q1a=(-B+2*B*B*r1+B*B*(1-2*B)*r1*r1)*phi_0**2

grids = {
    'dy/da': [dyda, Dpos, Dneg, DyDa],
    # 'original': [DyDa],
    # 'DyDa': [DyDa, Dpos, Dneg],
    # 'Psi': [Psiclip, Psipos, Psineg],
    # 'Psi_z': [Psizclip,Psizpos,Psizneg],
    # 'Psi_z_reg': [Psizclip*(1-Z)*ZtoB,Psizpos,Psizneg],
    '$\\Delta$': [np.clip(disc,-10,10)],
    'Q': [np.clip(Q1a,-10,10)]
    }



fig = plt.figure(figsize=(12,8))
rows, cols, pos = subplot_indeces(len(grids))
# rows, cols = 2,2
for i, (name, grids) in enumerate(grids.items()):
    ax = fig.add_subplot(rows, cols, pos[i], projection='3d')
    ax.plot_surface(Z,B,np.zeros_like(Z),color='black', alpha=0.1)
    colors = ['green', 'red']
    for grid, color in zip(grids, ['blue', 'green', 'red','yellow']):
        ax.plot_surface(Z,B,grid,color=color,rstride=1,cstride=1,linewidth=0, alpha=0.5)
    ax.set_xlabel('Z')
    ax.set_ylabel('B')
    ax.set_zlabel(name)
    ax.set_title(name)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()