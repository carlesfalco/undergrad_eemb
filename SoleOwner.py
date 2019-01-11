# Solving optimal harvest problem
# System of 4 PDEs
from numpy import linspace, zeros, concatenate, maximum, array, asarray
from numpy.random import uniform
from scipy.integrate import odeint
from time import time
import matplotlib.pyplot as plt

# Numerical constants and global variables
gamma_0 = 0
gamma_1 = 10
el = 15 # Lenght of habitat
w_0 = 0.01   # Cost per unit of fishing effort
w_1 = 0.001
p = 1
case = 3
N = 120
lr = el*0.8
tau_ns = 0.40
x = linspace(-el/2,el/2,N) #Habitat
h = el/N

# Divided second order finite differences
def div_diff2(g): 
    return (g[:-2] + g[2:] - 2*g[1:-1])/h**2

# Fishing effort tax
def tau(case,u,lt):
    if case == 1: # nonspatial effort tax
        tau = zeros(u.size)
        tau[:] = tau_ns
        return tau
    elif case == 2: # tax and reserve
        tau = zeros(min(x.size,u.size))
        y = x
        if u.size < x.size:
            y = x[1:-1]
        tau += (abs(y) < lr ) * tau_ns
        return tau
    elif case == 3:
        tau = 1/2*(p*u - w_0 + lt*(gamma_0*u + gamma_1*u*u + u))
        return tau
                 
# Fishing effort
def f_so(u,lt):
    fstar = ( p*u - w_0 - tau(case,u,lt) )/w_1
    return maximum(fstar,zeros(fstar.size))

# PDEs -> ODEs + boundary conditions
def odesfnc(y,t):
    u_res = y[:N-2]
    lt_res = y[N-2:2*N-4]
    u = concatenate([[0],u_res,[0]])
    lt = concatenate([[0],lt_res,[0]])
    f_res = f_so(u_res,lt_res)
    f = f_so(u,lt)
    y3 = y[2*N-4:3*N-4]
    y4 = y[3*N-4:4*N-4]
    du = (y3[1:-1] - y4[1:-1]*u_res)*u_res - f_res*u_res + div_diff2(u)
    dlambda = p*f_res + lt_res*(y3[1:-1] - 2*u_res*y4[1:-1] - f_res) + div_diff2(lt)
    dy3 = 1/2*(1 - gamma_0*f - y3)
    dy4 = 1/2*(1 + gamma_1*f - y4)
    return concatenate([du,dlambda,dy3,dy4])

# Initial conditions randomized
def icfnc():
    u0 = zeros(x.size-2)
    lt0 = zeros(x.size-2)
    y30 = zeros(x.size)
    y40 = zeros(x.size)
    u0[:] = 0.5*(1+(2*uniform(0,1)-1))
    lt0[:] = 0.5*(1+(2*uniform(0,1)-1))
    y30[:] = 0.01
    y40[:] = 0.01
    return concatenate([u0,lt0,y30,y40])

# Solver
def solve_model(t_end,nt):
    tic = time()
    print('Choice of gamma_1 = %.2f ' % gamma_1)
    print('Running model until t = %f s' % t_end) 
    tspan = linspace(0,t_end,nt)
    sol = odeint(odesfnc,icfnc(),tspan)
    print(' Solution in %.2f s' % (time()-tic))
    u = zeros(x.size)
    lambda_v = zeros(x.size)
    y3 = zeros(x.size)
    y4 = zeros(x.size)
    u = concatenate([[0],sol[-1,:N-2],[0]])
    lambda_v = concatenate([[0],sol[-1,N-2:2*N-4],[0]])
    y3 = sol[-1,:2*N-4:3*N-4]
    y4 = sol[-1,3*N-4:4*N-4]
    return [u,lambda_v,y3,y4]
   
# Plots
def effort_stock_plot(u,f,gamma,scale_stock):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    ax1.set_title('$\gamma = %.1f$' % gamma)
    ax1.plot(x,u, 
         color = 'black', linestyle = '-', linewidth = 1)
    ax1.set_ylabel('Stock density')
    ax1.set_xlabel('Location x')
    ax1.set_ylim(min(u),scale_stock*max(u))
    ax2.plot(x,f, 
         color = 'black', linestyle = '--', linewidth = 1)
    ax2.set_ylabel('Effort density')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_ticks_position('right')
    ax2.set_ylim(min(f),max(f)*1.1)
    plt.show()
    return 