# Solving optimal harvest problem
# System of 4 PDEs
from numpy import linspace, zeros, concatenate, maximum, array, asarray
from numpy.random import rand
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from time import time
import matplotlib.pyplot as plt

# Numerical constants and global variables
#gamma_0 = 0
#gamma_1 = 10
el = 15 # Lenght of habitat
w_0 = 0.01   # Cost per unit of fishing effort
w_1 = 0.001
p = 1
case = 3
N = 90
N_plot = 1000
lr = el*0.8
tau_ns = 0.40
x = linspace(-el/2,el/2,N) #Habitat for PDE
z = linspace(-el/2,el/2,N_plot) # Habitat for plots
h = el/N
dz = el/N_plot
tol = 1e-1 # Tolerance

# Evaluating convergence
def dist(u,v):
    return sum( (u-v)**2 )

# Divided second order finite differences
def div_diff2(g): 
    return (g[:-2] + g[2:] - 2*g[1:-1])/h**2

# Fishing effort tax
def tau(case,u,lt,gamma_0,gamma_1):
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
def f_so(u,lt,gamma_0,gamma_1):
    fstar = ( p*u - w_0 - tau(case,u,lt,gamma_0,gamma_1) )/w_1
    return maximum(fstar,zeros(fstar.size))

# PDEs -> ODEs + boundary conditions
def odesfnc(y,t,gamma_0,gamma_1):
    u_res = y[:N-2]
    lt_res = y[N-2:2*N-4]
    u = concatenate([[0],u_res,[0]])
    lt = concatenate([[0],lt_res,[0]])
    f_res = f_so(u_res,lt_res,gamma_0,gamma_1)
    f = f_so(u,lt,gamma_0,gamma_1)
    y3 = y[2*N-4:3*N-4]
    y4 = y[3*N-4:4*N-4]
    du = (y3[1:-1] - y4[1:-1]*u_res)*u_res - f_res*u_res + div_diff2(u)
    dlambda = p*f_res + lt_res*(y3[1:-1] - 2*u_res*y4[1:-1] - f_res) + div_diff2(lt)
    dy3 = 1/2*(1 - gamma_0*f - y3)
    dy4 = 1/2*(1 + gamma_1*f - y4)
    return concatenate([du,dlambda,dy3,dy4])

# Initial conditions randomized
def icfnc():
    u0 = 0.5*(1+(2*rand(x.size-2)-1))
    lt0 = 0.5*(1+(2*rand(x.size-2)-1))
    y30 = zeros(x.size)
    y40 = zeros(x.size)
    y30[:] = 0.01
    y40[:] = 0.01
    return concatenate([u0,lt0,y30,y40])

# Solver
def solve_model(t_end,nt,gamma_0,gamma_1):
    tspan = linspace(0,t_end,nt)
    print('Choice of gamma_0 = %.2f ' % gamma_0)
    print('Choice of gamma_1 = %.2f ' % gamma_1)
    print('Running model until t = %.0f s' % t_end) 
    tic = time()
    function = lambda y,t: odesfnc(y,t,gamma_0 = gamma_0,gamma_1 = gamma_1)
    sol = odeint(function,icfnc(),tspan)
    print('Solution in %.0f s' % (time()-tic))
    return sol

def final_sol(sol):
    u = concatenate([[0],sol[-1,:N-2],[0]])
    lambda_v = concatenate([[0],sol[-1,N-2:2*N-4],[0]])
    y3 = sol[-1,:2*N-4:3*N-4]
    y4 = sol[-1,3*N-4:4*N-4]
    fu = interp1d(x,u,kind = 'cubic')
    fl = interp1d(x,lambda_v,kind = 'cubic')
    #f3 = interp1d(x,y3,kind = 'cubic') # Don't need them
    #f4 = interp1d(x,y4,kind = 'cubic')
    return [fu(z),fl(z)] #,f3(z),yf4(z)]

# Finding biomass
def biomass(gamma_0,gamma_1,u): # u needs to be z.size
    return sum(u)*dz

def biomass_analysis_1(gamma_1_in,gamma_1_fin,ngamma):
    gamma = linspace(gamma_1_in,gamma_1_fin,ngamma)
    t_end = 5000 # hope it's enough for convergence
    nt = 600000
    bio = []
    for g in gamma:
        print('-Solving for gamma_1 = %.1f' % g)
        sol_t = solve_model(t_end,nt,0,g)
        u = final_sol(sol_t)[0]
        bio.append(biomass(0,g,u))
    return [gamma,bio]

# Finding needed time for convergence (too slow)
def solve_t(y0,t0,t_end,nt):
    tspan = linspace(t0,t_end,nt)
    return odeint(odesfnc,y0,tspan)[-1:][0]

def it_solver(time_max):
    tic = time()
    y0 = icfnc()
    t_minus = 0
    t_plus = 1000
    v = solve_t(y0,t_minus,t_plus,70000)
    while( dist(y0,v) > tol and (time()-tic) < time_max):
        y0 = v
        t_minus += 500
        t_plus += 500
        print("Solving until %.0 s" % t_plus)
        v = solve_t(y0,t_minus,t_plus,100)
    if ((time()-tic) > time_max):
        print("Convergence not found in %.0f s" % time_max)
    else:
        print("Solution found in %.1f s" % (tic - time()) )
    return [v,t_plus,t_plus*100,dist(v,y0)]
    
# Plots
def effort_stock_plot(u,f,gamma,scale_stock):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    ax1.set_title('$\gamma = %.1f$' % gamma)
    ax1.plot(z,u, label = 'Stock density',
         color = 'black', linestyle = '-', linewidth = 1)
    ax1.set_ylabel('Stock density')
    ax1.set_xlabel('Location x')
    ax1.set_ylim(min(u),scale_stock*max(u))
    ax2.plot(z,f, label = 'Fishing effort',
         color = 'black', linestyle = '--', linewidth = 1)
    ax2.set_ylabel('Effort density')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_ticks_position('right')
    ax2.set_ylim(min(f),max(f)*1.1)
    plt.show()
    return 
