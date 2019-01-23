# PROBLEMS WITH CONVERGENCE AND INITIAL CONDITIONS DEPENDENCE
# REFORMULATE PROBLEM OUTSIDE EQUILIBRIUM

from numpy import linspace, vstack, zeros, array, maximum
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

el = 15 # Lenght of habitat
n_mesh = 500 # Number of points, habitat mesh
n_el = 500 # Number of points, habitat array
w_0 = 0.01   # Cost per unit of fishing effort
w_1 = 0.001  # Marginal cost per unit effort
x = linspace(-el/2,el/2,n_el) # Habitat array
dx = el/len(x)

# y = [u,v,lambda_u,lambda_v]

def f_so(y,gamma):
    fstar = ( y[0] - w_0 - y[3]*( gamma*y[0]*y[0] + y[0] ) )/w_1/2
    return maximum(fstar,zeros(fstar.size))

def function_i(x,y,gamma):
    u, v, lambda_u, lambda_v = y
    fso = f_so(y,gamma)
    return vstack(( - v,
                   u*(1-fso) - (1 + gamma*fso)*u*u,
                  -lambda_v*(1 - 2*u*(1 + gamma*fso) - fso),
                  lambda_u))

def bc_i(ya,yb,gamma):
    return array([ ya[0], yb[0], ya[3], yb[3] ])

def effort_stock_plot(u,f,gamma):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    ax1.set_title('$\gamma = %.1f$' % gamma)
    ax1.plot(x,u, 
         color = 'black', linestyle = '-', linewidth = 1)
    ax1.set_ylabel('Stock density')
    ax1.set_xlabel('Location x')
    ax1.set_ylim(min(u),5*max(u))
    ax2.plot(x,f, 
         color = 'black', linestyle = '--', linewidth = 1)
    ax2.set_ylabel('Effort density')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_ticks_position('right')
    ax2.set_ylim(min(f),max(f)*1.1)
    plt.show()
    return 

def biomass(): # Something wrong?
    gammavary = linspace(0,15,100)
    b = []
    x_mesh = linspace(-el/2,el/2,n_mesh) 
    y_solver = zeros((4,x_mesh.size)) 
    y_solver[0] = 0.5 
    y_solver[3] = -0.4
    for gamma in gammavary:
        sol = solve_bvp(lambda x,y: function_i(x,y,gamma = gamma),
                lambda ya, yb: bc_i(ya,yb, gamma = gamma),x_mesh,y_solver)
        y = sol.sol(x) 
        u = y[0]
        b.append(dx*sum(u))
    return array(b)