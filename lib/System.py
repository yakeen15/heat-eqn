import numpy as np

class BoundaryCondition():
    def apply(self, system):
        raise NotImplementedError("This method should be overriden by subclasses")
    
class Dirichlet(BoundaryCondition):
    def __init__(self, bd='left', q = lambda t, y: 1):
        self.bd = bd # The boundary this BC applies to
        self.q = q # The function that represents temperature at the boundary
    def apply(self, system):
        x = np.linspace(0, system.Lx, system.Nx)
        y = np.linspace(0, system.Ly, system.Ny)
        for nt in range(1, system.u.shape[2]):
            if self.bd == 'left' :
                temp = self.q(nt*system.dt, y)
                system.u[0,:,nt] = system.u[0,:,nt] + temp
            elif self.bd == 'right' :
                temp = self.q(nt*system.dt, y)
                system.u[system.Nx-1,:,nt] = system.u[system.Nx-1,:,nt] + temp
            elif self.bd == 'bottom' :
                temp = self.q(nt*system.dt, x)
                system.u[:,0,nt] = system.u[:,0,nt] + temp
            elif self.bd == 'top' :
                temp = self.q(nt*system.dt, x)
                system.u[:,system.Ny-1,nt] = system.u[:,system.Ny-1,nt] + temp


class RectHeat:
    def __init__(self, Lx=1.0, Ly=1.0, Nx=10, Ny=10, T=1.0, Nt=2, state="transient", source=lambda t,x,y: 0):
        # Initialize the dimensions and the solution array
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx/(Nx-1)
        self.dy = Ly/(Ny-1)
        self.state = state
        self.T = T 
        self.Nt = Nt
        self.dt = T/(Nt-1)
        self.alpha = 1.0 # Thermal conduction coefficient
        self.u = np.zeros((Nx, Ny, Nt))  # Temperature distribution u(x,y,t)
        self.icfunc = lambda x, y: 0
        self.source = source
        self.bc = []
        self.eqn = np.zeros((Nx*Ny, Nx*Ny))
       
    def __str__(self):
        # Return a descriptive string of the problem
        return (f"2D Rectangular Domain heat problem \nGrid: {self.Lx} x {self.Ly} \nTime: {self.T}, {self.state}\nSpace Points: {self.Nx}, {self.Ny}\n"
    f"Time Points: {self.Nt}\nIC: {self.icfunc}")


    
    def IC(self, func):
        # Apply Initical conditions to the first time slice t = 0
        self.icfunc = func
        x = 0
        y = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.u[i,j,0] = self.icfunc(x,y)
                y = y + self.dy
            y = 0
            x = x + self.dx

    def BC(self):
        for bc in self.bc:
            bc.apply(self)
        # Smoothing the left and bottom boundaries
        if isinstance(self.bc[0], Dirichlet) and isinstance(self.bc[2], Dirichlet):
            self.u[0,0,1:] = 0.5*self.u[0,0,1:]
        # Smoothing the left and top boundaries
        if isinstance(self.bc[0], Dirichlet) and isinstance(self.bc[3], Dirichlet):
            self.u[0,self.Ny-1,1:] = 0.5*self.u[0,self.Ny-1,1:]
        # Smoothing the right and bottom boundaries
        if isinstance(self.bc[1], Dirichlet) and isinstance(self.bc[2], Dirichlet):
            self.u[self.Nx-1,0,1:] = 0.5*self.u[self.Nx-1,0,1:]
        # Smoothing the right and top boundaries
        if isinstance(self.bc[1], Dirichlet) and isinstance(self.bc[3], Dirichlet):
            self.u[self.Nx-1,self.Ny-1,1:] = 0.5*self.u[self.Nx-1,self.Ny-1,1:]
        # For Neumann intersections : use the modified stencil
        # For Mixed intersections : cry
    
    def eqnForm(self, nt):
        # Take the self.eqn matrix (Nx*Ny) and populate it according to the stencil
        a = self.alpha*self.dt/self.dx**2
        b = self.alpha*self.dt/self.dy**2
        for k in range(self.Nx*self.Ny):
            pass
        temp = np.zeros((self.Nx,self.Ny))
        
