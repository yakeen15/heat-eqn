import numpy as np

class BoundaryCondition():
    def apply(self, u):
        raise NotImplementedError("This method should be overriden by subclasses")

class RectHeat:
    def __init__(self, Lx=1.0, Ly=1.0, Nx=10, Ny=10, T=1.0, Nt=1):
        # Initialize the dimensions and the solution array
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx/(Nx-1)
        self.dy = Ly/(Ny-1)
        self.T = T 
        self.Nt = Nt
        self.alpha = 1.0 # Thermal conduction coefficient
        self.u = np.zeros((Nx, Ny, Nt))  # Temperature distribution u(x,y,t)
        self.icfunc = lambda x, y: 0
        self.bc = []
        self.eqn = np.zeros((Nx*Ny, Nx*Ny))
       
    def __str__(self):
        # Return a descriptive string of the problem
        return (f"2D Rectangular Domain heat problem \nGrid: {self.Lx} x {self.Ly} \nTime: {self.T}\nSpace Points: {self.Nx}, {self.Ny}\n"
    f"Time Points: {self.Nt}\nIC: {self.icfunc}")
    
    def IC(self, func):
        # Apply Initical conditions to the first time slice t = 0
        self.icfunc = func
        x= 0
        y= 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.u[i,j,0] = self.icfunc(x,y)
                y = y + self.dy
            y = 0
            x = x + self.dx
