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
        self.bc = [BoundaryCondition() for i in range(4)] # Empty BC, to be replaced by actual BCs
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
    
    def numunk(self):
        # Finds the number of unknowns
        # In a dirichlet boundary, all the values at the boundaries are known functions
        Dx = 0
        Dy = 0
        if isinstance(self.bc[0], Dirichlet):
            Dx = Dx + 1
        if isinstance(self.bc[1], Dirichlet):
            Dx = Dx + 1
        if isinstance(self.bc[2], Dirichlet):
            Dy = Dy + 1
        if isinstance(self.bc[3], Dirichlet):
            Dy = Dy + 1
        return Dx, Dy

    def geteqn(self, nx, ny, nt=1):
        # This will return a (Nx-Dx,Ny-Dy) sized vector of coefficients associated with the equation for (nx, ny) point
        # The point (nx, ny) must be an interior (that is, nx > 0 and ny > 0)
        a = self.alpha*self.dt/self.dx**2
        b = self.alpha*self.dt/self.dy**2
        temp = np.zeros((self.Nx,self.Ny))
        temp[nx, ny] = 1+2*a+2*b
        temp[nx, ny-1], temp[nx, ny+1] = -a, -a 
        temp[nx-1, ny], temp[nx+1, ny] = -b, -b
        if isinstance(self.bc[0], Dirichlet):
            temp = np.delete(temp, 0, axis=0)
        if isinstance(self.bc[1], Dirichlet):
            temp = np.delete(temp, -1, axis=0)
        if isinstance(self.bc[2], Dirichlet):
            temp = np.delete(temp, 0, axis=1)
        if isinstance(self.bc[3], Dirichlet):
            temp = np.delete(temp, -1, axis=1)
        return np.reshape(temp, np.shape(temp)[0]*np.shape(temp)[1])

    def getconst(self, nx, ny, nt):
        a = self.alpha*self.dt/self.dx**2
        b = self.alpha*self.dt/self.dy**2
        cost = self.dt*self.source(nt*self.dt, nx*self.dx, ny*self.dy) + self.u[nx, ny, nt-1]
        if nx-1==0 and isinstance(self.bc[0], Dirichlet):
            cost = cost + b*self.bc[0].q(nt*self.dt, ny*self.dy)
        elif nx+1==self.Nx-1 and isinstance(self.bc[1], Dirichlet):
            cost = cost + b*self.bc[1].q(nt*self.dt, ny*self.dy)
        if ny-1==0 and isinstance(self.bc[2], Dirichlet):
            cost = cost + a*self.bc[2].q(nt*self.dt, nx*self.dx)
        elif ny+1==self.Ny-1 and isinstance(self.bc[3], Dirichlet):
            cost = cost + a*self.bc[3].q(nt*self.dt, nx*self.dx)
        return cost
        
