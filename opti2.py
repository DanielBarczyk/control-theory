import casadi as ca
import matplotlib.pyplot as plt
from math import fabs

G = 9.81

class Opti2Drone:
    def __init__(self, mass = 1.0, max_force = 10.0, N = 120):
        self.mass = mass
        self.max_force = max_force
        self.N = N

        self.opti = ca.Opti()

        self.x = self.opti.variable(6, self.N + 1)
        self.u = self.opti.variable(3, self.N)
        self.t1 = self.opti.variable()
        self.opti.minimize(self.t1)


    def f(self, x, u):
        return ca.vertcat(
            x[3],
            x[4],
            x[5],
            u[0] / self.mass,
            u[1] / self.mass,
            u[2] / self.mass - G
        )


    def eval(self, time = 60, goal = [5.0, 5.0, 5.0, 0.0, 0.0, 0.0]):
        self.time = time
        
        dt = self.t1 / self.N
        for i in range(self.N):
            k1 = self.f(self.x[:,i], self.u[:,i])
            k2 = self.f(self.x[:,i] + dt / 2 * k1, self.u[:,i])
            k3 = self.f(self.x[:,i] + dt / 2 * k2, self.u[:,i])
            k4 = self.f(self.x[:,i] + dt * k3, self.u[:,i])
            x_next = self.x[:,i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(self.x[:,i+1] == x_next)

        self.opti.subject_to(self.u[0, :] * self.u[0, :] + self.u[1, :] * self.u[1, :] + self.u[2, :] * self.u[2, :] <= self.max_force * self.max_force)
        self.opti.subject_to(self.x[:, 0] == 0)
        self.opti.subject_to(self.x[:, self.N] == goal)

        self.opti.subject_to(self.t1 >= 0)

        self.opti.set_initial(self.t1, 1)

        self.opti.solver('ipopt')
        sol = self.opti.solve()

        self.sol = sol


    def plot_xyz(self):
        t1 = self.sol.value(self.t1) 
        tgrid = [t1 / self.N * k for k in range(self.N + 1)]

        x = self.sol.value(self.x)

        plt.figure(1)
        plt.clf()
        plt.plot(tgrid, x[0, :], '--')
        plt.plot(tgrid, x[1, :], '-.')
        plt.plot(tgrid, x[2, :], '--')
        plt.xlabel('t (s)')
        plt.legend(['x','y','z'])
        plt.title('Pozycja')
        plt.grid()
        plt.show()


    def plot_v(self):
        t1 = self.sol.value(self.t1) 
        tgrid = [t1 / self.N * k for k in range(self.N + 1)]

        x = self.sol.value(self.x)

        plt.figure(1)
        plt.clf()
        plt.plot(tgrid, x[3, :], '--')
        plt.plot(tgrid, x[4, :], '-.')
        plt.plot(tgrid, x[5, :], '--')
        plt.xlabel('t (s)')
        plt.title('Prędkość')
        plt.legend(['vx','vy','vz'])
        plt.grid()
        plt.show()


    def plot_u(self):
        t1 = self.sol.value(self.t1) 
        tgrid = [t1 / self.N * k for k in range(self.N)]

        x = self.sol.value(self.u)

        plt.figure(1)
        plt.clf()
        plt.plot(tgrid, x[0, :], '--')
        plt.plot(tgrid, x[1, :], '-.')
        plt.plot(tgrid, x[2, :], '--')
        plt.xlabel('t (s)')
        plt.title('Sterowanie')
        plt.legend(['F_x','F_y','F_z'])
        plt.grid()
        plt.show()
