class Dynamics:
    def __init__(self):
        self.state = []
        self.Ts = []
        self.x0 = []

        self.A = []
        self.B = []
        self.C = []
        self.D = []

        self.sys_cont = []
        self.sys_disc = []

    def simulate(self, u, mode=None, x0=None):
        return 0

    def step(self,u,mode=None):
        return 0
