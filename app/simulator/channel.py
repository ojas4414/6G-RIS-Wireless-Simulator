import numpy as np


class WirelessEnvironment:
    def __init__(self, Nt=4, M=16, velocity=5,
                 wavelength=0.1, Ts=0.01):

        self.Nt = Nt
        self.M = M
        self.v = velocity
        self.lambda_ = wavelength
        self.Ts = Ts

        # Doppler-based temporal correlation
        fD = self.v / self.lambda_
        self.alpha = np.exp(-2 * np.pi * fD * self.Ts)

        # Initialize channels
        self.H_BU = self._init_channel((M, Nt))
        self.g_U = self._init_channel((M, 1))

        # Initialize RIS phases
        self.theta = np.random.uniform(0, 2*np.pi, M)

    def _init_channel(self, shape):
        return (np.random.randn(*shape) +
                1j*np.random.randn(*shape)) / np.sqrt(2)

    def _evolve_channel(self, H):
        noise = (np.random.randn(*H.shape) +
                 1j*np.random.randn(*H.shape)) / np.sqrt(2)
        return self.alpha * H + np.sqrt(1 - self.alpha**2) * noise

    def step(self):
        # Evolve channels
        self.H_BU = self._evolve_channel(self.H_BU)
        self.g_U = self._evolve_channel(self.g_U)

        # RIS phase matrix
        Theta = np.diag(np.exp(1j * self.theta))

        # Effective cascaded channel
        h_eff = self.g_U.T @ Theta @ self.H_BU

        return h_eff.flatten()  # shape (Nt,)
