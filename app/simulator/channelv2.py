"""
channelv2.py  -  UAV-assisted RIS channel simulator
====================================================
Drop-in replacement for the original channelv2.
The alias at the bottom keeps all existing imports working:
    from app.simulator.channelv2 import channel_version2
"""

import numpy as np


# ── utility: ULA steering vector ──────────────────────────────────────────────
def steering_vector(n_elements, angle_rad, d_over_lambda=0.5):
    n   = np.arange(n_elements)
    phi = 2 * np.pi * d_over_lambda * np.sin(angle_rad)
    return (1 / np.sqrt(n_elements)) * np.exp(1j * n * phi)


# ── UAV trajectory ─────────────────────────────────────────────────────────────
class UAVTrajectory:
    def __init__(self, pattern="circle", center=None, radius=50.0,
                 altitude=100.0, speed=5.0, Ts=0.01):
        self.pattern  = pattern
        self.center   = center if center is not None else np.array([300.0, 300.0, altitude])
        self.radius   = radius
        self.altitude = altitude
        self.speed    = speed
        self.Ts       = Ts
        self.angle    = 0.0

        if pattern == "circle":
            self.pos = self.center + np.array([radius, 0.0, 0.0])
            self.d_angle = (speed / radius) * Ts
        elif pattern == "linear":
            self.pos = np.array([100.0, 300.0, altitude], dtype=float)
        else:
            self.pos = self.center.copy().astype(float)

        self.vel = self._compute_vel()

    def _compute_vel(self):
        if self.pattern == "circle":
            return np.array([
                -self.speed * np.sin(self.angle),
                 self.speed * np.cos(self.angle),
                 0.0
            ])
        elif self.pattern == "linear":
            return np.array([self.speed, 0.0, 0.0])
        else:
            return np.random.randn(3) * self.speed / np.sqrt(3)

    def step(self):
        if self.pattern == "circle":
            self.angle += self.d_angle
            self.pos = self.center + np.array([
                self.radius * np.cos(self.angle),
                self.radius * np.sin(self.angle),
                0.0
            ])
            self.vel = self._compute_vel()
        elif self.pattern == "linear":
            self.pos = self.pos + self.vel * self.Ts
            for i in range(3):
                if self.pos[i] < 0 or self.pos[i] > 600:
                    self.vel[i] *= -1
        else:
            accel    = np.random.randn(3) * 0.5
            self.vel = self.vel + accel * self.Ts
            spd = np.linalg.norm(self.vel)
            if spd > self.speed * 2:
                self.vel *= self.speed * 2 / spd
            self.pos = self.pos + self.vel * self.Ts
            self.pos[2] = self.altitude
        return self.pos.copy(), self.vel.copy()


# ── main channel class ────────────────────────────────────────────────────────
class channel_version3:
    """
    UAV-assisted RIS downlink channel.
    BS (Nt antennas) -> UAV/RIS (M elements) -> K users (single antenna each).

    step() returns: (h_eff, contributions, uav_state, aocsi)
    - h_eff        : (K, Nt)  complex  effective channel per user
    - contributions: (M,)     float    per-tile beam strength for visualisation
    - uav_state    : dict     UAV position, velocity, Doppler alpha
    - aocsi        : (K,)     int      Age-of-CSI per user
    """

    def __init__(self,
                 nt=4, m=64, n_users=2, n_urllc=1,
                 wavelength=0.1, Ts=0.01,
                 rician_K=3.0, phase_bits=2,
                 uav_pattern="circle", uav_speed=5.0,
                 uav_altitude=100.0, pl_exp=2.2,
                 # v2 backward-compat: velocity maps to uav_speed
                 velocity=None, **kwargs):

        if velocity is not None:
            uav_speed = float(velocity)

        self.nt      = nt
        self.m       = m
        self.n_users = n_users
        self.n_urllc = n_urllc
        self.lambda_ = wavelength
        self.Ts      = Ts
        self.rician_K = rician_K
        self.phase_bits = phase_bits
        self.pl_exp  = pl_exp
        self.t       = 0

        # fixed node positions (3D metres)
        self.bs_pos = np.array([0.0, 0.0, 30.0])
        np.random.seed(42)
        self.ue_pos = np.array([
            [500.0 + 80 * np.cos(2*np.pi*k/max(n_users,1)),
             300.0 + 80 * np.sin(2*np.pi*k/max(n_users,1)),
             1.5]
            for k in range(n_users)
        ])

        # UAV trajectory
        self.uav = UAVTrajectory(
            pattern=uav_pattern, center=np.array([300.0, 300.0, uav_altitude]),
            radius=80.0, altitude=uav_altitude, speed=uav_speed, Ts=Ts
        )
        self.uav_pos, self.uav_vel = self.uav.pos.copy(), self.uav.vel.copy()

        # phase codebook
        if phase_bits > 0:
            self.codebook = np.linspace(0, 2*np.pi, 2**phase_bits, endpoint=False)
        else:
            self.codebook = None

        # initial channels
        self.H_BU = self._init_rician((m, nt), self.bs_pos, self.uav_pos)
        self.g_kU = np.stack([
            self._init_rician((m,), self.uav_pos, self.ue_pos[k])
            for k in range(n_users)
        ], axis=0)   # (K, M)

        # initial theta
        self.theta = np.zeros(m)

        # Age-of-CSI
        self.aocsi            = np.zeros(n_users, dtype=int)
        self.last_pilot_slot  = np.zeros(n_users, dtype=int)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _los(self, shape, pos_tx, pos_rx):
        diff = pos_rx - pos_tx
        dist = np.linalg.norm(diff) + 1e-9
        az   = np.arctan2(diff[1], diff[0])
        el   = np.arcsin(np.clip(diff[2] / dist, -1, 1))
        if len(shape) == 2:
            return np.outer(steering_vector(shape[0], az),
                            steering_vector(shape[1], el).conj())
        return steering_vector(shape[0], az)

    def _init_rician(self, shape, pos_tx, pos_rx):
        K = self.rician_K
        c = np.sqrt(K / (K + 1))
        s = np.sqrt(1 / (K + 1))
        nlos = (np.random.randn(*shape) + 1j*np.random.randn(*shape)) / np.sqrt(2)
        return c * self._los(shape, pos_tx, pos_rx) + s * nlos

    def _doppler_alpha(self, pos_tx, pos_rx, vel):
        diff    = pos_rx - pos_tx
        dist    = np.linalg.norm(diff) + 1e-9
        v_proj  = np.dot(vel, diff / dist)
        fd      = abs(v_proj) / self.lambda_
        return float(np.clip(np.exp(-2 * np.pi * fd * self.Ts), 0.01, 0.9999))

    def _evolve(self, H_prev, shape, pos_tx, pos_rx, alpha):
        K = self.rician_K
        c = np.sqrt(K / (K + 1))
        s = np.sqrt(1 / (K + 1))
        innov  = (np.random.randn(*shape) + 1j*np.random.randn(*shape)) / np.sqrt(2)
        H_los  = c * self._los(shape, pos_tx, pos_rx)
        H_nlos = alpha * (H_prev - H_los) + np.sqrt(1 - alpha**2) * s * innov
        return H_los + H_nlos

    def _path_loss(self, a, b):
        return 1.0 / (max(np.linalg.norm(a - b), 1.0) ** (self.pl_exp / 2))

    def _quantise_theta(self, theta_cont):
        if self.codebook is None:
            return theta_cont
        return np.array([
            self.codebook[np.argmin(np.abs(
                np.exp(1j * self.codebook) - np.exp(1j * th)
            ))]
            for th in theta_cont
        ])

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, pilot_users=None, external_theta=None, ue_pos=None):
        self.t += 1

        if ue_pos is not None:
            self.ue_pos = np.array(ue_pos)

        # 1. move UAV
        self.uav_pos, self.uav_vel = self.uav.step()

        # 2. Doppler alphas
        alpha_BU = self._doppler_alpha(self.bs_pos,  self.uav_pos, self.uav_vel)
        alpha_kU = [
            self._doppler_alpha(self.uav_pos, self.ue_pos[k], self.uav_vel)
            for k in range(self.n_users)
        ]

        # 3. evolve channels
        self.H_BU = self._evolve(self.H_BU, (self.m, self.nt),
                                 self.bs_pos, self.uav_pos, alpha_BU)
        for k in range(self.n_users):
            self.g_kU[k] = self._evolve(self.g_kU[k], (self.m,),
                                        self.uav_pos, self.ue_pos[k], alpha_kU[k])

        # 4. path losses
        pl_BU = self._path_loss(self.bs_pos, self.uav_pos)
        pl_kU = np.array([self._path_loss(self.uav_pos, self.ue_pos[k])
                          for k in range(self.n_users)])

        # 5. RIS phase optimisation
        if external_theta is not None:
            self.theta = self._quantise_theta(external_theta)
        else:
            # Fallback to naive alignment to user 0
            combined   = self.g_kU[0] * (self.H_BU @ np.ones(self.nt))
            self.theta = self._quantise_theta(-np.angle(combined))
            
        phase      = np.exp(1j * self.theta)
        Theta      = np.diag(phase)

        # 6. effective channel  hk = g_kU^T Θ H_BU  (eq. 4)
        h_eff = np.zeros((self.n_users, self.nt), dtype=complex)
        for k in range(self.n_users):
            h_eff[k] = (self.g_kU[k] * pl_kU[k]) @ Theta @ (self.H_BU * pl_BU)

        # 7. per-tile contributions (visualisation)
        contributions = np.abs(self.g_kU[0] * phase) ** 2

        # 8. Age-of-CSI
        if pilot_users is None:
            pilot_users = list(range(self.n_users))
        for k in range(self.n_users):
            if k in pilot_users:
                self.last_pilot_slot[k] = self.t
            self.aocsi[k] = self.t - self.last_pilot_slot[k]

        uav_state = {
            "pos":      self.uav_pos.copy(),
            "vel":      self.uav_vel.copy(),
            "alpha_BU": alpha_BU,
            "alpha_kU": alpha_kU,
            "t":        self.t,
            "H_BU":     self.H_BU.copy(),
            "g_kU":     self.g_kU.copy(),
        }

        return h_eff, contributions, uav_state, self.aocsi.copy()

    def get_state_vector(self):
        return np.concatenate([
            self.H_BU.real.flatten(), self.H_BU.imag.flatten(),
            self.g_kU.real.flatten(), self.g_kU.imag.flatten(),
            self.uav_pos, self.uav_vel, self.theta,
        ])

    def channel_effect(self, user=0):
        """Backward compat for test_ls.py"""
        h_eff, _, _, _ = self.step()
        return h_eff[user]


# ── backward compatibility alias ─────────────────────────────────────────────
# All imports of channel_version2 continue to work unchanged
channel_version2 = channel_version3