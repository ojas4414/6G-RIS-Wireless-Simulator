import numpy as np

class RobustBeamformer:
    """
    Solves for the robust Base Station precoder w and RIS phase vector theta
    under Channel State Information (CSI) uncertainty.
    
    This implements Alternating Optimization (AO) with a hard iteration limit
    for real-time performance to avoid frame drops.
    """
    def __init__(self, nt, m, max_iters=3):
        self.nt = nt
        self.m = m
        self.max_iters = max_iters

    def solve(self, H_BU, g_kU, uncertainty, P_max=1.0):
        """
        Alternating optimization for w and theta.
        H_BU: (M, Nt) complex channel from BS to RIS
        g_kU: (M,) complex channel from RIS to User
        uncertainty: Float, representing uncertainty radius.
        P_max: Maximum transmit power.
        
        Returns:
            w: (Nt,) complex precoding vector
            theta: (M,) real phase shift vector (0 to 2pi)
        """
        # Initialize theta to random phases (or naive alignment)
        # Using naive alignment as a good starting point
        combined = g_kU * (H_BU @ np.ones(self.nt))
        theta = -np.angle(combined)
        
        # Pre-allocate w
        w = np.ones(self.nt, dtype=complex) / np.sqrt(self.nt)
        
        for _ in range(self.max_iters):
            # Step 1: Optimize w for fixed theta
            # cascaded channel: h_casc = g_kU^T * diag(exp(j*theta)) * H_BU
            phase = np.exp(1j * theta)
            h_casc = (g_kU * phase) @ H_BU  # Shape (Nt,)
            
            # Robust precoder w: align with h_casc (Maximum Ratio Transmission)
            # Under spherical uncertainty, SNR worst-case is maximized when w is aligned with h_casc
            norm_h = np.linalg.norm(h_casc)
            if norm_h > 1e-9:
                w = np.conj(h_casc) / norm_h
            w = w * np.sqrt(P_max)
            
            # Step 2: Optimize theta for fixed w
            # cascaded response: r = g_kU * (H_BU @ w)  -> shape (M,)
            r = g_kU * (H_BU @ w)
            # To maximize |sum(r * phase)|, we align phase with the conjugate of r
            theta = -np.angle(r)
            
        # Eq (16) Constraint: Snap to Discrete Quantized Set Q (2-bit phase)
        quantization_steps = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        diffs = np.abs(np.exp(1j * theta[:, None]) - np.exp(1j * quantization_steps[None, :]))
        indices = np.argmin(diffs, axis=1)
        theta = quantization_steps[indices]
            
        return w, theta
