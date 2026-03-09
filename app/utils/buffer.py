import numpy as np


class Rollingbuffer:
    """
    Circular buffer that stores the last `window_size` channel vectors.
    Used to build (X, y) pairs for online ANN training.
    """

    def __init__(self, window_size, Nt):
        self.window_size = window_size
        self.Nt          = Nt
        self.buffer      = np.zeros((window_size, Nt), dtype=complex)
        self.index       = 0
        self.full        = False

    # FIX: these methods were wrongly indented inside __init__ in the original

    def update(self, new_value):
        """Push one channel vector (shape: Nt,) into the buffer."""
        self.buffer[self.index] = new_value
        self.index = (self.index + 1) % self.window_size
        if self.index == 0:
            self.full = True

    def get_flattened(self):
        """
        Returns the buffer contents as a flat real feature vector:
        [real_0, ..., real_{W-1}, imag_0, ..., imag_{W-1}]  shape: (2*W*Nt,)
        Returns None if the buffer is not yet full.
        """
        if not self.full:
            return None

        ordered = np.roll(self.buffer, -self.index, axis=0)  # oldest first

        real = ordered.real.reshape(-1)
        imag = ordered.imag.reshape(-1)

        return np.concatenate([real, imag])

    def get_ordered(self):
        """Returns the buffer as a (window_size, Nt) complex array, oldest first."""
        if not self.full:
            return None
        return np.roll(self.buffer, -self.index, axis=0)

    def size(self):
        """Number of valid entries currently stored."""
        if self.full:
            return self.window_size
        return self.index