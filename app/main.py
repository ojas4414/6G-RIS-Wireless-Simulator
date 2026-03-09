from app.simulator.channel import WirelessEnvironment
import matplotlib.pyplot as plt
import numpy as np


def test_environment():
    env = WirelessEnvironment(Nt=4, M=16, velocity=5)

    T = 500
    magnitudes = []

    for _ in range(T):
        h = env.step()
        magnitudes.append(np.linalg.norm(h))

    plt.plot(magnitudes)
    plt.title("Effective Channel Norm Over Time")
    plt.xlabel("Time Slot")
    plt.ylabel("||h_eff||")
    plt.show()


if __name__ == "__main__":
    test_environment()