

import numpy as np

from app.simulator.channelv2    import channel_version2
from app.visualization.ris_scene import RISScene
from app.utils.hand_controller   import HandController
from app.utils.buffer            import Rollingbuffer
from app.utils.csi_logger        import CSILogger
from app.models.ann              import ANN


# ── hyper-parameters ──────────────────────────────────────────────────────────
NT          = 4
M           = 64
WINDOW_SIZE = 20
HIDDEN_DIM  = 64
INPUT_DIM   = 2 * WINDOW_SIZE * NT   # real + imag of window
OUTPUT_DIM  = 2 * NT + 1             # mu (real+imag of Nt) + log_var
# ─────────────────────────────────────────────────────────────────────────────

env    = channel_version2(nt=NT, m=M, n_users=1, velocity=5)
scene  = RISScene(M=M)
hand   = HandController()
buffer = Rollingbuffer(window_size=WINDOW_SIZE, Nt=NT)
logger = CSILogger()
ann    = ANN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
             output_dim=OUTPUT_DIM, nt=NT, lr=0.0005)

power_history = []
running       = True

# ANN output state passed to scene each frame
pred_pos    = None
uncertainty = None
ann_loss    = None


def h_to_pred_pos(h_pred, scene_ref):
    """
    Map predicted channel power to an estimated pixel position.
    Larger predicted power -> UE closer to RIS.
    """
    power = np.linalg.norm(h_pred) ** 2
    t     = float(np.tanh(power / 2.0))
    px    = int(scene_ref.ris_x + (scene_ref.width - scene_ref.ris_x - 80) * t)
    py    = scene_ref.user_pos[1]
    return px, py


try:
    while running:
        try:
            # 1. finger input
            finger = hand.read()
            if finger is not None:
                fx, fy = finger
                scene.user_pos[0] = int(800 + fx * 400)
                scene.user_pos[1] = int(fy * scene.height)

            # 2. channel step — v3 returns 4 values
            h_all, contributions, uav_state, aocsi = env.step()
            h = h_all[0]   # single user, shape (NT,)

            power = float(np.linalg.norm(h) ** 2)
            power_history.append(power)
            if len(power_history) > 200:
                power_history.pop(0)

            # 3. log
            logger.record(h, env.theta, scene.user_pos)

            # 4. rolling buffer
            buffer.update(h)

            # 5. ANN predict + train
            x_flat = buffer.get_flattened()
            if x_flat is not None:
                X      = x_flat.reshape(1, -1)
                h_pred, uncertainty = ann.predict_h(X)
                pred_pos = h_to_pred_pos(h_pred, scene)
                y_true   = np.concatenate([h.real, h.imag]).reshape(1, -1)
                ann.backward(X, y_true)
                ann_loss = ann.latest_loss

            # 6. render
            scene.update(
                theta         = env.theta,
                contributions = contributions,
                power_history = power_history,
                pred_pos      = pred_pos,
                uncertainty   = uncertainty,
                loss          = ann_loss,
                train_count   = ann.train_count,
                buffer_size   = logger.count(),
            )

        except KeyboardInterrupt:
            running = False

finally:
    hand.close()
    logger.save("dataset/csi_dataset.npy")
    ann.save("models/ann_weights.pkl")

    print("=" * 50)
    print(f"  Training steps  : {ann.train_count}")
    print(f"  Samples logged  : {logger.count()}")
    if ann.latest_loss:
        print(f"  Final loss      : {ann.latest_loss:.6f}")
    else:
        print("  Final loss      : N/A (buffer never filled)")
    print(f"  Dataset saved   : dataset/csi_dataset.npy")
    print(f"  Model saved     : models/ann_weights.pkl")
    print("=" * 50)