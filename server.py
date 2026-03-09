"""
server.py  -  FastAPI WebSocket backend for RIS Wireless Simulator
Run with:   uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import threading
import time
import numpy as np
import os
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Ensure these paths match your project structure
from app.simulator.channelv2   import channel_version2
from app.utils.hand_controller import HandController
from app.utils.buffer           import Rollingbuffer
from app.utils.csi_logger      import CSILogger
from app.models.ann            import ANN
from app.optimization.robust_beamforming import RobustBeamformer
from app.optimization.lyapunov_scheduler import LyapunovScheduler

# ── config ────────────────────────────────────────────────────────────────────
NT          = 4
M           = 64
WINDOW_SIZE = 20
HIDDEN_DIM  = 64
INPUT_DIM   = 2 * WINDOW_SIZE * NT
OUTPUT_DIM  = 2 * NT + 1
TARGET_FPS  = 30  # Increased to 30 for smoother finger tracking

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── shared state ──────────────────────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.running  = False
        self.thread   = None
        self.frame    = {}
        self.lock     = threading.Lock()
        self.user_pos = [[800, 350, 1.5], [400, 350, 1.5]]
        self.user_dir = 5

state = SimState()

# ── uncertainty zone helper ───────────────────────────────────────────────────
def compute_uncertainty_zone(uncertainty: float, user_pos: list) -> dict:
    scale = float(np.clip(uncertainty, 0.1, 10.0))
    rx    = int(30 + scale * 20)
    ry    = int(30 + scale * 12)
    return {"cx": user_pos[0], "cy": user_pos[1], "rx": rx, "ry": ry}

# ── simulation loop (background thread) ──────────────────────────────────────
def simulation_loop():
    env    = channel_version2(nt=NT, m=M, n_users=2, velocity=5)
    hand   = None
    try:
        hand = HandController()
        print("[Simulation] HandController initialized")
    except Exception as e:
        print(f"[Simulation] WARNING: Hand tracking unavailable: {e}")
        print("[Simulation] Running without hand control (use mouse on canvas instead)")
    
    buf    = Rollingbuffer(window_size=WINDOW_SIZE, Nt=NT)
    logger = CSILogger()
    ann    = ANN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=OUTPUT_DIM, nt=NT, lr=0.0005)
    
    beamformer = RobustBeamformer(nt=NT, m=M)
    scheduler  = LyapunovScheduler(n_users=2, V=5.0)

    power_history = []
    pred_zone     = None
    uncertainty   = 1.0
    ann_loss      = None
    frame_dt      = 1.0 / TARGET_FPS

    print("[Simulation] Loop Started")
    
    uav_state_prev = {}
    last_snr = [0.0, 0.0]
    external_theta = None
    pilot_users = [0]
    lyapunov_dpp_arr = [0.0, 0.0]
    
    # Initialize AoCSI tracking explicitly
    if not hasattr(env, 'aocsi'):
        env.aocsi = np.zeros(2)

    try:
        while state.running:
            try:
                t0 = time.time()

                # 1. hand input (with fallback)
                if hand is not None:
                    try:
                        finger = hand.read()
                        if finger is not None:
                            fx, fy = finger
                            # Coordinate Mapping
                            state.user_pos[0][0] = int(800 + fx * 400)
                            state.user_pos[0][1] = int(300 + fy * 400)
                    except Exception as e:
                        print(f"[Simulation] Hand read error (simulation continues): {e}")

                # Autonomous movement for User 1 (eMBB)
                state.user_pos[1][0] += state.user_dir
                if state.user_pos[1][0] > 1200 or state.user_pos[1][0] < 200:
                    state.user_dir *= -1

                # 2. Advanced Algorithms - Lyapunov AoCSI Scheduler
                try:
                    estimated_rate = [np.log2(1 + 10**(s / 10)) for s in last_snr]
                    pilot_users = scheduler.step(env.aocsi, estimated_rate)
                except Exception as e:
                    print(f"[Simulation] Scheduler error: {e}")
                    pilot_users = [0]
                
                # Record the evaluated Drift-Plus-Penalty metric
                try:
                    lyapunov_dpp_arr = [0.0, 0.0]
                    for k in range(2):
                        A = env.aocsi[k]
                        R = estimated_rate[k]
                        if k in pilot_users:
                            lyapunov_dpp_arr[k] = -0.5 * (A**2)
                        else:
                            lyapunov_dpp_arr[k] = (A + 0.5) - float(scheduler.V[k] if hasattr(scheduler.V, '__len__') else scheduler.V) * R
                except Exception as e:
                    print(f"[Simulation] Lyapunov calc error: {e}")
                    lyapunov_dpp_arr = [0.0, 0.0]
                
                # Robust Beamforming / RIS Optimization
                try:
                    if 'H_BU' in uav_state_prev:
                        w, external_theta = beamformer.solve(uav_state_prev['H_BU'], uav_state_prev['g_kU'][0], uncertainty)
                    else:
                        external_theta = None
                except Exception as e:
                    print(f"[Simulation] Beamformer error: {e}")
                    external_theta = None

                # 3. channel step
                try:
                    h_all, contributions, uav_state, aocsi = env.step(pilot_users=pilot_users, external_theta=external_theta, ue_pos=state.user_pos)
                    h = h_all[0]   # single user
                    uav_state_prev = uav_state
                except Exception as e:
                    print(f"[Simulation] Channel step error: {e}")
                    traceback.print_exc()
                    continue

                power = [float(np.linalg.norm(h_all[k]) ** 2) for k in range(2)]
                power_history.append(power[0])
                if len(power_history) > 200:
                    power_history.pop(0)

                # Log CSI
                try:
                    logger.record(h, env.theta, state.user_pos[0][:2])
                except Exception as e:
                    print(f"[Simulation] Logger error: {e}")

                # Buffer + ANN training
                try:
                    buf.update(h)
                    x_flat = buf.get_flattened()
                    if x_flat is not None:
                        X      = x_flat.reshape(1, -1)
                        y_true = np.concatenate([h.real, h.imag]).reshape(1, -1)
                        ann.backward(X, y_true)
                        _, log_var  = ann.forward(X)
                        uncertainty = float(np.exp(np.clip(log_var[0, 0], -10, 10)))
                        ann_loss    = ann.latest_loss
                        pred_zone   = compute_uncertainty_zone(uncertainty, state.user_pos[0][:2])
                except Exception as e:
                    print(f"[Simulation] ANN/Buffer error: {e}")

                # Build frame for React
                try:
                    power_db = [float(10 * np.log10(max(p, 1e-12))) for p in power]
                    snr      = [p + 90.0 for p in power_db]
                    last_snr = snr
                    distance = [float(np.linalg.norm(np.array([200, 350]) - np.array(state.user_pos[k][:2]))) for k in range(2)]

                    frame = {
                        "theta":         env.theta.tolist(),
                        "contributions": contributions.tolist(),
                        "user_pos":      [list(pos) for pos in state.user_pos],
                        "uav_pos":       uav_state["pos"].tolist(),
                        "uav_vel":       uav_state["vel"].tolist(),
                        "alpha_bu":      round(float(uav_state["alpha_BU"]), 4),
                        "aocsi":         [int(np.sum(a)) for a in aocsi],  # Defensive cast in case aocsi becomes array
                        "power_db":      [round(float(p), 3) for p in power_db],
                        "snr":           [round(float(s), 3) for s in snr],
                        "distance":      [round(float(d), 1) for d in distance],
                        "uncertainty":   round(float(uncertainty), 4),
                        "lyapunov_dpp":  [round(float(np.sum(d)), 2) for d in lyapunov_dpp_arr], # Defensive sum
                        "pilot_sent":    pilot_users,
                        "ann_loss":      round(float(ann_loss), 5) if ann_loss else None,
                        "train_count":   ann.train_count,
                        "buffer_size":   logger.count(),
                        "pred_zone":     pred_zone,
                        "power_history": power_history[-60:],
                    }

                    with state.lock:
                        state.frame = frame
                except Exception as e:
                    print(f"[Simulation] Frame build error: {e}")
                    traceback.print_exc()

                elapsed = time.time() - t0
                time.sleep(max(0.0, frame_dt - elapsed))

            except Exception as e:
                print(f"[Simulation] Iteration error: {e}")
                traceback.print_exc()
                time.sleep(0.01)  # Prevent tight loop if errors keep occurring

    except Exception as e:
        print(f"[Simulation Error] {e}")
        traceback.print_exc()

    finally:
        if hand is not None:
            hand.close()
        # Create directories if they don't exist
        os.makedirs("dataset", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        logger.save("dataset/csi_dataset.npy")
        ann.save("models/ann_weights.pkl")
        print(f"[Server] Stopped. Steps: {ann.train_count}, Samples: {logger.count()}")

# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    try:
        while True:
            try:
                # Listen for commands from React
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                msg = json.loads(raw)
                cmd = msg.get("cmd")

                if cmd == "start" and not state.running:
                    state.running = True
                    state.thread  = threading.Thread(target=simulation_loop, daemon=True)
                    state.thread.start()
                    await websocket.send_text(json.dumps({"status": "started"}))

                elif cmd == "stop" and state.running:
                    state.running = False
                    await websocket.send_text(json.dumps({"status": "stopped"}))

                elif cmd == "move":
                    state.user_pos[0][0] = int(msg.get("x", state.user_pos[0][0]))
                    state.user_pos[0][1] = int(msg.get("y", state.user_pos[0][1]))

            except asyncio.TimeoutError:
                pass

            # Send the current frame to React
            if state.running:
                frame_to_send = {}
                with state.lock:
                    frame_to_send = dict(state.frame)
                
                if frame_to_send:
                    try:
                        packed_data = json.dumps(frame_to_send)
                        await websocket.send_text(packed_data)
                    except TypeError as e:
                        print(f"[WS] Critical JSON Dump Error: {e}")
                        # Defensive scan to find the exact un-serializable feature and strip it
                        safe_frame = {}
                        for key, val in frame_to_send.items():
                            try:
                                json.dumps(val)
                                safe_frame[key] = val
                            except TypeError:
                                print(f"[WS] Stripping un-serializable Numpy variable: {key} -> {type(val)}")
                        await websocket.send_text(json.dumps(safe_frame))

            await asyncio.sleep(1.0 / TARGET_FPS)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
        state.running = False


# ── Main entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("[Server] Starting FastAPI + WebSocket server on port 8005...")
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")