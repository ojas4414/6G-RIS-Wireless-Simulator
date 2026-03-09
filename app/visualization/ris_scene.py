import pygame
import numpy as np
import time


# ── colour palette ────────────────────────────────────────────────────────────
BG          = (15,  18,  30)
GRID_BORDER = (180, 180, 180)
BS_COL      = (255, 220,  0)
UE_COL      = (  0, 150, 255)
UE_GHOST    = (  0, 200, 120)    # predicted position ghost
LABEL_COL   = (200, 200, 200)
GRAPH_COL   = (  0, 220,  80)
GRAPH_WARN  = (255, 160,   0)
GRAPH_BAD   = (220,  50,  50)
TRAIN_COL   = ( 80, 200, 255)
BEAM_BASE   = ( 60,  60,  60)


class RISScene:

    def __init__(self, M=64):
        pygame.init()

        self.width  = 1400
        self.height = 700

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RIS Wireless Simulator")

        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("Arial", 15)
        self.font_big = pygame.font.SysFont("Arial", 18, bold=True)

        # Node positions
        self.bs_pos   = [200, 350]
        self.user_pos = [1100, 350]

        # Predicted (ghost) user position — updated externally
        self.pred_pos      = None
        self.uncertainty   = None    # float variance from ANN

        # RIS grid
        self.grid_size = int(np.sqrt(M))
        self.cell_size = 20
        self.ris_x     = 600
        self.ris_y     = 250

        self.tile_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.ris_x + j * self.cell_size + self.cell_size // 2
                y = self.ris_y + i * self.cell_size + self.cell_size // 2
                self.tile_positions.append((x, y))

        # Smooth theta to reduce visual flicker
        self._theta_smooth = None

        # Smooth user_pos for lerp
        self._user_pos_smooth = np.array(self.user_pos, dtype=float)

        # Metrics history
        self.start_time       = time.time()
        self.last_graph_update = 0

        self.time_hist        = []
        self.power_hist       = []
        self.snr_hist         = []    # replaces duplicate RSSI
        self.dist_hist        = []
        self.uncertainty_hist = []    # real ANN log_var
        self.loss_hist        = []    # ANN training loss

        # Training state (updated externally)
        self.train_count  = 0
        self.latest_loss  = None
        self.buffer_size  = 0

        # Pulse animation for training indicator
        self._pulse_t = 0

    # ── smoothing helpers ─────────────────────────────────────────────────────

    def _lerp(self, a, b, alpha=0.25):
        return a * (1 - alpha) + b * alpha

    def _smooth_theta(self, theta, alpha=0.2):
        if self._theta_smooth is None:
            self._theta_smooth = theta.copy()
        else:
            self._theta_smooth = self._lerp(self._theta_smooth, theta, alpha)
        return self._theta_smooth

    # ── drawing: RIS tiles ────────────────────────────────────────────────────

    def draw_ris(self, theta):
        theta = self._smooth_theta(theta)

        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                phase = theta[idx]
                norm  = (phase % (2 * np.pi)) / (2 * np.pi)   # 0..1

                # Colour: cyan → magenta sweep
                r = int(80  + 175 * norm)
                g = int(20  + 60  * (1 - abs(2 * norm - 1)))
                b = int(180 - 100 * norm)

                x = self.ris_x + j * self.cell_size
                y = self.ris_y + i * self.cell_size
                pygame.draw.rect(self.screen, (r, g, b),
                                 (x, y, self.cell_size - 1, self.cell_size - 1))
                idx += 1

        pygame.draw.rect(self.screen, GRID_BORDER,
                         (self.ris_x - 4, self.ris_y - 4,
                          self.grid_size * self.cell_size + 8,
                          self.grid_size * self.cell_size + 8), 2)

    # ── drawing: nodes ────────────────────────────────────────────────────────

    def draw_nodes(self):
        # Smooth lerp for user dot movement
        target = np.array(self.user_pos, dtype=float)
        self._user_pos_smooth = self._lerp(self._user_pos_smooth, target, alpha=0.3)
        up = (int(self._user_pos_smooth[0]), int(self._user_pos_smooth[1]))

        # Base station
        pygame.draw.circle(self.screen, BS_COL, self.bs_pos, 12)
        pygame.draw.circle(self.screen, (255, 255, 255), self.bs_pos, 12, 1)

        # Real user position
        pygame.draw.circle(self.screen, UE_COL, up, 12)
        pygame.draw.circle(self.screen, (255, 255, 255), up, 12, 1)

        # Ghost predicted position (if ANN has made a prediction)
        if self.pred_pos is not None:
            px, py = int(self.pred_pos[0]), int(self.pred_pos[1])
            # Uncertainty ring: bigger + more transparent when uncertain
            u = min(self.uncertainty or 1.0, 5.0)
            ring_r = int(14 + u * 6)
            pygame.draw.circle(self.screen, UE_GHOST, (px, py), 10, 2)
            pygame.draw.circle(self.screen, (*UE_GHOST, 80), (px, py), ring_r, 1)
            # dashed connector
            self._draw_dashed_line(UE_GHOST, up, (px, py), dash=8)

    def _draw_dashed_line(self, color, p1, p2, dash=10):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = max(1, int((dx**2 + dy**2) ** 0.5))
        for i in range(0, dist, dash * 2):
            t0 = i / dist
            t1 = min((i + dash) / dist, 1.0)
            x0 = int(p1[0] + dx * t0)
            y0 = int(p1[1] + dy * t0)
            x1 = int(p1[0] + dx * t1)
            y1 = int(p1[1] + dy * t1)
            pygame.draw.line(self.screen, color, (x0, y0), (x1, y1), 1)

    # ── drawing: beams ────────────────────────────────────────────────────────

    def draw_tile_beams(self, contributions):
        max_c = max(contributions) + 1e-9
        up    = (int(self._user_pos_smooth[0]), int(self._user_pos_smooth[1]))

        for i, tile in enumerate(self.tile_positions):
            s = contributions[i] / max_c

            # BS → tile: grey
            pygame.draw.line(self.screen, BEAM_BASE, self.bs_pos, tile, 1)

            # tile → UE: colour shifts grey→white→yellow with strength
            if s > 0.6:
                r_val = 255
                g_val = int(55 + 200 * (s - 0.6) / 0.4)
                b_val = int(255 * (1.0 - s))
            else:
                r_val = int(60 + 195 * s)
                g_val = int(60 + 195 * s)
                b_val = int(60 + 195 * s)
            # clamp all channels to valid pygame range 0-255
            r_val = max(0, min(255, r_val))
            g_val = max(0, min(255, g_val))
            b_val = max(0, min(255, b_val))
            w = 1 + int(2 * s)
            pygame.draw.line(self.screen, (r_val, g_val, b_val), tile, up, w)

    # ── drawing: labels ───────────────────────────────────────────────────────

    def draw_labels(self):
        up = (int(self._user_pos_smooth[0]), int(self._user_pos_smooth[1]))

        for text, pos, anchor in [
            ("Base Station",   (self.bs_pos[0] - 45, self.bs_pos[1] + 18),  None),
            ("User Equipment", (up[0] - 60,          up[1] + 18),           None),
            ("RIS Surface",    (self.ris_x,           self.ris_y - 26),      None),
        ]:
            surf = self.font.render(text, True, LABEL_COL)
            self.screen.blit(surf, pos)

        if self.pred_pos is not None:
            px = int(self.pred_pos[0])
            py = int(self.pred_pos[1])
            surf = self.font.render("Predicted", True, UE_GHOST)
            self.screen.blit(surf, (px - 28, py + 16))

    # ── drawing: training HUD ─────────────────────────────────────────────────

    def draw_training_hud(self):
        self._pulse_t += 0.08
        pulse = int(128 + 127 * np.sin(self._pulse_t))

        # Training sample count
        txt = f"Collected: {self.buffer_size} samples"
        surf = self.font.render(txt, True, TRAIN_COL)
        self.screen.blit(surf, (10, 10))

        # Pulsing "TRAINING" dot
        if self.train_count > 0:
            dot_col = (0, pulse, 255)
            pygame.draw.circle(self.screen, dot_col, (16, 45), 6)
            txt2 = f"Training steps: {self.train_count}"
            surf2 = self.font.render(txt2, True, dot_col)
            self.screen.blit(surf2, (28, 38))

        # Latest loss
        if self.latest_loss is not None:
            loss_col = GRAPH_COL if self.latest_loss < 1.0 else \
                       GRAPH_WARN if self.latest_loss < 5.0 else GRAPH_BAD
            txt3 = f"Loss: {self.latest_loss:.4f}"
            surf3 = self.font.render(txt3, True, loss_col)
            self.screen.blit(surf3, (10, 60))

    # ── metrics update ────────────────────────────────────────────────────────

    def update_metrics(self, power, uncertainty=None, loss=None):
        now = time.time()
        if now - self.last_graph_update < 0.15:    # ~6 graph updates/sec
            return
        self.last_graph_update = now

        distance  = np.linalg.norm(
            np.array(self.bs_pos) - np.array(self._user_pos_smooth))
        power_db  = float(10 * np.log10(max(power, 1e-12)))

        # SNR: approximate noise floor at -90 dBm
        snr = power_db + 90.0

        t = now - self.start_time

        self.time_hist.append(t)
        self.power_hist.append(power_db)
        self.snr_hist.append(snr)
        self.dist_hist.append(distance)
        self.uncertainty_hist.append(float(uncertainty) if uncertainty else 0.0)
        if loss is not None:
            self.loss_hist.append(float(loss))

        for arr in [self.time_hist, self.power_hist, self.snr_hist,
                    self.dist_hist, self.uncertainty_hist, self.loss_hist]:
            if len(arr) > 120:
                arr.pop(0)

    # ── graph drawing ─────────────────────────────────────────────────────────

    def draw_graph(self, x, y, w, h, data, title, color=GRAPH_COL):
        pygame.draw.rect(self.screen, (45, 45, 55), (x, y, w, h))
        pygame.draw.rect(self.screen, (80, 80, 90), (x, y, w, h), 1)

        label = self.font.render(title, True, color)
        self.screen.blit(label, (x, y - 20))

        if len(data) < 2:
            return

        min_v = min(data)
        max_v = max(data) + 1e-9

        pts = []
        for i, v in enumerate(data):
            px = x + int((i / len(data)) * w)
            py = y + h - int((v - min_v) / (max_v - min_v) * h)
            pts.append((px, max(y, min(y + h, py))))

        pygame.draw.lines(self.screen, color, False, pts, 2)

        # Latest value label
        val_txt = f"{data[-1]:.2f}"
        vs = self.font.render(val_txt, True, color)
        self.screen.blit(vs, (x + w - vs.get_width() - 4, y + 2))

    def draw_graphs(self):
        g_w = 290
        g_h = 110
        y   = self.height - 145

        self.draw_graph(20,  y, g_w, g_h, self.power_hist,
                        "Received Power (dB)", GRAPH_COL)
        self.draw_graph(330, y, g_w, g_h, self.snr_hist,
                        "SNR (dB)", GRAPH_COL)
        self.draw_graph(640, y, g_w, g_h, self.dist_hist,
                        "Distance (px)", GRAPH_WARN)

        # Uncertainty: colour by ANN confidence
        u_col = GRAPH_COL if (self.uncertainty or 0) < 1.0 else GRAPH_WARN
        self.draw_graph(950, y, g_w, g_h, self.uncertainty_hist,
                        "ANN Uncertainty (var)", u_col)

        # ANN loss — only if data exists
        if len(self.loss_hist) > 1:
            loss_col = GRAPH_COL if self.loss_hist[-1] < 1.0 else \
                       GRAPH_WARN if self.loss_hist[-1] < 5.0 else GRAPH_BAD
            pygame.draw.rect(self.screen, (45, 45, 55),
                             (1260, y, 120, g_h))
            self.draw_graph(1260, y, 120, g_h, self.loss_hist,
                            "Train Loss", loss_col)

    # ── main update ───────────────────────────────────────────────────────────

    def update(self, theta, contributions, power_history,
               pred_pos=None, uncertainty=None, loss=None,
               train_count=0, buffer_size=0):
        """
        theta           : np.ndarray (M,)  current RIS phases
        contributions   : np.ndarray (M,)  per-tile beam strength
        power_history   : list of floats   recent power values
        pred_pos        : (x, y) pixel coords of ANN predicted user position
        uncertainty     : float  ANN output variance
        loss            : float  latest training loss
        train_count     : int    cumulative training steps
        buffer_size     : int    samples collected so far
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Store externally provided ANN state
        self.pred_pos    = pred_pos
        self.uncertainty = uncertainty
        self.latest_loss = loss
        self.train_count = train_count
        self.buffer_size = buffer_size

        self.screen.fill(BG)

        self.draw_tile_beams(contributions)
        self.draw_nodes()
        self.draw_ris(theta)
        self.draw_labels()
        self.draw_training_hud()

        power_scalar = power_history[-1] if isinstance(power_history, list) \
                       and len(power_history) > 0 else float(power_history)
        self.update_metrics(power_scalar, uncertainty=uncertainty, loss=loss)
        self.draw_graphs()

        pygame.display.flip()
        self.clock.tick(15)    # 15 fps — readable for humans