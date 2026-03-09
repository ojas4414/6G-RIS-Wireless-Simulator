import { useEffect, useRef } from "react";

const W = 1100; const H = 380;
const BS_POS = [120, 190];
const RIS_X = 480; const RIS_Y = 100;
const GRID = 8; const CELL = 22;

export function Sparkline({ data, color, label, unit, height = 80 }) {
  const ref = useRef();
  useEffect(() => {
    const c = ref.current;
    if (!c || data.length < 2) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    const mn = Math.min(...data), mx = Math.max(...data) + 1e-9;
    const pts = data.map((v, i) => [(i / (data.length - 1)) * w, h - ((v - mn) / (mx - mn)) * h * 0.88 - h * 0.06]);
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.stroke();
  }, [data, color]);

  return (
    <div style={{ flex: 1, minWidth: 160 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: "#666", textTransform: "uppercase" }}>{label}</span>
      </div>
      <canvas ref={ref} width={200} height={height} style={{ width: "100%", height, background: "rgba(0,0,0,0.3)", borderRadius: 4 }} />
    </div>
  );
}

export default function RISSimulator({ frame, onMouseMove }) {
  const ref = useRef();
  const smooth = useRef({ u0X: 800, u0Y: 350, u1X: 400, u1Y: 350, uavX: 480, uavY: 260, theta: null });

  useEffect(() => {
    if (!frame || !ref.current) return;
    const ctx = ref.current.getContext("2d");
    const sm = smooth.current;

    // Smooth movement URLLC (Ambulance)
    const tx0 = frame.user_pos[0][0] * (W / 1400);
    const ty0 = frame.user_pos[0][1] * (H / 700);
    sm.u0X += (tx0 - sm.u0X) * 0.3;
    sm.u0Y += (ty0 - sm.u0Y) * 0.3;
    const up0 = [~~sm.u0X, ~~sm.u0Y];

    // Smooth movement eMBB (Car)
    const tx1 = frame.user_pos[1][0] * (W / 1400);
    const ty1 = frame.user_pos[1][1] * (H / 700);
    sm.u1X += (tx1 - sm.u1X) * 0.3;
    sm.u1Y += (ty1 - sm.u1Y) * 0.3;
    const up1 = [~~sm.u1X, ~~sm.u1Y];

    const uav3d = frame.uav_pos || [300, 300, 100];
    const uavPx = [~~(200 + (uav3d[0] / 600) * 700), ~~(30 + (uav3d[1] / 600) * 280)];
    sm.uavX += (uavPx[0] - sm.uavX) * 0.15;
    sm.uavY += (uavPx[1] - sm.uavY) * 0.15;
    const uavP = [~~sm.uavX, ~~sm.uavY];

    if (!sm.theta) sm.theta = [...frame.theta];
    sm.theta = sm.theta.map((t, i) => t + (frame.theta[i] - t) * 0.2);

    ctx.clearRect(0, 0, W, H);

    // Draw Sleek RIS Hardware Border
    const hwPadding = 6;
    ctx.shadowColor = "#00d4ff"; ctx.shadowBlur = 10;
    ctx.strokeStyle = "rgba(0, 212, 255, 0.5)"; ctx.lineWidth = 2;
    ctx.strokeRect(RIS_X - hwPadding, RIS_Y - hwPadding, GRID * CELL + hwPadding * 2, GRID * CELL + hwPadding * 2);
    ctx.shadowBlur = 0; // reset

    // Draw RIS Meta-lens elements (sleek internal nodes)
    sm.theta.forEach((phase, idx) => {
      const i = ~~(idx / GRID), j = idx % GRID;
      const norm = ((phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) / (2 * Math.PI);
      const cx = RIS_X + j * CELL + CELL / 2;
      const cy = RIS_Y + i * CELL + CELL / 2;

      ctx.fillStyle = `hsla(${160 + norm * 200}, 90%, 65%, 0.85)`;
      ctx.beginPath();
      // Render as a sleek optical lens rather than a block
      ctx.arc(cx, cy, CELL / 2 - 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Dynamic ANN Uncertainty Bubble
    if (frame.pred_zone) {
      const pz = frame.pred_zone;
      const bX = pz.cx * (W / 1400);
      const bY = pz.cy * (H / 700);
      // Pulsing effect based on uncertainty scale
      ctx.beginPath();
      ctx.ellipse(bX, bY, pz.rx, pz.ry, 0, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0, 255, 157, 0.15)";
      ctx.fill();
      ctx.strokeStyle = "rgba(0, 255, 157, 0.5)";
      ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]); ctx.stroke();
      ctx.setLineDash([]);
    }

    // DEBUG: RAW INPUT DOT (Red)
    ctx.beginPath(); ctx.arc(tx0, ty0, 4, 0, Math.PI * 2);
    ctx.fillStyle = "red"; ctx.fill();

    // The "Searchlight" Effect (Animated beams via Lyapunov Tracker)
    const rx = RIS_X + 80; const ry = RIS_Y + 80;
    const isAmbulanceTracked = frame.pilot_sent && frame.pilot_sent.includes(0);
    const isCarTracked = frame.pilot_sent && frame.pilot_sent.includes(1);

    // Tower to RIS
    ctx.beginPath(); ctx.moveTo(...BS_POS); ctx.lineTo(rx, ry);
    ctx.strokeStyle = "rgba(0, 212, 255, 0.5)"; ctx.lineWidth = 2; ctx.stroke();

    // RIS to URLLC Ambulance
    ctx.beginPath();
    if (isAmbulanceTracked) {
      // Actively Tracking: Bright Ping + Solid Beam
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)"; ctx.shadowColor = "white"; ctx.shadowBlur = 15; ctx.lineWidth = 3;
      ctx.arc(up0[0], up0[1], 30, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 80, 80, 0.3)"; ctx.fill();
      ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(...up0);
    } else {
      // Predictive/Blind: Faded Dotted Red 
      ctx.moveTo(rx, ry); ctx.lineTo(...up0);
      ctx.strokeStyle = "rgba(255, 80, 80, 0.2)"; ctx.shadowBlur = 0; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
    }
    ctx.stroke(); ctx.setLineDash([]); ctx.shadowBlur = 0;

    // RIS to eMBB Car
    ctx.beginPath();
    if (isCarTracked) {
      // Actively Tracking 
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)"; ctx.shadowColor = "white"; ctx.shadowBlur = 15; ctx.lineWidth = 3;
      ctx.arc(up1[0], up1[1], 30, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0, 212, 255, 0.3)"; ctx.fill();
      ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(...up1);
    } else {
      // Predictive/Blind Faded Blue
      ctx.moveTo(rx, ry); ctx.lineTo(...up1);
      ctx.strokeStyle = "rgba(0, 212, 255, 0.2)"; ctx.shadowBlur = 0; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
    }
    ctx.stroke(); ctx.setLineDash([]); ctx.shadowBlur = 0;

    // Entities
    ctx.font = "30px Arial"; ctx.textAlign = "center";
    ctx.fillText("🗼", BS_POS[0], BS_POS[1]);
    ctx.fillText("🛸", uavP[0], uavP[1]);
    ctx.fillText("🚑", up0[0], up0[1]);
    ctx.fillText("🚗", up1[0], up1[1]);

  }, [frame]);

  return <canvas ref={ref} width={W} height={H} onMouseMove={onMouseMove} style={{ width: "100%", background: "#040812", borderRadius: 8 }} />;
}