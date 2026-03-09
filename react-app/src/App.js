import { useState, useEffect, useRef, useCallback } from "react";
import RISSimulator, { Sparkline } from "./RISSimulator";

// Ensure this matches your uvicorn server port (8005)
const WS_URL = "ws://localhost:8005/ws";

export default function App() {
  const ws = useRef(null);
  const [wsReady, setWsReady] = useState(false);
  const [status, setStatus] = useState("idle");
  const [frame, setFrame] = useState(null);
  const [error, setError] = useState(null);

  const [powerH, setPowerH] = useState([]);
  const [snrH, setSnrH] = useState([]);
  const [distH, setDistH] = useState([]);
  const [analysisText, setAnalysisText] = useState("📡 Predictive Coasting: Waiting for next decision sequence.");
  const [displayFrame, setDisplayFrame] = useState(null);
  const lastUpdate = useRef(0);
  const lastDisplayUpdate = useRef(0);
  const lastMouseTx = useRef(0);

  const push = (setter, val) => setter(p => {
    const n = [...p, val];
    return n.length > 80 ? n.slice(-80) : n;
  });

  useEffect(() => {
    const sock = new WebSocket(WS_URL);
    ws.current = sock;

    sock.onopen = () => {
      console.log("WS Opened");
      setError(null); // Clear previous errors
      setWsReady(true);
    };

    sock.onclose = () => {
      setWsReady(false);
      setFrame(null);
    };

    sock.onerror = () => {
      setError("WebSocket connection failed. Ensure backend is running on port 8000.");
    };

    sock.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.status) {
        setStatus(d.status === "started" ? "running" : "idle");
      } else {
        setFrame(d);
        push(setPowerH, d.power_db[0]);
        push(setSnrH, d.snr[0]);
        push(setDistH, d.distance[0]);

        const now = Date.now();
        if (now - lastDisplayUpdate.current > 800) {
          setDisplayFrame(d);
          lastDisplayUpdate.current = now;
        }

        if (now - lastUpdate.current > 1500) {
          let text = "📡 Predictive Coasting: Waiting for next decision sequence.";
          if (d.pilot_sent?.includes(0) && d.pilot_sent?.includes(1)) {
            text = "🚨 WARNING: Both vehicles tracking simultaneously (Hardware Limit Exceeded).";
          } else if (d.pilot_sent?.includes(0)) {
            text = `🚨 Ambulance tracking became stale. AI forcing RIS to aim searchlight at Ambulance, dropping Car connection.`;
          } else if (d.pilot_sent?.includes(1)) {
            text = `✅ Ambulance signal is fresh. AI flipping searchlight to Civilian Car to maximize throughput.`;
          }
          setAnalysisText(text);
          lastUpdate.current = now;
        }
      }
    };

    return () => sock.close();
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (status !== "running" || !ws.current || ws.current.readyState !== WebSocket.OPEN) return;

    // Throttle UI mouse emission to 20fps (50ms) to prevent server backend flooding
    const now = Date.now();
    if (now - lastMouseTx.current < 50) return;
    lastMouseTx.current = now;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = ~~((e.clientX - rect.left) * (1400 / rect.width));
    const y = ~~((e.clientY - rect.top) * (700 / rect.height));
    ws.current.send(JSON.stringify({ cmd: "move", x, y }));
  }, [status]);

  return (
    <div style={{ minHeight: "100vh", background: "#080b14", color: "#cdd6f4", padding: "20px 28px", fontFamily: "'Orbitron',monospace" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');`}</style>

      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 900, color: "#00d4ff" }}>RIS WIRELESS SIMULATOR</div>
          <div style={{ fontSize: 10, color: "#444" }}>UAV-MOUNTED RIS · LIVE CHANNEL PREDICTION</div>
        </div>
        <button
          disabled={!wsReady}
          onClick={() => ws.current?.send(JSON.stringify({ cmd: status === "running" ? "stop" : "start" }))}
          style={{
            padding: "10px 26px", borderRadius: 6,
            cursor: wsReady ? "pointer" : "not-allowed",
            background: !wsReady ? "#444" : (status === "running" ? "#ff4060" : "#00d4ff"),
            fontWeight: 700, opacity: wsReady ? 1 : 0.5
          }}>
          {!wsReady ? "CONNECTING..." : (status === "running" ? "■ STOP" : "▶ START")}
        </button>
      </div>

      {error && <div style={{ color: "#ff4060", fontSize: 12, marginBottom: 10 }}>⚠ {error}</div>}

      <RISSimulator frame={frame} onMouseMove={handleMouseMove} />

      {displayFrame && (
        <div style={{ display: "flex", gap: 20, marginTop: 18 }}>
          {/* URLLC Ambulance Metrics */}
          <div style={{ flex: 1, padding: 12, border: "1px solid rgba(255, 80, 80, 0.4)", borderRadius: 8, background: "rgba(255, 80, 80, 0.05)" }}>
            <div style={{ fontSize: 14, fontWeight: "bold", color: "#ff5050", marginBottom: 10 }}>🚑 URLLC (AMBULANCE)</div>
            <div style={{ display: "flex", gap: 10 }}>
              {[
                ["STRENGTH", displayFrame.power_db[0], " dB", "#00d4ff", "Beam Power - The raw intensity of the radio wave hitting the vehicle."],
                ["CLARITY", displayFrame.snr[0], " dB", "#00ff9d", "Signal-to-Noise Ratio (SNR) - How clear the channel is for high-speed data transfer."],
                ["DISTANCE", displayFrame.distance[0], " m", "#ffaa00", "Physical distance from the RIS relay to the vehicle."],
                ["DELAY", displayFrame.aocsi[0], " slots", "#ff4060", "Age of CSI - How old the Base Station's tracking data is. Higher age = flying blind!"]
              ].map(([l, v, u, c, t]) => (
                <div key={l} title={t} style={{ flex: 1, padding: "8px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: `1px solid ${c}20`, cursor: "help" }}>
                  <div style={{ fontSize: 9, color: "#555" }}>{l} ⓘ</div>
                  <div style={{ fontSize: 16, color: c }}>{v?.toFixed(1)}{u}</div>
                </div>
              ))}
            </div>
          </div>

          {/* eMBB Car Metrics */}
          <div style={{ flex: 1, padding: 12, border: "1px solid rgba(0, 212, 255, 0.4)", borderRadius: 8, background: "rgba(0, 212, 255, 0.05)" }}>
            <div style={{ fontSize: 14, fontWeight: "bold", color: "#00d4ff", marginBottom: 10 }}>🚗 eMBB (CAR)</div>
            <div style={{ display: "flex", gap: 10 }}>
              {[
                ["STRENGTH", displayFrame.power_db[1], " dB", "#00d4ff", "Beam Power - The raw intensity of the radio wave hitting the vehicle."],
                ["CLARITY", displayFrame.snr[1], " dB", "#00ff9d", "Signal-to-Noise Ratio (SNR) - How clear the channel is for high-speed data transfer."],
                ["DISTANCE", displayFrame.distance[1], " m", "#ffaa00", "Physical distance from the RIS relay to the vehicle."],
                ["DELAY", displayFrame.aocsi[1], " slots", "#ff4060", "Age of CSI - How old the Base Station's tracking data is. Higher age = flying blind!"]
              ].map(([l, v, u, c, t]) => (
                <div key={l} title={t} style={{ flex: 1, padding: "8px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: `1px solid ${c}20`, cursor: "help" }}>
                  <div style={{ fontSize: 9, color: "#555" }}>{l} ⓘ</div>
                  <div style={{ fontSize: 16, color: c }}>{v?.toFixed(1)}{u}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {frame && (
        <div style={{ marginTop: 20, padding: 15, borderRadius: 8, background: "rgba(0,0,0,0.5)", borderLeft: "4px solid #ff4060" }}>
          <div style={{ fontSize: 10, color: "#888", marginBottom: 4, fontWeight: "bold" }}>6G ALGORITHM LIVE ANALYSIS:</div>
          <div style={{ fontSize: 13, color: "#ddd", minHeight: "20px" }}>
            {analysisText}
          </div>
        </div>
      )}

      <div style={{ display: "flex", gap: 14, marginTop: 20 }}>
        <Sparkline data={powerH} color="rgb(0,212,255)" label="Ambulance Beam Strength" unit=" dB" />
        <Sparkline data={snrH} color="rgb(0,255,157)" label="Ambulance Signal Clarity" unit=" dB" />
      </div>
    </div>
  );
}