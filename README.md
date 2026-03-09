<div align="center">
  <h1>📡 6G RIS Wireless Physics Simulator</h1>
  <p><strong>A Next-Generation AI-Driven Telecommunications Engine</strong></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" />
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
    <img src="https://img.shields.io/badge/MediaPipe-000000?style=for-the-badge&logo=google&logoColor=white" />
  </p>

  <p><i>Building the future of wireless connectivity: Merging Stochastic Physics, Calculus-Based Deep Learning, and Real-Time WebSockets into one massive physics simulator.</i></p>
</div>

---

## 📖 The Story: What exactly is this project?

Imagine it is the year 2035. You are the conductor of a massive **6G Cell Tower (Base Station)**. Your job is to blast hyper-fast internet/radio waves out into a city.

Normally, radio waves bounce randomly off buildings and get lost. But in this city, there is a drone flying in the sky carrying a **Reconfigurable Intelligent Surface (RIS)**. An RIS is basically a giant sci-fi "smart mirror" made of 64 tiny digital tiles. If you hit the RIS with a radio beam, you can instantly program its 64 tiles to physically rotate, bending and reflecting the radio wave perfectly down to anyone in the city!

**The Problem:**
You have a limited physical spotlight. You can only track and beam internet to **one** vehicle at a time. Right now, there are two vehicles driving below the drone:
1. 🚑 **An Ambulance (URLLC - Ultra-Reliable Low-Latency Communication):** It is carrying a critical patient and needs absolute, perfect, uninterrupted tracking data. (In this simulator, **YOU** drive the ambulance physically by waving your hand in front of your webcam!)
2. 🚗 **A Civilian eMBB Car (Enhanced Mobile Broadband):** This person is trying to download a 4K movie.

**The Solution:**
This project is an Artificial Intelligence Engine that continuously plays a tug-of-war. It uses a bleeding-edge mathematical algorithm called **Lyapunov Optimization** to decide: *"Do I aim the laser at the Car to maximize download speeds, or do I suddenly panic, drop the Car, and aim the laser back at the Ambulance because I haven't checked on it in 3 milliseconds?"*

---

## 🔬 How The "Brain" Actually Works (The Science)

This is not a simple script; it is a full-stack physics and calculus engine. Even if you've never written code before, here is the magic happening under the hood:

### 1. The Physics Engine (Stochastic Geometry)
Radio waves in the real world don't travel in straight lines; they scatter. This project mathematically proves standard wireless theories using the **Rayleigh Fading Distribution**. It calculates real physical world variables like the **Doppler Shift Effect** (how radio waves squash and stretch as the drone and cars drive fast). 

### 2. The Artificial Neural Network (Built entirely from scratch)
Instead of using training wheels like PyTorch or TensorFlow, the Artificial Intelligence tracking the vehicles in this codebase is a **Probabilistic Feed-Forward Neural Network written entirely in raw Numpy matrix calculus**. 
* **What it does:** It watches the radio echoes bouncing off the cars and tries to predict where they will drive next in absolute 3D space.
* **The "Uncertainty Bubble":** The AI is legally bound by a custom Calculus Loss Equation to admit when it is "guessing." On the screen, you will see a green bubble drawn around vehicles. If the AI is confused about where a car went, the bubble physically grows larger!

### 3. The 64-Tile RIS Mirror (Alternating Optimization)
To aim the physical radio beam, the backend has to solve a nearly impossible matrix algebra equation (Convex Optimization). It mathematically un-twists the phase distortion caused by the physical air, calculates the perfect reflection angles, and geometrically forces the 64 continuous mirrors to "snap" to 4 physical hardware constraints (0°, 90°, 180°, 270°). 

---

## � Getting Started (How to run it yourself!)

You need to run two completely separate worlds: The **Python Math Engine**, and the **React Dashboard**.

### 🛠️ Step 1: Install Dependencies
Ensure you have `Python 3.10+` and `Node.js` installed on your machine.
```bash
# Clone the repository and enter it
cd wireless_project

# Python Backend Preparation
python -m venv venv
# On Windows, activate with: .\venv\Scripts\activate
# On Mac/Linux, activate with: source venv/bin/activate
pip install fastapi uvicorn numpy opencv-python mediapipe websockets

# React Frontend Preparation
cd react-app
npm install
```

### ▶️ Step 2: Ignite the Simulation
You need **TWO** terminal windows.

**Terminal 1 (Start the Physics Math Server):**
```bash
cd wireless_project
.\venv\Scripts\activate
python server.py
```
*(The physics engine will boot up at lightning speed, binding a WebSocket to port 8005).*

**Terminal 2 (Start the Visual Dashboard):**
```bash
cd wireless_project/react-app
npm start
```
*(Your web browser will instantly pop open to `http://localhost:3000`).*

### 🎮 Step 3: Play God
1. **Hold up your hand!** The Python server will automatically turn on your webcam. It uses Google's MediaPipe Computer Vision to digitally track your index finger tip. Moving your finger physically drives the Ambulance 🚑 around the simulated city.
2. Watch as the AI frantically bounces the radio beam between the Ambulance and the Car, calculating Age-of-CSI and maximizing SNR metrics in real-time.

---

## �️ Codebase Tour (Where does the magic happen?)

Want to dig into the matrix code? Here is exactly where to look:

#### � The Python Backend
* 📂 **`server.py`:** The massive asynchronous heart of the project. It runs the heavy math block in a protected background thread, slicing the Numpy matrices into primitive JSON payloads and firing them to React at 30 frames per second over WebSockets. (It even has defensive safeguards—if the physics engine hallucinates an unsolvable matrix, `server.py` instantly strips the crash before it hits the UI!).
* 📂 **`app/simulator/channelv2.py`:** Go here to see the pure physical science. This is where variables like `pl_exp` (Path Loss Exponents) and Matrix Transposes `@` actually model the Doppler radio waves flying through the virtual air.
* � **`app/optimization/lyapunov_scheduler.py`:** The "AI Judge." It physically calculates the `diff = metric_pilot - metric_data` score to decide whether tracking the ambulance is more important than downloading a movie.
* 📂 **`app/models/ann.py`:** The massive hand-written Deep Learning Engine calculating the Calculus Chain Rule (`backward()` function) via matrix transposition.

#### ⚛️ The Javascript Frontend
* 📂 **`react-app/src/App.js`:** The React hooks dashboard. It acts as the "receptor," establishing the WebSocket, sorting the radio power metrics, and applying strict algorithmic throttling (`Date.now()`) to prevent your mouse from crashing the Python queue.
* 📂 **`react-app/src/RISSimulator.jsx`:** The GPU UI Renderer. It takes the gross raw matrix data and uses HTML5 Canvas `ctx.arc()` and Linear Interpolation to generate beautiful, buttery-smooth radar visualizations, uncertainty bubbles, and sweeping searchlight beams.

---
<div align="center">
  <i>Developed for cutting-edge IEEE Publication Simulation.</i>
</div>
