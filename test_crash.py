import time
import traceback
import sys
from server import SimState, simulation_loop, state

# Mock the WebSocket startup sequence
state.running = True

try:
    print("Testing simulation loop...")
    simulation_loop()
    print("Simulation loop ended gracefully!")
except Exception as e:
    print("FATAL UNCAUGHT:", e)
    traceback.print_exc()
