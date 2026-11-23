STUDENT B - CLEAN VS CODE PYTHON PROJECT
========================================

This ZIP intentionally does not contain a .venv folder. Virtual environments
are specific to your computer and must be created locally.

FOLDER STRUCTURE
  simulation/run_platoon.py      Stage 1 leader-only simulation test
  generate_student_b_graphs.py   Plot real six-follower results from a CSV
  results/                      Outputs from the leader test
  figures/                      Future report graphs

IN VS CODE
1. Extract the ZIP. Open the Student_B_Clean_Project folder in VS Code.
2. Terminal > New Terminal. Make sure terminal path ends in Student_B_Clean_Project.
3. Run the following commands ONE AT A TIME:

   py -3.10 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib
   .\.venv\Scripts\python.exe .\simulation\run_platoon.py

If 'py -3.10' is not available, try 'py -m venv .venv'.

EXPECTED OUTPUT
   SUCCESS: leader trajectory created
   Final leader velocity: 6.000 m/s
   results/leader_velocity.png
   results/leader_reference.csv

IMPORTANT: This first script simulates ONLY the leader trajectory. It does
NOT simulate the six followers, FTRE, or STISMC. It deliberately does not
create a misleading results.csv file. The graph generator requires real
six-vehicle simulation results with columns e1..e6 and others. We will add
the validated full model in the next stage.

Once a validated results.csv exists, run:
   .\.venv\Scripts\python.exe .\generate_student_b_graphs.py --data results.csv --out figures

All plots are black on white.
