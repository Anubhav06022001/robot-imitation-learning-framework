import numpy as np
import pandas as pd

class TrajectoryLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []
        self.active = True

    def record(self, theta1, theta2, dtheta1, dtheta2, control):
        if not self.active:
            return
        self.data.append([
            theta1, theta2, dtheta1, dtheta2, control
        ])

    def stop(self):
        self.active = False

    def save(self, filename):
        cols = ["theta1", "theta2", "dtheta1", "dtheta2", "control"]
        df = pd.DataFrame(self.data, columns=cols)
        df.to_csv(filename, index=False)
        print(f"[TrajectoryLogger] Saved {len(df)} samples to {filename}")