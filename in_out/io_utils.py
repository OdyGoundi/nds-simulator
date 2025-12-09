import numpy as np

def save_trajectory_csv(filename, t, y):
    """
    Save the solution (t, y1, y2, y3) to CSV.
    """
    data = np.column_stack((t, y.T))
    np.savetxt(filename, data, delimiter=",", header="t,y1,y2,y3", comments="")