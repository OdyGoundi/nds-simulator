import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def phase_portrait(state_trajectory,
                   x_index=0, y_index=1, transient_steps=0,
                   xlabel="Y1", ylabel="Y2", title="Phase Portrait"):
    """
    Plots the phase portrait of a multi-dimensional trajectory:
    plots state[y_index] vs state[x_index] after removing transient steps.
    """
    # Convert to numpy array
    states = np.array(state_trajectory)

    # Remove transient part
    states_trimmed = states[:, transient_steps:]

    # Extract components for plotting
    x_component = states_trimmed[x_index]
    y_component = states_trimmed[y_index]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(x_component, y_component, "teal", linewidth=0.8)
    plt.xlabel(xlabel, color="black")
    plt.ylabel(ylabel, color="black")
    plt.title(title, color="black")

    ax = plt.gca()
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    ax.set_aspect("equal", adjustable="box")
    plt.show()
    
def animated_phase_portrait(state_trajectory,
                            x_index=0, y_index=1, transient_steps=0,
                            xlabel="Y1", ylabel="Y2",
                            title="Phase Portrait (animated)",
                            total_duration_seconds=15,
                            filename="phase_portrait.gif",
                            max_frames= None):
    """
    Create an animated phase portrait (GIF).
    The trajectory is drawn progressively over time.

    state_trajectory : array-like, shape (n_states, n_points)
        Solution from the integrator (e.g. sol.y or y_rk4).
    total_duration_seconds : float
        Total duration of the GIF in seconds (e.g. 15).
    filename : str
        Output GIF filename.
    max_frames : int
        Maximum number of frames used in the animation (controls file size).
    """
    max_frames = max_frames if max_frames is not None else total_duration_seconds*50
    # Convert and trim transient
    states = np.array(state_trajectory)
    states_trimmed = states[:, transient_steps:]

    x_component = states_trimmed[x_index]
    y_component = states_trimmed[y_index]

    n_points = x_component.size

    # Number of frames (capped)
    n_frames = min(n_points, max_frames)

    # Indices to use for each frame
    frame_indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    # Time per frame in ms
    interval_ms = (total_duration_seconds * 1000.0) / float(n_frames)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Axis labels & style
    ax.set_xlabel(xlabel, color="black")
    ax.set_ylabel(ylabel, color="black")
    ax.set_title(title, color="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    # --- Fixed frame that fits ALL the trajectory + small margin ---
    x_min = np.min(x_component)
    x_max = np.max(x_component)
    y_min = np.min(y_component)
    y_max = np.max(y_component)

    # Avoid zero span
    if x_max == x_min:
        x_max = x_min + 1e-6
    if y_max == y_min:
        y_max = y_min + 1e-6

    x_span = x_max - x_min
    y_span = y_max - y_min

    margin_factor = 0.05  # 5% margin like a nice frame
    x_margin = margin_factor * x_span
    y_margin = margin_factor * y_span

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    # --------------------------------------------------------------

    ax.set_aspect("equal", adjustable="box")

    # Line that grows with time
    trajectory_line, = ax.plot([], [], "teal", linewidth=0.8)

    def init():
        trajectory_line.set_data([], [])
        return (trajectory_line,)

    def update(frame):
        idx = frame_indices[frame]
        x_data = x_component[:idx + 1]
        y_data = y_component[:idx + 1]
        trajectory_line.set_data(x_data, y_data)
        return (trajectory_line,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=interval_ms,
        blit=True,
    )

    anim.save(filename, writer="pillow", fps=n_frames / total_duration_seconds)
    plt.close(fig)
    print(f"Saved animated phase portrait to '{filename}'")
