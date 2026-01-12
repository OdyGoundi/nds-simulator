import numpy as np
import matplotlib.pyplot as plt

SYM_FILE = "henon_heiles_trajectory.csv"
RK_FILE  = "rk45Henon.csv"

def load_csv_5cols(path):
    """
    Reads CSV with 5 columns: t,y1,y2,y3,y4 (header optional).
    Returns: t (N,), y (4,N) where y=[q1,q2,p1,p2] = [y1,y2,y3,y4]
    """
    # Try: header present
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        if data.ndim == 1:
            raise ValueError("Single row after skip_header=1")
        if data.shape[1] != 5:
            raise ValueError(f"Expected 5 columns, got {data.shape[1]}")
    except Exception:
        # Fallback: no header
        data = np.genfromtxt(path, delimiter=",", skip_header=0)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != 5:
            raise ValueError(f"{path}: expected 5 columns (t,y1,y2,y3,y4), got {data.shape[1]}")

    t = data[:, 0]
    y = data[:, 1:5].T  # (4, N)
    return t, y

def interp_y_to(t_src, y_src, t_target):
    y_itp = np.zeros((y_src.shape[0], t_target.size), dtype=float)
    for k in range(y_src.shape[0]):
        y_itp[k] = np.interp(t_target, t_src, y_src[k])
    return y_itp

def henon_heiles_hamiltonian(y):
    # y=[q1,q2,p1,p2]
    q1, q2, p1, p2 = y
    T = 0.5 * (p1**2 + p2**2)
    V = 0.5 * (q1**2 + q2**2) + (q1**2) * q2 - (1.0/3.0) * (q2**3)
    return T + V

def stats(arr):
    arr = np.asarray(arr)
    return np.max(np.abs(arr)), np.sqrt(np.mean(arr**2)), np.abs(arr[-1])

# --- Load
t_sym, y_sym = load_csv_5cols(SYM_FILE)
t_rk,  y_rk  = load_csv_5cols(RK_FILE)

# --- Align RK45 to Symplectic time grid (interpolation)
y_rk_on_sym = interp_y_to(t_rk, y_rk, t_sym)

# --- Deviations
labels = ["q1", "q2", "p1", "p2"]
dy = y_sym - y_rk_on_sym
dy_norm = np.linalg.norm(dy, axis=0)

print("\nState deviation summary (Symplectic - RK45 interpolated on Symplectic t-grid)\n")
for i, name in enumerate(labels):
    mx, rms, fin = stats(dy[i])
    print(f"{name:>2s}: max|Δ| = {mx:.6e} | RMS(Δ) = {rms:.6e} | |Δ(t_end)| = {fin:.6e}")

mx, rms, fin = stats(dy_norm)
print(f"\n||Δy||2(t): max = {mx:.6e} | RMS = {rms:.6e} | final = {fin:.6e}\n")

# --- Energy drift
H_sym = henon_heiles_hamiltonian(y_sym)
H_rk  = henon_heiles_hamiltonian(y_rk_on_sym)
dH_sym = H_sym - H_sym[0]
dH_rk  = H_rk  - H_rk[0]
dH_between = H_sym - H_rk

mx, rms, fin = stats(dH_sym)
print("Energy drift ΔH(t)=H(t)-H(0)\n")
print(f"Symplectic ΔH: max|.| = {mx:.6e} | RMS = {rms:.6e} | final = {fin:.6e}")
mx, rms, fin = stats(dH_rk)
print(f"RK45       ΔH: max|.| = {mx:.6e} | RMS = {rms:.6e} | final = {fin:.6e}")
mx, rms, fin = stats(dH_between)
print(f"H_sym - H_rk : max|.| = {mx:.6e} | RMS = {rms:.6e} | final = {fin:.6e}\n")

# --- Plots (project style: black + magenta)
c_sym, c_rk = "black", "magenta"

# 1) q1,q2,p1,p2 vs time
fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(t_sym, y_sym[i], color=c_sym, linewidth=1.0, label="Symplectic")
    ax.plot(t_sym, y_rk_on_sym[i], color=c_rk, linewidth=1.0, alpha=0.9, label="RK45 (interp)")
    ax.set_ylabel(labels[i])
    ax.grid(True, alpha=0.25)
axes[-1].set_xlabel("t")
axes[0].set_title("Henon–Heiles: q1,q2,p1,p2 vs time (Symplectic vs RK45)")
axes[0].legend(loc="best")
plt.tight_layout()
plt.show()

# 2) ||Δy||2 vs time
plt.figure(figsize=(10, 3.5))
plt.plot(t_sym, dy_norm, color=c_sym, linewidth=1.0)
plt.xlabel("t")
plt.ylabel("||Δy||2")
plt.title("Deviation over time: || y_symplectic(t) - y_rk45(t) ||2")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# 3) H(t)
plt.figure(figsize=(10, 3.5))
plt.plot(t_sym, H_sym, color=c_sym, linewidth=1.0, label="Symplectic")
plt.plot(t_sym, H_rk,  color=c_rk,  linewidth=1.0, alpha=0.9, label="RK45 (interp)")
plt.xlabel("t")
plt.ylabel("H")
plt.title("Hamiltonian H(t)")
plt.grid(True, alpha=0.25)
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# 4) ΔH(t)
plt.figure(figsize=(10, 3.5))
plt.plot(t_sym, dH_sym, color=c_sym, linewidth=1.0, label="Symplectic ΔH")
plt.plot(t_sym, dH_rk,  color=c_rk,  linewidth=1.0, alpha=0.9, label="RK45 ΔH (interp)")
plt.xlabel("t")
plt.ylabel("ΔH = H(t)-H(0)")
plt.title("Energy drift over time")
plt.grid(True, alpha=0.25)
plt.legend(loc="best")
plt.tight_layout()
plt.show()
