import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from scipy.stats import entropy

width, height = 600, 600
habit = 0.90
tychism = 0.08

x, y = np.meshgrid(np.linspace(-1.6, 1.6, width), np.linspace(-1.6, 1.6, height))
r = np.sqrt(x**2 + y**2)
field = np.exp(-r * 4.0)
field = np.clip(field, 0.0, 1.0)

entropy_history = []

fig = plt.figure(figsize=(8, 8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
im = ax.imshow(field, cmap='magma', vmin=0, vmax=1.4)
ax.axis('off')
plt.tight_layout(pad=0)

def update(frame):
    global field, entropy_history

    avg = np.zeros_like(field)
    avg[1:-1, 1:-1] = 0.25 * (
        field[0:-2, 1:-1] + field[2:, 1:-1] +
        field[1:-1, 0:-2] + field[1:-1, 2:]
    )

    # Non-linear habit
    mix = (1 - habit) * field + habit * avg * (1.3 - field * 0.3)

    # Noise boosted at low entropy
    flat = field.flatten()
    hist, _ = np.histogram(flat, bins=300, density=True)
    curr_ent = entropy(hist + 1e-12)
    noise_scale = tychism * 0.015 * (5.0 - curr_ent)  # stronger when entropy low
    noise = (np.random.rand(height, width) - 0.5) * noise_scale

    field = mix + noise

    # Self-reinforcing growth
    center_growth = 0.00015 * (1.0 - r) * field
    field += center_growth

    field = np.clip(field, 0.0, 1.45)

    entropy_history.append(curr_ent)

    # Adaptive cmap for better color contrast
    vmin, vmax = field.min(), field.max()
    im.set_clim(vmin, vmax)

    ax.set_title(f"Frame: {frame} | Entropie: {curr_ent:.3f}", fontsize=12, color='white')

    im.set_array(field)
    return [im]

ani = FuncAnimation(fig, update, interval=40, blit=True, cache_frame_data=False)

root = tk.Tk()
root.title("Steuerung")
root.geometry("350x250")
root.attributes('-topmost', True)

tk.Label(root, text="Klicke hier, dann drücke:").pack(pady=10)

tk.Label(root, text="Q/ESC = Beenden + Plot").pack(anchor='w')
tk.Label(root, text="H = Habit + (Ordnung ↑, Entropie ↓)").pack(anchor='w')
tk.Label(root, text="J = Habit -").pack(anchor='w')
tk.Label(root, text="T = Tychism +").pack(anchor='w')
tk.Label(root, text="Y = Tychism -").pack(anchor='w')
tk.Label(root, text="R = Reset").pack(anchor='w')

def on_key(event):
    global habit, tychism, field
    key = event.keysym.lower()
    if key in ('q', 'escape'):
        plt.figure(figsize=(8, 4))
        plt.plot(entropy_history)
        plt.title("Entropie-Entwicklung")
        plt.xlabel("Frame")
        plt.ylabel("Shannon-Entropie")
        plt.grid(True)
        plt.savefig(r"C:\github\CS-Peirce\output\entropy_plot.png")
        print("Plot gespeichert: entropy_plot.png")
        root.quit()
    elif key == 'h':
        habit = min(0.98, habit + 0.015)
        print(f"Habit: {habit:.2f}")
    elif key == 'j':
        habit = max(0.45, habit - 0.015)
        print(f"Habit: {habit:.2f}")
    elif key == 't':
        tychism = min(0.45, tychism + 0.015)
        print(f"Tychism: {tychism:.2f}")
    elif key == 'y':
        tychism = max(0.0, tychism - 0.015)
        print(f"Tychism: {tychism:.2f}")
    elif key == 'r':
        print("Reset")
        field = np.exp(-r * 4.0) + np.random.uniform(-0.04, 0.04, (height, width))
        field = np.clip(field, 0.0, 1.0)
        im.set_array(field)

root.bind('<Key>', on_key)

plt.show(block=False)
root.mainloop()