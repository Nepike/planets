import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.style.use('dark_background')


class Body:
	def __init__(self, mass, position, velocity):
		self.mass = mass
		self.position = np.array(position, dtype=np.float64)
		self.velocity = np.array(velocity, dtype=np.float64)
		self.acceleration = np.zeros(3, dtype=np.float64)


def compute_accelerations(bodies, G):
	accs = [np.zeros(3) for _ in bodies]
	for i in range(len(bodies)):
		for j in range(len(bodies)):
			if i != j:
				r_vec = bodies[j].position - bodies[i].position
				r_mag = np.linalg.norm(r_vec)
				if r_mag != 0:
					accs[i] += G * bodies[j].mass * r_vec / r_mag ** 3
	return accs


def velocity_verlet_step(bodies, dt, G):
	for i, body in enumerate(bodies):
		body.position += body.velocity * dt + 0.5 * body.acceleration * dt ** 2
	new_accs = compute_accelerations(bodies, G)
	for i, body in enumerate(bodies):
		body.velocity += 0.5 * (body.acceleration + new_accs[i]) * dt
		body.acceleration = new_accs[i]


class App:
	def __init__(self, root):
		self.root = root
		self.root.title("Симулятор N-тел")
		self.bodies = []
		self.entries = []

		main_frame = ttk.Frame(root)
		main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

		left_frame = ttk.LabelFrame(main_frame, text="Управление телами")
		left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

		right_frame = ttk.LabelFrame(main_frame, text="Параметры симуляции")
		right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

		self.canvas = tk.Canvas(left_frame)
		self.scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.canvas.yview)
		self.scrollable_frame = ttk.Frame(self.canvas)

		self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
		self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
		self.canvas.configure(yscrollcommand=self.scrollbar.set)

		self.canvas.pack(side="left", fill="both", expand=True)
		self.scrollbar.pack(side="right", fill="y")

		headers = ["Mass", "X", "Y", "Z", "VX", "VY", "VZ", ""]
		for col, text in enumerate(headers):
			ttk.Label(self.scrollable_frame, text=text, padding=3).grid(row=0, column=col)

		self.add_entry_row([15, 1, 0, 0, 0, 1.5, 0])
		self.add_entry_row([4, -0.5, 0.866, 0, 0, -1.0, 0.5])
		self.add_entry_row([4, -0.5, -0.866, 0, 0, -0.5, -0.5])

		btn_frame = ttk.Frame(left_frame)
		btn_frame.pack(pady=5)
		ttk.Button(btn_frame, text="Добавить тело", command=self.add_entry_row).pack(side=tk.LEFT, padx=2)
		ttk.Button(btn_frame, text="Сбросить всё", command=self.reset_entries).pack(side=tk.LEFT, padx=2)

		# Параметры симуляции
		params = [
			("Гравитационная постоянная (G)", "0.5"),
			("Шаг времени (dt)", "0.01"),
			("Общее время", "10")
		]

		self.G_entry = ttk.Entry(right_frame, width=10)
		self.dt_entry = ttk.Entry(right_frame, width=10)
		self.total_time_entry = ttk.Entry(right_frame, width=10)

		for i, (label, val) in enumerate(params):
			ttk.Label(right_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=2)

		self.G_entry.grid(row=0, column=1, padx=5, pady=2)
		self.G_entry.insert(0, "0.5")
		self.dt_entry.grid(row=1, column=1, padx=5, pady=2)
		self.dt_entry.insert(0, "0.01")
		self.total_time_entry.grid(row=2, column=1, padx=5, pady=2)
		self.total_time_entry.insert(0, "10")

		# Опции визуализации
		self.record_gif = tk.BooleanVar()
		self.track_com_var = tk.BooleanVar(value=True)

		ttk.Checkbutton(right_frame, text="Сохранить как GIF", variable=self.record_gif).grid(row=3, column=0, columnspan=2,
		                                                                                      pady=5)
		ttk.Checkbutton(right_frame, text="Отслеживать центр масс", variable=self.track_com_var).grid(row=4, column=0,
		                                                                                              columnspan=2, pady=5)

		ttk.Label(right_frame, text="Имя файла:").grid(row=5, column=0, sticky="w", padx=5)
		self.gif_name = ttk.Entry(right_frame, width=15)
		self.gif_name.grid(row=5, column=1, padx=5, pady=2)
		self.gif_name.insert(0, "animation.gif")

		ttk.Button(right_frame, text="Начать симуляцию", command=self.run_simulation).grid(row=6, column=0, columnspan=2,
		                                                                                   pady=10)

	def add_entry_row(self, default=None):
		row = len(self.entries) + 1
		entry_row = []
		for i in range(7):
			e = ttk.Entry(self.scrollable_frame, width=7)
			e.grid(row=row, column=i, padx=2, pady=2)
			if default:
				e.insert(0, str(default[i]))
			entry_row.append(e)

		btn = ttk.Button(
			self.scrollable_frame,
			text="×",
			width=3,
			command=lambda r=row: self.remove_entry_row(r - 1)
		)
		btn.grid(row=row, column=7, padx=2)
		entry_row.append(btn)
		self.entries.append(entry_row)

	def remove_entry_row(self, row):
		for widget in self.entries[row]:
			widget.destroy()
		del self.entries[row]
		for r in range(row, len(self.entries)):
			for c in range(7):
				self.entries[r][c].grid(row=r + 1, column=c)
			self.entries[r][7].grid(row=r + 1, column=7)

	def reset_entries(self):
		for entry_row in self.entries:
			for widget in entry_row:
				widget.destroy()
		self.entries.clear()

	def run_simulation(self):
		try:
			G = float(self.G_entry.get())
			dt = float(self.dt_entry.get())
			total_time = float(self.total_time_entry.get())
			steps = int(total_time / dt)

			self.bodies = []
			for row in self.entries:
				mass = float(row[0].get())
				pos = [float(row[i].get()) for i in range(1, 4)]
				vel = [float(row[i].get()) for i in range(4, 7)]
				self.bodies.append(Body(mass, pos, vel))

			if len(self.bodies) < 1:
				raise ValueError("Добавьте хотя бы одно тело")

		except Exception as e:
			messagebox.showerror("Ошибка ввода", f"Некорректные параметры:\n{e}")
			return

		initial_accs = compute_accelerations(self.bodies, G)
		for i, body in enumerate(self.bodies):
			body.acceleration = initial_accs[i]

		trajectories = [np.zeros((steps, 3)) for _ in self.bodies]
		speeds = [np.zeros(steps) for _ in self.bodies]

		for step in range(steps):
			for i, body in enumerate(self.bodies):
				trajectories[i][step] = body.position
				speeds[i][step] = np.linalg.norm(body.velocity)
			velocity_verlet_step(self.bodies, dt, G)

		# Вычисление параметров для отслеживания (только если включено)
		centers_of_mass = []
		max_distances = []
		if self.track_com_var.get():
			total_mass = sum(body.mass for body in self.bodies)
			for step in range(steps):
				com = np.zeros(3)
				for i in range(len(self.bodies)):
					com += self.bodies[i].mass * trajectories[i][step]
				com /= total_mass
				centers_of_mass.append(com)

				max_dist = 0
				for i in range(len(self.bodies)):
					pos = trajectories[i][step]
					dist = np.linalg.norm(pos - com)
					max_dist = max(max_dist, dist)
				max_distances.append(max_dist)

		self.visualize(trajectories, speeds, centers_of_mass, max_distances)

	def visualize(self, trajectories, speeds, centers_of_mass, max_distances):
		fig = plt.figure(figsize=(12, 9))
		ax = fig.add_subplot(111, projection='3d')
		ax.set_facecolor('black')
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")
		ax.set_title("Траектории тел", fontsize=14)

		if not self.track_com_var.get():
			ax.set_xlim(-2, 2)
			ax.set_ylim(-2, 2)
			ax.set_zlim(-2, 2)

		cmap = cm.get_cmap('plasma')
		tail_length = 200
		lines = [ax.plot([], [], [], lw=2)[0] for _ in self.bodies]
		dots = [ax.plot([], [], [], 'o', markersize=6)[0] for _ in self.bodies]

		# Параметры для анимации с отслеживанием
		buffer = 0.5
		min_radius = 1.0
		prev_com = None
		smoothing_factor = 0.3

		def update(frame):
			nonlocal prev_com
			if self.track_com_var.get() and centers_of_mass:
				current_com = centers_of_mass[frame]
				current_max_dist = max(max_distances[frame], min_radius) + buffer

				if prev_com is None:
					prev_com = current_com.copy()
				else:
					prev_com = prev_com * (1 - smoothing_factor) + current_com * smoothing_factor

				ax.set_xlim(prev_com[0] - current_max_dist, prev_com[0] + current_max_dist)
				ax.set_ylim(prev_com[1] - current_max_dist, prev_com[1] + current_max_dist)
				ax.set_zlim(prev_com[2] - current_max_dist, prev_com[2] + current_max_dist)

			for i in range(len(self.bodies)):
				start = max(0, frame - tail_length)
				x = trajectories[i][start:frame, 0]
				y = trajectories[i][start:frame, 1]
				z = trajectories[i][start:frame, 2]
				speed_norm = np.clip(speeds[i][frame] / np.max(speeds[i]), 0, 1)
				color = cmap(speed_norm)
				lines[i].set_data(x, y)
				lines[i].set_3d_properties(z)
				lines[i].set_color(color)
				dots[i].set_data([trajectories[i][frame, 0]], [trajectories[i][frame, 1]])
				dots[i].set_3d_properties([trajectories[i][frame, 2]])
				dots[i].set_color(color)
			return lines + dots

		ani = FuncAnimation(fig, update, frames=len(trajectories[0]), interval=10, blit=False)

		if self.record_gif.get():
			try:
				filename = self.gif_name.get() or "animation.gif"
				ani.save(filename, writer=PillowWriter(fps=30))
				messagebox.showinfo("Сохранено", f"Анимация сохранена как {filename}")
			except Exception as e:
				messagebox.showerror("Ошибка", f"Не удалось сохранить GIF:\n{e}")

		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	root = tk.Tk()
	root.geometry("900x600")
	App(root)
	root.mainloop()