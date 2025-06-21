import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageOps
from collections import deque
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PathfindingVisualizer:
    # Color constants
    START_COLOR = 'green'
    GOAL_COLOR = 'red'
    BFS_VISITED_COLOR = (173, 216, 230, 150)  # Light blue with transparency
    DFS_VISITED_COLOR = (144, 238, 144, 150)  # Light green with transparency
    BFS_PATH_COLOR = 'blue'
    DFS_PATH_COLOR = 'green'
    OBSTACLE_COLOR = (0, 0, 0, 200)  # Black with transparency
    GRID_LINE_COLOR = 'gray'
    
    def __init__(self, root):
        self.root = root
        self.root.title("Visualisasi Pencarian Jalur BFS vs DFS")
        self.root.geometry("1400x900")
        
        # Grid settings
        self.grid_size = 30
        self.cell_size = 20
        
        # Image variables
        self.original_image = None
        self.background_image = None
        self.map_grid = None
        self.display_image = None
        
        # Positions
        self.start_pos = None
        self.goal_pos = None
        
        # Simulation
        self.is_simulating = False
        self.simulation_speed = 100  # ms
        self.current_algorithm = None
        self.visited_nodes = set()
        self.frontier_nodes = deque()
        self.current_path = []
        self.solution_path = []
        
        # Results
        self.bfs_results = {'time': 0, 'nodes': 0, 'path_length': 0}
        self.dfs_results = {'time': 0, 'nodes': 0, 'path_length': 0}
        
        # Obstacle settings
        self.obstacle_mode = False
        self.obstacle_percentage = 0.2  # Default 20% obstacles
        
        # Setup GUI
        self.setup_gui()
        self.initialize_grid()
    
    def setup_gui(self):
        """Setup the main GUI components"""
        self.setup_control_frame()
        self.setup_visualization_frame()
        self.setup_info_frame()
    
    def setup_control_frame(self):
        """Setup the control panel frame"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Image buttons
        tk.Button(control_frame, text="Muat Gambar Latar", command=self.load_background_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Buat Grid", command=self.initialize_grid).pack(side=tk.LEFT, padx=5)
        
        # Position buttons
        self.start_btn = tk.Button(control_frame, text="Atur Titik Awal", state=tk.NORMAL,
                                  command=lambda: self.set_point("start"))
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.goal_btn = tk.Button(control_frame, text="Atur Titik Tujuan", state=tk.NORMAL,
                                 command=lambda: self.set_point("goal"))
        self.goal_btn.pack(side=tk.LEFT, padx=5)
        
        # Obstacle controls
        self.setup_obstacle_controls(control_frame)
        
        # Simulation controls
        self.simulate_btn = tk.Button(control_frame, text="Jalankan Kedua Algoritma", state=tk.NORMAL,
                                    command=self.run_both_algorithms)
        self.simulate_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Reset", command=self.reset_simulation).pack(side=tk.LEFT, padx=5)
        
        # Speed control
        self.setup_speed_control(control_frame)
        
        # Comparison button
        tk.Button(control_frame, text="Tampilkan Peta Perbandingan", 
                 command=self.show_comparison_map).pack(side=tk.LEFT, padx=5)
    
    def setup_obstacle_controls(self, parent_frame):
        """Setup obstacle-related controls"""
        self.obstacle_btn = tk.Button(parent_frame, text="Mode Rintangan", 
                                     command=self.toggle_obstacle_mode)
        self.obstacle_btn.pack(side=tk.LEFT, padx=5)
        
        obstacle_frame = tk.Frame(parent_frame)
        obstacle_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(obstacle_frame, text="Persen Rintangan:").pack()
        self.obstacle_slider = tk.Scale(obstacle_frame, from_=0, to=50, orient=tk.HORIZONTAL,
                                      command=self.update_obstacle_percentage)
        self.obstacle_slider.set(self.obstacle_percentage * 100)
        self.obstacle_slider.pack()
        
        tk.Button(parent_frame, text="Rintangan Acak", command=self.generate_random_obstacles).pack(side=tk.LEFT, padx=5)
        tk.Button(parent_frame, text="Hapus Rintangan", command=self.clear_obstacles).pack(side=tk.LEFT, padx=5)
    
    def setup_speed_control(self, parent_frame):
        """Setup speed control slider"""
        speed_frame = tk.Frame(parent_frame)
        speed_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(speed_frame, text="Kecepatan:").pack()
        self.speed_slider = tk.Scale(speed_frame, from_=10, to=500, orient=tk.HORIZONTAL,
                                   command=self.set_simulation_speed)
        self.speed_slider.set(self.simulation_speed)
        self.speed_slider.pack()
    
    def setup_visualization_frame(self):
        """Setup the visualization area with map and graphs"""
        vis_frame = tk.Frame(self.root)
        vis_frame.pack(fill=tk.BOTH, expand=True)
        
        # Map canvas
        self.map_canvas = tk.Canvas(vis_frame, bg='white', 
                                   width=self.grid_size*self.cell_size, 
                                   height=self.grid_size*self.cell_size)
        self.map_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Graph frame
        graph_frame = tk.Frame(vis_frame, width=400)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig, (self.ax, self.complexity_ax) = plt.subplots(2, 1, figsize=(5, 8), dpi=100, gridspec_kw={'height_ratios': [3, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_info_frame(self):
        """Setup the information frame at the bottom"""
        info_frame = tk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.bfs_label = tk.Label(info_frame, text="BFS: Belum dijalankan", font=('Arial', 10), fg='blue')
        self.bfs_label.pack(side=tk.LEFT, padx=10)
        
        self.dfs_label = tk.Label(info_frame, text="DFS: Belum dijalankan", font=('Arial', 10), fg='green')
        self.dfs_label.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(info_frame, text="Status: Siap", font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=10)
    
    def load_background_image(self):
        """Load an image as grid background"""
        file_path = filedialog.askopenfilename(filetypes=[("File gambar", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.background_image = self.original_image.resize(
                    (self.grid_size*self.cell_size, self.grid_size*self.cell_size)
                )
                self.draw_grid()
                self.status_label.config(text=f"Status: Gambar latar dimuat - {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat gambar: {str(e)}")
    
    def set_simulation_speed(self, value):
        """Set simulation speed"""
        self.simulation_speed = int(value)
    
    def initialize_grid(self):
        """Initialize a 30x30 grid"""
        self.grid_size = 30
        self.map_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.start_pos = None
        self.goal_pos = None
        self.draw_grid()
        self.status_label.config(text="Status: Grid diinisialisasi")
    
    def toggle_obstacle_mode(self):
        """Toggle obstacle placement mode"""
        self.obstacle_mode = not self.obstacle_mode
        if self.obstacle_mode:
            self.obstacle_btn.config(bg='red')
            self.status_label.config(text="Status: Mode rintangan - klik sel untuk menambah/menghapus rintangan")
        else:
            self.obstacle_btn.config(bg='SystemButtonFace')
            self.status_label.config(text="Status: Siap")
    
    def update_obstacle_percentage(self, value):
        """Update obstacle percentage"""
        self.obstacle_percentage = int(value) / 100
    
    def generate_random_obstacles(self):
        """Generate random obstacles based on percentage"""
        if self.map_grid is None:
            return
            
        total_cells = self.grid_size * self.grid_size
        obstacle_count = int(total_cells * self.obstacle_percentage)
        
        # Clear existing obstacles (except start and goal)
        self.clear_obstacles(keep_start_goal=True)
        
        # Add random obstacles
        for _ in range(obstacle_count):
            while True:
                x = random.randint(0, self.grid_size-1)
                y = random.randint(0, self.grid_size-1)
                if (x, y) != self.start_pos and (x, y) != self.goal_pos:
                    self.map_grid[y][x] = 1
                    break
        
        self.draw_grid()
        self.status_label.config(text=f"Status: Rintangan acak dibuat ({int(self.obstacle_percentage*100)}%)")
    
    def clear_obstacles(self, keep_start_goal=False):
        """Clear all obstacles (optionally keeping start and goal positions)"""
        if self.map_grid is None:
            return
            
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if keep_start_goal and ((x, y) == self.start_pos or (x, y) == self.goal_pos):
                    continue
                self.map_grid[y][x] = 0
        
        self.draw_grid()
        self.status_label.config(text="Status: Semua rintangan dihapus")
    
    def set_point(self, point_type):
        """Set start or goal position"""
        self.map_canvas.bind("<Button-1>", lambda e: self.handle_click(e, point_type))
        status_text = f"Status: Klik pada grid untuk menetapkan {point_type}"
        if point_type == "start":
            status_text += " (hijau)"
        else:
            status_text += " (merah)"
        self.status_label.config(text=status_text)
    
    def handle_click(self, event, point_type):
        """Handle mouse clicks on grid"""
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if self.obstacle_mode:
                # Toggle obstacle
                if (x, y) != self.start_pos and (x, y) != self.goal_pos:
                    self.map_grid[y][x] = 1 - self.map_grid[y][x]
            else:
                # Set start or goal position
                if self.map_grid[y][x] == 0:  # Only allow empty cells
                    if point_type == "start":
                        self.start_pos = (x, y)
                    else:
                        self.goal_pos = (x, y)
            
            self.draw_grid()
        
        self.map_canvas.unbind("<Button-1>")
        self.status_label.config(text="Status: Siap")
        self.obstacle_mode = False
        self.obstacle_btn.config(bg='SystemButtonFace')
    
    def run_both_algorithms(self):
        """Run both algorithms and compare results"""
        if not self.start_pos or not self.goal_pos:
            messagebox.showwarning("Peringatan", "Harap tetapkan posisi awal dan tujuan")
            return
        
        # Disable controls during simulation
        self.toggle_controls(False)
        self.status_label.config(text="Status: Menjalankan BFS...")
        self.root.update()
        
        # Reset previous results
        self.bfs_results = {'time': 0, 'nodes': 0, 'path_length': 0}
        self.dfs_results = {'time': 0, 'nodes': 0, 'path_length': 0}
        
        # Run BFS with step-by-step visualization
        self.bfs_path, self.bfs_visited = self.run_algorithm_with_visualization("BFS")
        
        # Add delay between algorithms
        time.sleep(1)
        self.status_label.config(text="Status: Menjalankan DFS...")
        self.root.update()
        time.sleep(1)
        
        # Run DFS with step-by-step visualization
        self.dfs_path, self.dfs_visited = self.run_algorithm_with_visualization("DFS")
        
        # Re-enable controls
        self.toggle_controls(True)
        
        # Update result labels
        self.update_result_labels()
        
        # Update comparison chart
        self.update_comparison_chart()
        
        # Show comparison map
        self.show_comparison_map()
        
        self.status_label.config(text="Status: Selesai menjalankan kedua algoritma")
    
    def toggle_controls(self, enable):
        """Enable or disable control buttons"""
        state = tk.NORMAL if enable else tk.DISABLED
        self.start_btn.config(state=state)
        self.goal_btn.config(state=state)
        self.simulate_btn.config(state=state)
        self.obstacle_btn.config(state=state)
    
    def run_algorithm_with_visualization(self, algorithm):
        """Run algorithm with step-by-step visualization"""
        if not self.start_pos or not self.goal_pos:
            return None, set()
        
        start = (self.start_pos[1], self.start_pos[0])  # (y, x)
        goal = (self.goal_pos[1], self.goal_pos[0])
        
        visited = set()
        frontier = deque()
        parent = {}
        
        frontier.append(start)
        visited.add(start)
        parent[start] = None
        
        found = False
        
        # Start timing before the main loop
        start_time = time.perf_counter()
        
        # Initial visualization
        self.visualize_step(algorithm, start, None, frontier, visited, parent)
        self.root.update()
        
        while frontier:
            if algorithm == "BFS":
                current = frontier.popleft()
            else:  # DFS
                current = frontier.pop()
            
            # Visualize current node being explored
            self.visualize_step(algorithm, current, None, frontier, visited, parent)
            self.root.update()
            time.sleep(self.simulation_speed / 1000)
            
            if current == goal:
                found = True
                break
            
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    if self.map_grid[ny][nx] == 0 and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        frontier.append((ny, nx))
                        parent[(ny, nx)] = current
                        
                        # Visualize new node being added to frontier
                        self.visualize_step(algorithm, current, (ny, nx), frontier, visited, parent)
                        self.root.update()
                        time.sleep(self.simulation_speed / 1000)
        
        # Calculate execution time
        exec_time = time.perf_counter() - start_time
        
        # Reconstruct path if found
        path = []
        if found:
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            
            # Visualize final path
            self.visualize_step(algorithm, None, None, frontier, visited, parent, path)
            self.root.update()
        else:
            messagebox.showinfo("Informasi", f"{algorithm}: Tidak ditemukan jalur ke tujuan")
        
        # Store results - path length is now correctly calculated as steps between nodes
        path_length = len(path) - 1 if path else 0  # Subtract 1 to get number of steps between nodes
        
        if algorithm == "BFS":
            self.bfs_results = {
                'time': exec_time,
                'nodes': len(visited),
                'path_length': path_length
            }
        else:
            self.dfs_results = {
                'time': exec_time,
                'nodes': len(visited),
                'path_length': path_length
            }
            if path:
                messagebox.showinfo("Informasi DFS", "Catatan: Jalur yang ditemukan DFS tidak dijamin yang terpendek")
        
        return path, visited
    
    def show_comparison_map(self):
        """Show a comparison map with both paths and visited nodes"""
        if not hasattr(self, 'bfs_path') or not hasattr(self, 'dfs_path'):
            messagebox.showwarning("Peringatan", "Harap jalankan kedua algoritma terlebih dahulu")
            return
        
        # Create new image for drawing
        if self.background_image:
            img = self.background_image.copy()
        else:
            img = Image.new('RGB', (self.grid_size*self.cell_size, self.grid_size*self.cell_size), 'white')
            
        # Convert to RGBA for transparency support
        img = img.convert('RGBA')
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines (only if no background image)
        if not self.background_image:
            self.draw_grid_lines(draw)
        
        # Draw obstacles
        self.draw_obstacles(img)
        
        # Draw visited nodes for both algorithms with reduced opacity
        self.draw_visited_nodes_comparison(draw, self.bfs_visited, self.dfs_visited)
        
        # Draw both paths
        if self.bfs_path:
            self.draw_path_comparison(draw, self.bfs_path, 'BFS')
        if self.dfs_path:
            self.draw_path_comparison(draw, self.dfs_path, 'DFS')
        
        # Draw start and goal positions
        self.draw_start_goal_positions(img)
        
        # Create a new window for comparison
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Peta Perbandingan BFS vs DFS")
        
        # Create canvas for comparison map
        comparison_canvas = tk.Canvas(comparison_window, 
                                     width=self.grid_size*self.cell_size, 
                                     height=self.grid_size*self.cell_size)
        comparison_canvas.pack()
        
        # Add legend
        legend_frame = tk.Frame(comparison_window)
        legend_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # BFS legend
        tk.Label(legend_frame, text="BFS", fg='blue').pack(side=tk.LEFT, padx=10)
        tk.Canvas(legend_frame, width=20, height=20, bg='lightblue').pack(side=tk.LEFT)
        tk.Label(legend_frame, text="Node Dikunjungi").pack(side=tk.LEFT, padx=5)
        
        # DFS legend
        tk.Label(legend_frame, text="DFS", fg='green').pack(side=tk.LEFT, padx=10)
        tk.Canvas(legend_frame, width=20, height=20, bg='lightgreen').pack(side=tk.LEFT)
        tk.Label(legend_frame, text="Node Dikunjungi").pack(side=tk.LEFT, padx=5)
        
        # Path legend
        path_legend_frame = tk.Frame(comparison_window)
        path_legend_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # BFS path legend
        tk.Label(path_legend_frame, text="Jalur BFS", fg='blue').pack(side=tk.LEFT, padx=10)
        canvas = tk.Canvas(path_legend_frame, width=50, height=20)
        canvas.pack(side=tk.LEFT)
        canvas.create_line(5, 10, 45, 10, fill='blue', width=3)
        
        # DFS path legend
        tk.Label(path_legend_frame, text="Jalur DFS", fg='green').pack(side=tk.LEFT, padx=10)
        canvas = tk.Canvas(path_legend_frame, width=50, height=20)
        canvas.pack(side=tk.LEFT)
        canvas.create_line(5, 10, 45, 10, fill='green', width=1, dash=(4,4))
        
        # Update display
        img_tk = ImageTk.PhotoImage(img)
        comparison_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        comparison_canvas.image = img_tk
    
    def draw_visited_nodes_comparison(self, draw, bfs_visited, dfs_visited):
        """Draw visited nodes for both algorithms with reduced opacity"""
        # Draw BFS visited nodes with reduced opacity
        bfs_color = (self.BFS_VISITED_COLOR[0], self.BFS_VISITED_COLOR[1], 
                     self.BFS_VISITED_COLOR[2], self.BFS_VISITED_COLOR[3]//2)
        
        # Draw DFS visited nodes with reduced opacity
        dfs_color = (self.DFS_VISITED_COLOR[0], self.DFS_VISITED_COLOR[1], 
                     self.DFS_VISITED_COLOR[2], self.DFS_VISITED_COLOR[3]//2)
        
        # Draw all visited nodes
        for y, x in bfs_visited.union(dfs_visited):
            if (y, x) != (self.start_pos[1], self.start_pos[0]) and (y, x) != (self.goal_pos[1], self.goal_pos[0]):
                # Determine which color to use (blend if visited by both)
                if (y, x) in bfs_visited and (y, x) in dfs_visited:
                    # Blend colors for nodes visited by both
                    color = (
                        (bfs_color[0] + dfs_color[0]) // 2,
                        (bfs_color[1] + dfs_color[1]) // 2,
                        (bfs_color[2] + dfs_color[2]) // 2,
                        max(bfs_color[3], dfs_color[3])
                    )
                elif (y, x) in bfs_visited:
                    color = bfs_color
                else:
                    color = dfs_color
                
                draw.rectangle([
                    x * self.cell_size, y * self.cell_size,
                    (x + 1) * self.cell_size, (y + 1) * self.cell_size
                ], fill=color)
    
    def draw_path_comparison(self, draw, path, algorithm):
        """Draw path with algorithm-specific style"""
        path_color = self.BFS_PATH_COLOR if algorithm == "BFS" else self.DFS_PATH_COLOR
        path_width = 3 if algorithm == "BFS" else 1  # Make BFS path thicker
        path_points = []
        
        for node in path:
            y, x = node
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            path_points.append((center_x, center_y))
            
            # Draw a circle at each path node
            draw.ellipse([
                x * self.cell_size + 2, y * self.cell_size + 2,
                (x + 1) * self.cell_size - 2, (y + 1) * self.cell_size - 2
            ], outline=path_color, width=2)
        
        # Draw connecting lines between path nodes
        if len(path_points) > 1:
            # Draw dashed line for DFS to distinguish from BFS
            if algorithm == "DFS":
                dash_pattern = [4, 4]  # Dashed pattern
            else:
                dash_pattern = None  # Solid line for BFS
            
            # Draw the line segment by segment to support dashed pattern
            for i in range(len(path_points)-1):
                if dash_pattern:
                    # Custom dashed line drawing
                    self.draw_dashed_line(draw, path_points[i], path_points[i+1], 
                                        fill=path_color, width=path_width, dash_pattern=dash_pattern)
                else:
                    draw.line([path_points[i], path_points[i+1]], 
                             fill=path_color, width=path_width)
    
    def draw_dashed_line(self, draw, start, end, fill, width, dash_pattern):
        """Draw a dashed line between two points"""
        # Calculate line length and direction
        x1, y1 = start
        x2, y2 = end
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        # Calculate unit vector
        if length == 0:
            return
        dx = (x2 - x1)/length
        dy = (y2 - y1)/length
        
        # Draw dashed segments
        dash_length = sum(dash_pattern)
        distance = 0
        dash_index = 0
        draw_segment = True
        
        while distance < length:
            segment_length = min(dash_pattern[dash_index % len(dash_pattern)], length - distance)
            
            if draw_segment:
                seg_start = (x1 + dx * distance, y1 + dy * distance)
                seg_end = (x1 + dx * (distance + segment_length), y1 + dy * (distance + segment_length))
                draw.line([seg_start, seg_end], fill=fill, width=width)
            
            distance += segment_length
            dash_index += 1
            draw_segment = not draw_segment
    
    def visualize_step(self, algorithm, current, new_node, frontier, visited, parent, path=None):
        """Visualize a single step of the algorithm"""
        # Create new image for drawing
        if self.background_image:
            img = self.background_image.copy()
        else:
            img = Image.new('RGB', (self.grid_size*self.cell_size, self.grid_size*self.cell_size), 'white')
            
        # Convert to RGBA for transparency support
        img = img.convert('RGBA')
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines (only if no background image)
        if not self.background_image:
            self.draw_grid_lines(draw)
        
        # Draw obstacles
        self.draw_obstacles(img)
        
        # Draw visited nodes
        self.draw_visited_nodes(draw, visited, algorithm)
        
        # Draw frontier nodes
        self.draw_frontier_nodes(draw, frontier, algorithm)
        
        # Draw current node being explored
        if current:
            self.draw_current_node(draw, current)
        
        # Draw new node being added to frontier
        if new_node:
            self.draw_new_node(draw, new_node)
        
        # Draw path if available
        if path:
            self.draw_path(draw, path, algorithm)
        
        # Draw start and goal positions
        self.draw_start_goal_positions(img)
        
        # Update display
        self.update_canvas_display(img)
    
    def draw_grid_lines(self, draw):
        """Draw grid lines on the image"""
        for i in range(self.grid_size + 1):
            # Vertical lines
            draw.line([(i * self.cell_size, 0), (i * self.cell_size, self.grid_size * self.cell_size)], 
                     fill=self.GRID_LINE_COLOR)
            # Horizontal lines
            draw.line([(0, i * self.cell_size), (self.grid_size * self.cell_size, i * self.cell_size)], 
                     fill=self.GRID_LINE_COLOR)
    
    def draw_obstacles(self, img):
        """Draw obstacles on the image"""
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.map_grid[y][x] == 1:
                    obstacle = Image.new('RGBA', (self.cell_size, self.cell_size), self.OBSTACLE_COLOR)
                    img.paste(obstacle, (x * self.cell_size, y * self.cell_size), obstacle)
    
    def draw_visited_nodes(self, draw, visited, algorithm):
        """Draw visited nodes with algorithm-specific color"""
        color = self.BFS_VISITED_COLOR if algorithm == "BFS" else self.DFS_VISITED_COLOR
        for y, x in visited:
            if (y, x) != (self.start_pos[1], self.start_pos[0]) and (y, x) != (self.goal_pos[1], self.goal_pos[0]):
                draw.rectangle([
                    x * self.cell_size, y * self.cell_size,
                    (x + 1) * self.cell_size, (y + 1) * self.cell_size
                ], fill=color)
    
    def draw_frontier_nodes(self, draw, frontier, algorithm):
        """Draw frontier nodes with a different shade"""
        base_color = self.BFS_VISITED_COLOR if algorithm == "BFS" else self.DFS_VISITED_COLOR
        frontier_color = (base_color[0], base_color[1], base_color[2], 200)  # More opaque
        
        for y, x in frontier:
            if (y, x) != (self.start_pos[1], self.start_pos[0]) and (y, x) != (self.goal_pos[1], self.goal_pos[0]):
                draw.rectangle([
                    x * self.cell_size, y * self.cell_size,
                    (x + 1) * self.cell_size, (y + 1) * self.cell_size
                ], fill=frontier_color)
    
    def draw_current_node(self, draw, node):
        """Highlight the current node being explored"""
        y, x = node
        draw.rectangle([
            x * self.cell_size, y * self.cell_size,
            (x + 1) * self.cell_size, (y + 1) * self.cell_size
        ], fill=(255, 255, 0, 200))  # Yellow
    
    def draw_new_node(self, draw, node):
        """Highlight a new node being added to frontier"""
        y, x = node
        draw.rectangle([
            x * self.cell_size, y * self.cell_size,
            (x + 1) * self.cell_size, (y + 1) * self.cell_size
        ], fill=(255, 165, 0, 200))  # Orange
    
    def draw_path(self, draw, path, algorithm):
        """Draw the solution path directly on the image"""
        path_color = self.BFS_PATH_COLOR if algorithm == "BFS" else self.DFS_PATH_COLOR
        path_points = []
        
        for node in path:
            y, x = node
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            path_points.append((center_x, center_y))
            
            # Draw a circle at each path node
            draw.ellipse([
                x * self.cell_size + 2, y * self.cell_size + 2,
                (x + 1) * self.cell_size - 2, (y + 1) * self.cell_size - 2
            ], fill=path_color)
        
        # Draw connecting lines between path nodes
        if len(path_points) > 1:
            draw.line(path_points, fill=path_color, width=3)
    
    def draw_start_goal_positions(self, img):
        """Draw start and goal positions"""
        if self.start_pos:
            x, y = self.start_pos
            start_img = Image.new('RGBA', (self.cell_size, self.cell_size), (0, 0, 0, 0))
            start_draw = ImageDraw.Draw(start_img)
            start_draw.ellipse([2, 2, self.cell_size-2, self.cell_size-2], fill=self.START_COLOR)
            start_draw.text((self.cell_size//2 - 3, self.cell_size//2 - 5), "A", fill='white')
            img.paste(start_img, (x * self.cell_size, y * self.cell_size), start_img)
        
        if self.goal_pos:
            x, y = self.goal_pos
            goal_img = Image.new('RGBA', (self.cell_size, self.cell_size), (0, 0, 0, 0))
            goal_draw = ImageDraw.Draw(goal_img)
            goal_draw.ellipse([2, 2, self.cell_size-2, self.cell_size-2], fill=self.GOAL_COLOR)
            goal_draw.text((self.cell_size//2 - 3, self.cell_size//2 - 5), "T", fill='white')
            img.paste(goal_img, (x * self.cell_size, y * self.cell_size), goal_img)
    
    def update_canvas_display(self, img):
        """Update the canvas with the new image"""
        img_tk = ImageTk.PhotoImage(img)
        self.map_canvas.delete("all")
        self.map_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.map_canvas.image = img_tk
    
    def update_result_labels(self):
        """Update result labels for both algorithms"""
        self.bfs_label.config(text=f"BFS: Waktu={self.bfs_results['time']:.3f}s, "
                                 f"Node={self.bfs_results['nodes']}, "
                                 f"Jalur={self.bfs_results['path_length']}")
        
        self.dfs_label.config(text=f"DFS: Waktu={self.dfs_results['time']:.3f}s, "
                                 f"Node={self.dfs_results['nodes']}, "
                                 f"Jalur={self.dfs_results['path_length']}")
    
    def update_comparison_chart(self):
        """Update the comparison chart and complexity information"""
        self.ax.clear()
        self.complexity_ax.clear()
        
        algorithms = ['BFS', 'DFS']
        time_values = [self.bfs_results['time'], self.dfs_results['time']]
        nodes_values = [self.bfs_results['nodes'], self.dfs_results['nodes']]
        path_values = [self.bfs_results['path_length'], self.dfs_results['path_length']]
        
        x = range(len(algorithms))
        width = 0.25
        
        # Plot time comparison
        bars1 = self.ax.bar([i - width for i in x], time_values, width, label='Waktu (s)', color='blue')
        # Plot nodes comparison
        bars2 = self.ax.bar(x, nodes_values, width, label='Node Dikunjungi', color='green')
        # Plot path length comparison
        bars3 = self.ax.bar([i + width for i in x], path_values, width, label='Panjang Jalur', color='red')
        
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(algorithms)
        self.ax.set_title('Perbandingan Kinerja BFS vs DFS')
        self.ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                self.ax.annotate(f'{height:.1f}' if isinstance(height, float) else f'{height}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom')
        
        # Add time complexity information
        complexity_text = (
            "Kompleksitas Waktu:\n"
            "BFS: O(V + E) = O(b^d) - Optimal untuk graf tanpa bobot\n"
            "DFS: O(V + E) = O(b^m) - Tidak optimal tapi menggunakan lebih sedikit memori\n\n"
            "Keterangan:\n"
            "V = Vertex (simpul)\n"
            "E = Edge (sisi)\n"
            "b = Faktor percabangan\n"
            "d = Kedalaman solusi\n"
            "m = Kedalaman maksimum pohon"
        )
        
        self.complexity_ax.axis('off')
        self.complexity_ax.text(0.05, 0.95, complexity_text, 
                              ha='left', va='top', fontsize=9,
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        self.canvas.draw()
    
    def draw_grid(self):
        """Draw the grid with current state"""
        if self.background_image:
            img = self.background_image.copy()
        else:
            img = Image.new('RGB', (self.grid_size*self.cell_size, self.grid_size*self.cell_size), 'white')
            
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines (only if no background image)
        if not self.background_image:
            self.draw_grid_lines(draw)
        
        # Draw obstacles
        self.draw_obstacles(img)
        
        # Draw start and goal positions
        self.draw_start_goal_positions(img)
        
        # Update display
        self.update_canvas_display(img)
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.initialize_grid()
        self.bfs_results = {'time': 0, 'nodes': 0, 'path_length': 0}
        self.dfs_results = {'time': 0, 'nodes': 0, 'path_length': 0}
        self.update_result_labels()
        self.ax.clear()
        self.complexity_ax.clear()
        self.canvas.draw()
        self.status_label.config(text="Status: Direset")

if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingVisualizer(root)
    root.mainloop()