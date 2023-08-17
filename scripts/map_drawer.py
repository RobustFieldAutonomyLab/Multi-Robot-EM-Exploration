import matplotlib.pyplot as plt


class GridClickTool:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0] * cols for _ in range(rows)]  # 0 represents white, 1 represents black
        self.clicked_points = []
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.update_plot()

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.grid[row][col] = 1 - self.grid[row][col]  # Toggle between white and black
                self.clicked_points.append((row, col))
                self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.imshow(self.grid, cmap='binary', origin='upper')
        plt.draw()
        with open('map.txt', 'w') as file:
            file.write('map:\n')
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.grid[i][j] == 1:
                        file.write(str(i) + ', ' + str(j) + '\n')

    def show(self):
        plt.show()


# Create a grid with 10 rows and 10 columns
grid_tool = GridClickTool(50, 50)
grid_tool.show()
