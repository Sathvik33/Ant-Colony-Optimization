import numpy as np
import random

class ACO:
    def __init__(self, grid, ant_count=20, generations=30, evaporation_rate=0.5, alpha=1, beta=2):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.ant_count = ant_count
        self.generations = generations
        self.evaporation = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.num_nodes = self.rows * self.cols
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))
        self.graph = self._build_graph()

    def _cell_to_index(self, r, c):
        return r * self.cols + c

    def _index_to_cell(self, idx):
        return divmod(idx, self.cols)

    def _build_graph(self):
        graph = np.full((self.num_nodes, self.num_nodes), np.inf)
        for r in range(self.rows):
            for c in range(self.cols):
                idx = self._cell_to_index(r, c)
                neighbors = self._get_neighbors(r, c)
                for nr, nc in neighbors:
                    nidx = self._cell_to_index(nr, nc)
                    graph[idx][nidx] = 1  # uniform cost
        return graph

    def _get_neighbors(self, r, c):
        directions = [(-1,0), (1,0), (0,-1), (0,1)]  # up, down, left, right
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def _probability(self, current, unvisited):
        pheromone = self.pheromone[current][unvisited] ** self.alpha
        heuristic = (1.0 / self.graph[current][unvisited]) ** self.beta
        product = pheromone * heuristic
        total = np.sum(product)
        return product / total if total > 0 else np.ones(len(unvisited)) / len(unvisited)

    def _build_path(self, src, dst):
        path = [src]
        visited = set(path)
        current = src
        while current != dst:
            unvisited = [i for i in range(self.num_nodes) if i not in visited and self.graph[current][i] < np.inf]
            if not unvisited:
                break
            probs = self._probability(current, unvisited)
            next_node = np.random.choice(unvisited, p=probs)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        return path if path[-1] == dst else None

    def _path_cost(self, path):
        return sum(self.graph[path[i]][path[i+1]] for i in range(len(path)-1))

    def _update_pheromones(self, paths):
        self.pheromone *= (1 - self.evaporation)
        for path in paths:
            if path:
                cost = self._path_cost(path)
                for i in range(len(path)-1):
                    a, b = path[i], path[i+1]
                    self.pheromone[a][b] += 1.0 / cost
                    self.pheromone[b][a] += 1.0 / cost

    def run(self, start_val, end_val):
        try:
            start_pos = tuple(map(int, np.argwhere(self.grid == start_val)[0]))
            end_pos = tuple(map(int, np.argwhere(self.grid == end_val)[0]))
        except IndexError:
            print("Source or Destination value not found in the grid.")
            return None, None

        src = self._cell_to_index(*start_pos)
        dst = self._cell_to_index(*end_pos)

        best_path, best_cost = None, float('inf')

        for _ in range(self.generations):
            paths = [self._build_path(src, dst) for _ in range(self.ant_count)]
            self._update_pheromones(paths)
            for path in paths:
                if path:
                    cost = self._path_cost(path)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path

        return [self._index_to_cell(idx) for idx in best_path], best_cost


if __name__ == "__main__":
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))

    print(f"Enter the {rows}x{cols} grid row by row (space-separated integers):")
    grid_vals = []
    for i in range(rows):
        row = list(map(int, input(f"Row {i+1}: ").strip().split()))
        if len(row) != cols:
            raise ValueError(f"Row {i+1} must have {cols} values.")
        grid_vals.append(row)

    grid = np.array(grid_vals)

    source_val = int(input("Enter source value (e.g., 1): "))
    destination_val = int(input("Enter destination value (e.g., 9): "))

    aco = ACO(grid)
    path, cost = aco.run(start_val=source_val, end_val=destination_val)

    if path:
        print("Best Path from", source_val, "to", destination_val, " is:")
        for step in path:
            print(step, end=" ➝ ")
        print(f"Total Cost: {cost}")
    else:
        print("No valid path found!")
