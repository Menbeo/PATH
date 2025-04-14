class Graph:
    def __init__(self, size):
        self.adj_matrix = [[0]* size for _ in range(size)]
        self.size = size
        self.vertex_data = [''] * size
    def add_edge(self,u,v, weight):
        if 0 <= u < self.size and 0 <= v < self.size:
            
        