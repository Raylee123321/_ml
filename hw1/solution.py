import random

class Solution:
    # Class-level distance matrix.
    # distance_matrix[i][j] is the distance between city i and city j.
    # Cities are 1-indexed.
    distance_matrix = {}
    num_cities = 0

    def __init__(self, path):
        self.path = list(path)

    @classmethod
    def set_distance_matrix(cls, matrix):
        """Sets the shared distance matrix for all Solution instances."""
        cls.distance_matrix = matrix
        cls.num_cities = len(matrix) - 1

    def height(self):
        """Calculates the height of the solution, defined as -1 * total TSP distance."""
        if not self.distance_matrix:
            return 0.0
        
        dist = 0.0
        n = len(self.path)
        for i in range(n):
            u = self.path[i]
            v = self.path[(i + 1) % n]
            dist += self.distance_matrix[u][v]
        return -dist

    def neighbor(self):
        """
        Generates a random neighbor solution by performing a 2-opt swap.
        Selects two non-adjacent edges (a, b) and (c, d) in the tour,
        and reconnects them as (a, d) and (b, c) (reversing the path between them).
        """
        n = len(self.path)
        if n < 4:
            # For less than 4 cities, 2-opt doesn't change the path length/topology.
            # We just do a simple node swap to produce a permutation.
            new_path = list(self.path)
            if n > 1:
                i, j = random.sample(range(n), 2)
                new_path[i], new_path[j] = new_path[j], new_path[i]
            return Solution(new_path)

        # Select two random indices i and j where i + 1 < j
        # The edges to cut are (path[i], path[i+1]) and (path[j], path[j+1])
        # (with wrap-around if j is the last index).
        # To avoid reversing the whole list (which is the same tour), we require:
        i = random.randint(0, n - 3)
        j = random.randint(i + 2, n - 1)
        if i == 0 and j == n - 1:
            # Avoid reversing the entire array (same undirected tour)
            j = n - 2
        
        # 2-opt swap: reverse the subsegment from i+1 to j
        new_path = self.path[:i+1] + self.path[i+1:j+1][::-1] + self.path[j+1:]
        return Solution(new_path)

    def all_neighbors(self):
        """Generates all possible 2-opt neighbor solutions."""
        n = len(self.path)
        neighbors = []
        if n < 4:
            # Just return permutations for tiny tours
            import itertools
            for p in itertools.permutations(self.path):
                if p[0] == self.path[0]: # break symmetry/starting city fixed
                    neighbors.append(Solution(list(p)))
            return neighbors

        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    # Swapping 0 and n-1 reverses the entire tour, which is the same undirected tour
                    continue
                new_path = self.path[:i+1] + self.path[i+1:j+1][::-1] + self.path[j+1:]
                neighbors.append(Solution(new_path))
        return neighbors

    def __str__(self):
        # Format as 1=>2=>3=>...=>n=>1
        return "=>".join(map(str, self.path)) + f"=>{self.path[0]}"
