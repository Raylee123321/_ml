import math
import random
from solution import Solution
from solver import hill_climbing, steepest_ascent_hill_climbing, simulated_annealing

def generate_random_tsp(num_cities, grid_size=100, seed=42):
    """
    Generates random 2D city coordinates and computes the symmetric Euclidean distance matrix.
    City indices are 1-based.
    """
    random.seed(seed)
    coordinates = {i: (random.uniform(0, grid_size), random.uniform(0, grid_size)) for i in range(1, num_cities + 1)}
    
    # Compute Euclidean distance matrix
    matrix = [[0.0] * (num_cities + 1) for _ in range(num_cities + 1)]
    for i in range(1, num_cities + 1):
        for j in range(1, num_cities + 1):
            if i == j:
                matrix[i][j] = 0.0
            else:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                matrix[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
    return coordinates, matrix

def main():
    num_cities = 15
    print(f"=== 產生 {num_cities} 個城市的隨機 TSP 問題 ===")
    coords, distance_matrix = generate_random_tsp(num_cities, seed=42)
    
    # 設置全域距離矩陣
    Solution.set_distance_matrix(distance_matrix)
    
    # 1. 產生初始解: 1 => 2 => 3 => ... => n => 1
    initial_path = list(range(1, num_cities + 1))
    init_sol = Solution(initial_path)
    
    print("\n--- 初始解 ---")
    print(f"路徑: {init_sol}")
    print(f"高度 (height): {init_sol.height():.4f} (距離: {-init_sol.height():.4f})")
    
    # 2. 爬山演算法 (Hill Climbing)
    print("\n--- 執行爬山演算法 (Hill Climbing) ---")
    hc_sol, hc_height = hill_climbing(init_sol, max_iter=10000)
    print(f"最佳路徑: {hc_sol}")
    print(f"最佳高度: {hc_height:.4f} (距離: {-hc_height:.4f})")
    
    # 3. 最陡爬山演算法 (Steepest Ascent Hill Climbing)
    print("\n--- 執行最陡爬山演算法 (Steepest Ascent Hill Climbing) ---")
    sahc_sol, sahc_height = steepest_ascent_hill_climbing(init_sol, max_iter=1000)
    print(f"最佳路徑: {sahc_sol}")
    print(f"最佳高度: {sahc_height:.4f} (距離: {-sahc_height:.4f})")
    
    # 4. 模擬退火演算法 (Simulated Annealing)
    print("\n--- 執行模擬退火演算法 (Simulated Annealing) ---")
    sa_sol, sa_height = simulated_annealing(init_sol, init_temp=1000.0, cooling_rate=0.995, max_iter=20000)
    print(f"最佳路徑: {sa_sol}")
    print(f"最佳高度: {sa_height:.4f} (距離: {-sa_height:.4f})")

if __name__ == "__main__":
    main()
