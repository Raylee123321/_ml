import math
import random

def hill_climbing(initial_sol, max_iter=10000):
    """
    Standard Hill Climbing (Random Mutation Hill Climbing).
    Attempts to maximize the height.
    """
    current = initial_sol
    current_height = current.height()
    
    for _ in range(max_iter):
        cand = current.neighbor()
        cand_height = cand.height()
        if cand_height > current_height:
            current = cand
            current_height = cand_height
            
    return current, current_height

def steepest_ascent_hill_climbing(initial_sol, max_iter=1000):
    """
    Steepest Ascent Hill Climbing.
    Evaluates all neighbors and moves to the best one.
    """
    current = initial_sol
    current_height = current.height()
    
    for _ in range(max_iter):
        neighbors = current.all_neighbors()
        if not neighbors:
            break
        # Find the best neighbor
        best_cand = max(neighbors, key=lambda sol: sol.height())
        best_height = best_cand.height()
        if best_height > current_height:
            current = best_cand
            current_height = best_height
        else:
            # Local optimum reached
            break
            
    return current, current_height

def simulated_annealing(initial_sol, init_temp=1000.0, cooling_rate=0.995, min_temp=0.01, max_iter=10000):
    """
    Simulated Annealing to maximize height.
    """
    current = initial_sol
    current_height = current.height()
    best = current
    best_height = current_height
    
    temp = init_temp
    steps = 0
    
    while temp > min_temp and steps < max_iter:
        cand = current.neighbor()
        cand_height = cand.height()
        
        # We want to maximize height
        delta = cand_height - current_height
        
        # If delta > 0, cand is better. If delta <= 0, accept with probability e^(delta / temp)
        if delta > 0 or random.random() < math.exp(delta / temp):
            current = cand
            current_height = cand_height
            if current_height > best_height:
                best = current
                best_height = current_height
                
        temp *= cooling_rate
        steps += 1
        
    return best, best_height
