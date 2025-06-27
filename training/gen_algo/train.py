import pandas as pd
import random
import os

# GA parameters
POPULATION_SIZE = 100
GENERATIONS = 100
TOURNAMENT_SIZE = 3
NUM_PARENTS = 50
INITIAL_CROSSOVER_RATE = 0.9
DECAY_FACTOR = 0.5
MIN_MUTATION_RATE = 0.05
MAX_MUTATION_RATE = 0.1

# MIN_CROSSOVER_RATE = 0.4 (For Levenshtein)
# MAX_CROSSOVER_RATE = 0.9

def initialize_word_pool():
    try:
        print("\n[PROCESS] Parsing word_pool.xlsx...")

        df = pd.read_excel('data\word_pool.xlsx')
        df = df.map(lambda x: str(x).strip().replace('\xa0', '') if pd.notnull(x) else x)
        print("[STATUS] Successfully read word_pool.xlsx")
        word_categories = {}

        # For each column, extract trait and corresponding word list
        for column in df.columns:
            word_list = df[column].dropna().tolist()
            word_categories[column] = word_list
        
        return word_categories

    except FileNotFoundError:
        raise FileNotFoundError("[ERROR] word_pool.xlsx not found.")

# NN output (For checking only)
def get_NN_personality():
    traits = ["artista" , "diva", "oa", "wildcard", "achiever", "emo", "gamer", "softie"]
    chosen_trait = random.choice(traits)
    return chosen_trait
    
# Initialize Population
def initialize_population(word_pool, trait):
    if trait not in word_pool:
        raise ValueError(f"[ERROR] trait '{trait}' not found in word pool")
    
    # Words from the returned trait 
    selected_words = word_pool[trait]
    print(f"\n[INFO] Using words from trait: {trait}")

    population = []
    for _ in range(POPULATION_SIZE):
        individual = [random.choice(selected_words) for _ in range(2)] 
        population.append(individual)
    return population

# Fitness Function
def fitness(individual):
    # Return fitness score as basis of selection
    # Replace fitness score (creativity score) once Creativity NN is done
    # Import Creativity NN and use its function
    return random.random()

# Parent Selection
def tournament_selection(population):
    parents = []
    for _ in range(NUM_PARENTS):
        # Select a subset of the population for tournament
        tournament_group = random.sample(population, TOURNAMENT_SIZE)

        # Determine the winner of each tournament_group based on fitness score
        winner = max(tournament_group, key=fitness)
        parents.append(winner)
    return parents

# Levenshtein Distance: Measures the difference between two strings
def levenshtein_distance(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,     # Deletion
                dp[i][j - 1] + 1,     # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[len1][len2]


# Population Diversity thru Levenshtein
def population_diversity(population):
    total_distance = 0
    count = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Join the words to form a single string for comparison
            ind1_str = "".join(population[i])
            ind2_str = "".join(population[j])
            total_distance += levenshtein_distance(ind1_str, ind2_str)
            count += 1
    return total_distance / count if count > 0 else 0

# Mutation Rate changes based on diversity thru Levenshtein
def get_adaptive_mutation_rate(diversity_score, max_possible_distance):

    # Normalize the diversity score to a value between 0 and 1
    if max_possible_distance > 0:
        normalized_diversity = diversity_score / max_possible_distance
    else: 
        normalized_diversity = 0
    
    # Calculate the mutation rate using an inverse relationship to diversity
    mutation_rate = MAX_MUTATION_RATE - (normalized_diversity * (MAX_MUTATION_RATE - MIN_MUTATION_RATE))
    
    # Ensure the rate is within the defined min/max bounds
    return max(MIN_MUTATION_RATE, min(mutation_rate, MAX_MUTATION_RATE))

# Crossover Rate decreases each generation
def get_adaptive_crossover_rate(current_generation, max_generations):

    # Calculate how much the crossover rate should decrease based on the current generation
    decay = (INITIAL_CROSSOVER_RATE - DECAY_FACTOR) * (current_generation / max_generations)
    
    # Ensure that the crossover rate does not fall below the DECAY_FACTOR
    return max(DECAY_FACTOR, INITIAL_CROSSOVER_RATE - decay)

# Crossover Technique
def uniform_crossover(parents, CROSSOVER_RATE):
    offsprings = []
    crossover_count = 0
    no_crossover_count = 0

    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        # Ensure there is a second parent to pair with
        if i + 1 < len(parents):
            parent2 = parents[i + 1]
            
            if random.random() < CROSSOVER_RATE:
                child1 = []
                child2 = []
                for gene1, gene2 in zip(parent1, parent2):
                    if random.random() < 0.5:
                        child1.append(gene1)
                        child2.append(gene2)
                    else:
                        child1.append(gene2)
                        child2.append(gene1)
                crossover_count += 1
            else:
                child1, child2 = parent1[:], parent2[:]
                no_crossover_count += 1

            offsprings.extend([child1, child2])
    
    return offsprings, crossover_count, no_crossover_count

# Mutation Technique
def mutate(offsprings, selected_words, MUTATION_RATE):
    mutated_offsprings = []
    mutation_count = 0
    for individual in offsprings:
        mutated_individual = individual[:]  # Create a copy to modify
        for i in range(len(mutated_individual)):
            # Check if mutation should occur for this word
            if random.random() < MUTATION_RATE:
                # Replace the word with a new random word from the pool
                mutated_individual[i] = random.choice(selected_words)
                mutation_count += 1
        mutated_offsprings.append(mutated_individual)
    return mutated_offsprings, mutation_count

def get_next_log_filename(log_dir):
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
    base_name = "ga_log"
    counter = 1
    while True:
        log_file = f"{base_name}_{counter}.txt"
        log_path = os.path.join(log_dir, log_file)
        if not os.path.exists(log_path):
            return log_path
        counter += 1

def log_section(file, title, items):
    file.write(f"[{title}]\n")
    for i, item in enumerate(items):
        file.write(f"{i + 1: >3}. {''.join(item)}\n")
    file.write("\n")

if __name__ == "__main__":
    LOG_DIR = os.path.join("training", "gen_algo")
    LOG_FILE = get_next_log_filename(LOG_DIR)

    # Load word pool
    word_pool = initialize_word_pool()

    # Get the personality trait
    trait = get_NN_personality()

    # Initialize population
    population = initialize_population(word_pool, trait)
    
    # Get the word list for the chosen trait for mutation
    selected_words = word_pool[trait]

    # Calculate the max possible Levenshtein distance for normalization
    # Rough estimation only for normalization purposes
    longest_word_len = len(max(selected_words, key=len))
    max_possible_distance = longest_word_len * 2 # Usernames are two words long

    with open(LOG_FILE, 'w') as f:
        print(f"[PROCESS] GA running... Log will be saved to {LOG_FILE}")
        
        f.write("Epithet AI - Genetic Algorithm Log\n")
        f.write("="*40 + "\n\n")
        
        log_section(f, "Initial Population", population)

        # Enter the main GA loop
        for generation in range(GENERATIONS):
            f.write(f"========== Generation {generation + 1}/{GENERATIONS} ==========\n\n")

            # [1] Selection
            parents = tournament_selection(population)
            f.write("SELECTION\n")
            log_section(f, "Selected Parents", parents)
            
            # [2] Crossover
            # Get the adaptive crossover rate
            adaptive_crossover_rate = get_adaptive_crossover_rate(generation, GENERATIONS)
            offsprings, crossover_count, no_crossover_count = uniform_crossover(parents, adaptive_crossover_rate)
            f.write("CROSSOVER\n")
            f.write("[Crossover Info]\n")
            f.write(f"Adaptive Crossover Rate: {adaptive_crossover_rate:.4f}\n")
            f.write(f"Crossover Pairs: {crossover_count}\n")
            f.write(f"No Crossover Pairs: {no_crossover_count}\n\n")
            log_section(f, "Offspring after Crossover", offsprings)

            # [3] Mutation
            # First, calculate the diversity of the current population of offspring
            diversity = population_diversity(offsprings)
            # Next, get the adaptive mutation rate based on this diversity
            adaptive_mutation_rate = get_adaptive_mutation_rate(diversity, max_possible_distance)

            population, mutation_count = mutate(offsprings, selected_words, adaptive_mutation_rate)
            f.write("MUTATION\n")
            f.write("[Mutation Info]\n")
            f.write(f"Population Diversity: {diversity:.4f}\n")
            f.write(f"Adaptive Mutation Rate: {adaptive_mutation_rate:.4f}\n")
            f.write(f"Total Mutations: {mutation_count}\n\n")
            log_section(f, "New Population after Mutation", population)

        # End the GA loop
        f.write("="*40 + "\n")
        f.write("GA PROCESS COMPLETE\n")
        
        # Find and display the best individual from the final population
        best_individual = max(population, key=fitness)
        result_message = f"Best Username Found: {''.join(best_individual)}\n"
        
        f.write(result_message)
        print(f"\n{result_message}")