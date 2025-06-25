import pandas as pd
import random

# GA parameters
POPULATION_SIZE = 100
GENERATIONS = 100
TOURNAMENT_SIZE = 3
NUM_PARENTS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
# MIN_CROSSOVER_RATE = 0.4 (For Levenshtein)
# MAX_CROSSOVER_RATE = 0.9

def initialize_word_pool():
    try:
        print("\nParsing word_pool.xlsx...")

        df = pd.read_excel('data\word_pool.xlsx')
        df = df.map(lambda x: str(x).strip().replace('\xa0', '') if pd.notnull(x) else x)
        print("Successfully read word_pool.xlsx")
        word_categories = {}

        # For each column, extract trait and corresponding word list
        for column in df.columns:
            word_list = df[column].dropna().tolist()
            word_categories[column] = word_list

        # Print word pool
        # print("\nWord Pool: \n")
        # print_word_table(df)
        
        return word_categories

    except FileNotFoundError:
        raise FileNotFoundError("Error: word_pool.xlsx not found.")
    
# Print word pool table (For checking only)
def print_word_table(df):
    # Headers
    headers = list(df.columns)
    row_data = df.fillna('').values.tolist()

    # Print header
    col_widths = [max(len(str(item)) for item in [header] + [row[i] for row in row_data]) for i, header in enumerate(headers)]
    header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    divider = "-+-".join('-' * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(divider)

    # Print rows
    for row in row_data:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))

# NN output (For checking only)
def get_NN_personality():
    traits = ["artista" , "diva", "oa", "wildcard", "achiever", "emo", "gamer", "softie"]
    chosen_trait = random.choice(traits)
    return chosen_trait
    
# Initialize Population
def initialize_population(word_pool, trait):
    if trait not in word_pool:
        raise ValueError(f"trait '{trait}' not found in word pool")
    
    # Words from the returned trait 
    selected_words = word_pool[trait]
    print(f"\nUsing words from trait: {trait}")

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

'''
# Adaptive Crossover Rate

# Crossover Rate decreases each generation
def get_adaptive_crossover_rate(current_generation, max_generations):
 # 0.9 as the INITIAL_CROSSOVER_RATE and 0.5 as the DECAY FACTOR
 return 0.9 - (0.5 * (current_generation / max_generations))


# --- OR ---

# Hamming Distance
def hamming_distance(ind1, ind2):
    return sum(g1 != g2 for g1, g2 in zip(ind1, ind2))

# Crossover Rate changes based on diversity (Hamming)
def get_adaptive_crossover_rate(diversity):
    if diversity < 3:  # Too similar
        return 0.9  # High crossover to introduce more variety
    elif diversity > 6:  # Too diverse
        return 0.5  # More stable recombination
    else:
        return 0.7  # Balanced crossover

# --- OR ---

# Levenshtein Distance
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


# Population Diversity using Levenshtein
def population_diversity(population):
    total_distance = 0
    count = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            total_distance += levenshtein_distance(population[i], population[j])
            count += 1
    return total_distance / count if count > 0 else 0

# Crossover Rate changes based on diversity (Levenshtein)
def get_diversity_based_crossover_rate(diversity_score, max_possible_distance):
    normalized = diversity_score / max_possible_distance if max_possible_distance > 0 else 0
    # High diversity = less crossover, Low diversity = more crossover
    return MIN_CROSSOVER_RATE + (MAX_CROSSOVER_RATE - MIN_CROSSOVER_RATE) * (1 - normalized)
'''

# Crossover Technique
def uniform_crossover(parents, CROSSOVER_RATE):
    offsprings = []
    crossover_count = 0
    no_crossover_count = 0

    for i in range(0, len(parents), 2):
        parent1 = parents[i]
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
    
    print(f"\nCrossover pairs: {crossover_count}")
    print(f"No crossover pairs: {no_crossover_count}\n")

    return offsprings

# Mutation Technique
def mutate():
    print("For mutation technique")

if __name__ == "__main__":
    # Load word pool
    word_pool = initialize_word_pool()

    # Get the personality trait
    trait = get_NN_personality()

    # Initialize population
    population = initialize_population(word_pool, trait)

    max_possible_distance = 2

    # Print initial population
    # print("\nSample of initialized population:")
    # for i, individual in enumerate(population):
    #   print(f"Individual {i + 1}: {individual}")

    # Enter the main GA loop
    for generation in range(GENERATIONS):

        # [1] Selection
        parents = tournament_selection(population)
        
        # [2] Crossover

        # Get crossover_rate
        # crossover_rate = get_adaptive_crossover_rate(generation, GENERATIONS)

        # Get diversity of current population + crossover_rate
        # diversity = population_diversity(parents)
        # crossover_rate = get_diversity_based_crossover_rate(diversity, max_possible_distance)

        offsprings = uniform_crossover(parents, CROSSOVER_RATE)

        print(f"Generation {generation + 1}/{GENERATIONS}: Crossover Rate = {CROSSOVER_RATE:.2f}")

        # [3] Mutation
        # Crossover and mutation create the next population
        # Offsprings set as next population for now for testing purposes
        population = offsprings

    # End the GA loop
    print("\nGA PROCESS COMPLETE")
    
    # Find and display the best individual from the final population
    best_individual = max(population, key=fitness)
    print(f"Best Username Found: {''.join(best_individual)}")