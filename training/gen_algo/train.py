import pandas as pd
import random

# GA parameters
POPULATION_SIZE = 100
GENERATIONS = 100
TOURNAMENT_SIZE = 3
NUM_PARENTS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

def initialize_word_pool():
    try:
        print("\nParsing word_pool.xlsx...")

        df = pd.read_excel('training\gen_algo\word_pool.xlsx')
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
def fitness():
    print("For fitness function")
    # return fitness_score

# Parent Selection
def tournament_selection():
    print("For parent selection")

# Crossover Technique
def uniform_crossover():
    print("For crossover technique")

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

    # Print initial population
    print("\nSample of initialized population:")
    for i, individual in enumerate(population):
        print(f"Individual {i + 1}: {individual}")
    