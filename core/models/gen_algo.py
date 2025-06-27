import pandas as pd
import random
import os
from keras.models import load_model
from core.models.creativity_nn import creativity_nn
from core.predict.personality import predict_personality
import joblib

# GA parameters
POPULATION_SIZE = 50
GENERATIONS = 50
TOURNAMENT_SIZE = 3
NUM_PARENTS = 25
INITIAL_CROSSOVER_RATE = 0.9
DECAY_FACTOR = 0.5
MIN_MUTATION_RATE = 0.05
MAX_MUTATION_RATE = 0.2

model = load_model("models/creativity_model_final.keras")
scaler = joblib.load("training/nn_creativity/data/scaler.save")


def initialize_word_pool():
    try:
        print("\n[PROCESS] Parsing word_pool.xlsx...")

        df = pd.read_excel("data/word_pool.xlsx")
        df = df.map(
            lambda x: str(x).strip().replace("\xa0", "") if pd.notnull(x) else x
        )
        print("[STATUS] Successfully read word_pool.xlsx")
        word_categories = {}

        # For each column, extract trait and corresponding word list
        for column in df.columns:
            word_list = df[column].dropna().tolist()
            word_categories[column] = word_list

        return word_categories

    except FileNotFoundError:
        raise FileNotFoundError("[ERROR] word_pool.xlsx not found.")


# Personality NN output
def get_NN_personality(answers=None):
    if answers is None:
        # Default for standlone use
        answers = [1, 1, 2, 3, 1, 3, 2, 4, 3, 2, 1, 2, 1, 1, 4]
    chosen_trait = predict_personality(answers)
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

    fitness_pop = fitness(population)

    return population, fitness_pop


# Fitness Function
def fitness(population):
    # Return fitness score as basis of selection
    return creativity_nn(population, model, scaler)


# Parent Selection
def tournament_selection(population, fitness_pop):
    parents = []
    for _ in range(NUM_PARENTS):
        tournament_group = random.sample(population, TOURNAMENT_SIZE)
        tournament_indices = [population.index(ind) for ind in tournament_group]

        max_val = fitness_pop[tournament_indices[0]]
        max_index = tournament_indices[0]
        for indices in tournament_indices:
            if fitness_pop[indices] > max_val:
                max_val = fitness_pop[indices]
                max_index = indices

        winner = population[max_index]
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
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
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
    mutation_rate = MAX_MUTATION_RATE - (
        normalized_diversity * (MAX_MUTATION_RATE - MIN_MUTATION_RATE)
    )

    # Ensure the rate is within the defined min/max bounds
    return max(MIN_MUTATION_RATE, min(mutation_rate, MAX_MUTATION_RATE))


# Crossover Technique
def uniform_crossover(parents, CROSSOVER_RATE):
    offsprings = []
    crossover_count = 0
    no_crossover_count = 0

    while len(offsprings) < POPULATION_SIZE:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

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

        offsprings.append(child1)
        if len(offsprings) < POPULATION_SIZE:
            offsprings.append(child2)

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


def run_ga(answers=None):
    # Load word pool
    word_pool = initialize_word_pool()

    # Get the personality trait
    trait = get_NN_personality(answers)

    # Initialize population
    population, fitness_pop = initialize_population(word_pool, trait)

    # Get the word list for the chosen trait for mutation
    selected_words = word_pool[trait]

    # Calculate the max possible Levenshtein distance for normalization
    longest_word_len = len(max(selected_words, key=len))

    # Usernames are two words long
    max_possible_distance = longest_word_len * 2

    # Main GA loop
    for generation in range(GENERATIONS):
        # Step 1: Selection
        parents = tournament_selection(population, fitness_pop)

        # Step 2: Crossover
        crossover_rate = INITIAL_CROSSOVER_RATE
        offsprings, crossover_count, no_crossover_count = uniform_crossover(
            parents, crossover_rate
        )

        # Step 3: Mutation
        diversity = population_diversity(offsprings)
        adaptive_mutation_rate = get_adaptive_mutation_rate(
            diversity, max_possible_distance
        )

        population, mutation_count = mutate(
            offsprings, selected_words, adaptive_mutation_rate
        )
        fitness_pop = fitness(population)

    # Find and return the best individual from the final population
    max_val = fitness_pop[0]
    max_index = 0
    for i in range(len(fitness_pop)):
        if fitness_pop[i] > max_val:
            max_val = fitness_pop[i]
            max_index = i

    best_individual = population[max_index]
    best_username = "".join(best_individual)

    # Return both the trait and the best username for BE usecase
    return trait, best_username


if __name__ == "__main__":
    LOG_DIR = os.path.join("training", "gen_algo")
    LOG_FILE = get_next_log_filename(LOG_DIR)

    # Load word pool
    word_pool = initialize_word_pool()

    # Get the personality trait
    trait = get_NN_personality()

    # Initialize population
    population, fitness_pop = initialize_population(word_pool, trait)

    # Get the word list for the chosen trait for mutation
    selected_words = word_pool[trait]

    # Calculate the max possible Levenshtein distance for normalization
    # Rough estimation only for normalization purposes
    longest_word_len = len(max(selected_words, key=len))
    max_possible_distance = longest_word_len * 2  # Usernames are two words long

    with open(LOG_FILE, "w") as f:
        print(f"[PROCESS] GA running... Log will be saved to {LOG_FILE}")

        f.write("Epithet AI - Genetic Algorithm Log\n")
        f.write("=" * 40 + "\n\n")

        log_section(f, "Initial Population", population)

        # Enter the main GA loop
        for generation in range(GENERATIONS):
            print(f"Generation {generation+1}")
            print("=" * 80)
            f.write(
                f"========== Generation {generation + 1}/{GENERATIONS} ==========\n\n"
            )

            # [1] Selection
            parents = tournament_selection(population, fitness_pop)
            f.write("SELECTION\n")
            log_section(f, "Selected Parents", parents)

            # [2] Crossover
            # Get the adaptive crossover rate
            crossover_rate = INITIAL_CROSSOVER_RATE
            offsprings, crossover_count, no_crossover_count = uniform_crossover(
                parents, crossover_rate
            )
            f.write("CROSSOVER\n")
            f.write("[Crossover Info]\n")
            f.write(f"Crossover Rate: {crossover_rate:.4f}\n")
            f.write(f"Crossover Pairs: {crossover_count}\n")
            f.write(f"No Crossover Pairs: {no_crossover_count}\n\n")
            log_section(f, "Offspring after Crossover", offsprings)

            # [3] Mutation
            # First, calculate the diversity of the current population of offspring
            diversity = population_diversity(offsprings)
            # Next, get the adaptive mutation rate based on this diversity
            adaptive_mutation_rate = get_adaptive_mutation_rate(
                diversity, max_possible_distance
            )

            population, mutation_count = mutate(
                offsprings, selected_words, adaptive_mutation_rate
            )
            f.write("MUTATION\n")
            f.write("[Mutation Info]\n")
            f.write(f"Population Diversity: {diversity:.4f}\n")
            f.write(f"Adaptive Mutation Rate: {adaptive_mutation_rate:.4f}\n")
            f.write(f"Total Mutations: {mutation_count}\n\n")
            log_section(f, "New Population after Mutation", population)
            fitness_pop = fitness(population)

        # End the GA loop
        f.write("=" * 40 + "\n")
        f.write("GA PROCESS COMPLETE\n")

        # Find and display the best individual from the final population
        max_val = fitness_pop[0]
        max_index = 0
        for i in range(len(fitness_pop)):
            if fitness_pop[i] > max_val:
                max_val = fitness_pop[i]
                max_index = i

        # Determine the winner of each tournament_group based on fitness score
        best_individual = population[max_index]

        result_message = f"Best Username Found: {''.join(best_individual)}\n"

        f.write(result_message)
        print(f"\n{result_message}")
        print(f"Best Fitness: {fitness_pop[max_index]}")
