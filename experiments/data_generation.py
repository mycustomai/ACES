import pandas as pd
import numpy as np
from scipy.stats import t, lognorm

# Core generation logic per group
def apply_generation(df_group, experiment_id, permutation_only=False, experiment_label=None):
    df_group = df_group.sample(frac=1).reset_index(drop=True) # permute rows
    
    # Store original attributes
    df_group['original_price'] = df_group['price']
    df_group['original_rating'] = df_group['rating']
    df_group['original_rating_count'] = df_group['rating_count']

    if permutation_only:
        df_group['experiment_label'] = experiment_label
        df_group['experiment_number'] = experiment_id
        df_group['position_in_experiment'] = range(len(df_group))
        df_group['assigned_position'] = range(len(df_group))
        df_group['desired_choice'] = -1 # -1 means not applicable
        df_group['stock_quantity'] = 100
        df_group['sponsored'] = False
        df_group['overall_pick'] = False
        df_group['low_stock'] = False
        df_group['best_seller'] = False
        df_group['limited_time'] = False
        df_group['discounted'] = False
        return df_group

    # --- 2-4. Price, Rating, and Rating Count ---
    # For experiment 0, keep original quantities (price, rating, rating_count)
    if experiment_id != 0:
        # --- 2. Price ---
        # Use log-normal distribution with location μ=0 and scale σ=0.3
        f_j = lognorm.rvs(s=0.3, scale=1, size=len(df_group))
        df_group['price'] = df_group['price'] * f_j
        df_group['price'] = np.round(df_group['price'], 2)

        # --- 3. Rating ---
        # Use uniform distribution [-0.4, 0.4] and apply formula r_new = r + α * (5 - r)
        alpha_j = np.random.uniform(-0.4, 0.4, len(df_group))
        df_group['rating'] = df_group['rating'] + alpha_j * (5 - df_group['rating'])
        df_group['rating'] = np.round(df_group['rating'], 1)  # Round to nearest first decimal
        df_group['rating'] = df_group['rating'].clip(lower=0.0, upper=5.0)

        # --- 4. Rating Count ---
        # Use log-normal distribution with s=1 and scale=1
        q_j = lognorm.rvs(s=1, scale=1, size=len(df_group))
        df_group['rating_count'] = (df_group['rating_count'] * q_j).astype(int)

    # --- 5. Sponsored ---
    # Bernoulli with p_spons = 0.5
    if np.random.rand() < 0.5:
        num_sponsored = np.random.choice([1, 2, 3, 4])
        sponsored_indices = np.random.choice(df_group.index, num_sponsored, replace=False)
        df_group['sponsored'] = False
        df_group.loc[sponsored_indices, 'sponsored'] = True
    else:
        df_group['sponsored'] = False

    # --- 6. Overall Pick ---
    # Bernoulli with p_overall = 0.5, select from products without sponsored tag
    if np.random.rand() < 0.5:
        eligible = df_group[~df_group['sponsored']]
        df_group['overall_pick'] = False
        if not eligible.empty:
            pick_idx = np.random.choice(eligible.index)
            df_group.loc[pick_idx, 'overall_pick'] = True
    else:
        df_group['overall_pick'] = False

    # --- 7. Scarcity Tag (Low Stock) ---
    # Bernoulli with p_scarcity = 0.5, select from products with neither sponsored nor overall pick
    if np.random.rand() < 0.5:
        eligible = df_group[~(df_group['sponsored'] | df_group['overall_pick'])]
        df_group['low_stock'] = False
        if not eligible.empty:
            stock_idx = np.random.choice(eligible.index)
            df_group.loc[stock_idx, 'low_stock'] = True
    else:
        df_group['low_stock'] = False

    # --- 8. Add missing columns ---
    df_group['best_seller'] = False
    df_group['limited_time'] = False
    df_group['discounted'] = False
    df_group['stock_quantity'] = np.random.randint(5, 50, len(df_group))
    
    # Set stock_quantity to low values for low_stock items
    df_group.loc[df_group['low_stock'], 'stock_quantity'] = np.random.randint(1, 5, sum(df_group['low_stock']))
    
    # --- 9. Experiment tracking columns ---
    df_group['experiment_label'] = EXPERIMENT_LABEL
    df_group['experiment_number'] = experiment_id
    df_group['position_in_experiment'] = range(len(df_group))
    df_group['assigned_position'] = range(len(df_group))
    df_group['desired_choice'] = -1 # -1 means not applicable

    return df_group


def generate_data():
    df_skus = pd.read_csv("local_datasets/skus.csv")

    # Number of experiments
    # version 1: original data
    # version 2: rating perturbation from -0.8 to 0.8 to -0.4 to 0.4
    # version 3: set permutation_only to True
    VERSION = 3
    NUM_OF_EXPERIMENTS = 500
    PERMUTATION_ONLY = True  # Only permute when changing data
    EXPERIMENT_LABEL = f'master_experiment_v{VERSION}_permutation_only_{PERMUTATION_ONLY}'.lower()

    all_experiments = []

    for i in range(NUM_OF_EXPERIMENTS):
        np.random.seed(i)
        df_iter = df_skus.copy()
        df_iter = df_iter.groupby(by='query', group_keys=False).apply(
            lambda x: apply_generation(x, i, PERMUTATION_ONLY, EXPERIMENT_LABEL).assign(query=x.name), include_groups=False
        )
        all_experiments.append(df_iter)

    # Combine all into one DataFrame
    df_all = pd.concat(all_experiments, ignore_index=True)
    
    # Reorder columns to match expected format
    expected_columns = [
        'sku', 'title', 'url', 'image_url', 'sponsored', 'rating', 'rating_count', 
        'price', 'overall_pick', 'best_seller', 'limited_time', 'discounted', 
        'low_stock', 'stock_quantity', 'experiment_label', 'query', 
        'experiment_number', 'position_in_experiment', 'assigned_position', 
        'desired_choice', 'original_price', 'original_rating', 'original_rating_count'
    ]
    
    # Select only the columns we need in the right order
    df_all = df_all[expected_columns]
    
    # Save to CSV
    df_all.to_csv(f"local_datasets/{EXPERIMENT_LABEL}.csv", index=False)
    print(f"Generated dataset saved to local_datasets/{EXPERIMENT_LABEL}.csv")
    print(f"Dataset shape: {df_all.shape}")

if __name__ == "__main__":
    generate_data()

# command to run the script
# uv run python experiments/data_generation.py
# uv run run.py --local-dataset local_datasets/master_dataset.csv --runtime-type batch --include gpt-4.1