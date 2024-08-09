"""Run tests with arguments as:
python3 run_benchmarks.py func_number dim run n_cycles init_cycle"""

import argparse
import ast
import time
from datetime import timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from art import plot
from art import utility as utils
from art.core import RecommendationEngine

parser = argparse.ArgumentParser(description="Run benchmark tests.")
parser.add_argument(
    "func_number",
    metavar="func_number",
    type=int,
    choices=[1, 2, 3],
    help="Number of the benchmark function",
)
parser.add_argument(
    "dim", metavar="dim", type=int, default=2, help="Dimension of the problem"
)
parser.add_argument(
    "run",
    metavar="n_runs",
    type=int,
    default=10,
    help="Number of runs with different seed",
)
parser.add_argument(
    "n_cycles",
    metavar="n_cycles",
    type=int,
    default=10,
    help="Number of DBTL cycles to run",
)
parser.add_argument(
    "init_cycle",
    metavar="init_cycle",
    type=int,
    default=1,
    help="Which cycle to start from.",
)

args = parser.parse_args()

# Assign input parameters
func_number = args.func_number
dim = args.dim
run = args.run
n_cycles = args.n_cycles
init_cycle = args.init_cycle

# print('Reading arguments finished.')

bad_initial = False


################################################################
# Define benchmark functions
def f1(x, dimension):
    term1 = 0.0
    term2 = 0.0
    for i in range(dimension):
        term1 += (x[i] - 5) ** 2
        term2 += x[i] ** 2

    return -1 * (1 / dimension * term1 + np.exp(-term2)) + 25


def f2(x, dimension):
    f = 0.0
    for i in range(dimension):
        f += x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i]

    f *= 1 / dimension
    return -1.0 * f


def f3(x, dimension):
    f = 0.0
    for i in range(dimension):
        f += np.sqrt(x[i]) * np.sin(x[i])

    return f


################################################################
# Define dictionaries for variables
func = {1: f1, 2: f2, 3: f3}
global_optimum_value = {1: 25.0, 2: 78.332, 3: 2.808_13 * dim}
global_optimum = {
    1: 5.0 * np.ones(dim),
    2: -2.903_534 * np.ones(dim),
    3: 7.916 * np.ones(dim),
}
lb = {1: -5, 2: -5, 3: 0.0}
ub = {1: 10, 2: 5, 3: 12.0}

################################################################
# Assign specific function
func = func[func_number]
global_optimum_value = global_optimum_value[func_number]
global_optimum = global_optimum[func_number]
lb = lb[func_number]
ub = ub[func_number]

if n_cycles == 1:
    alphas = np.zeros(n_cycles).tolist()
else:
    alphas = np.linspace(0.9, 0, num=n_cycles).tolist()

    for i, alpha in enumerate(alphas):
        if alpha == 0:
            alphas[i] = ast.literal_eval("None")
        else:
            alphas[i] = round(float(alpha), 2)

# Additional variables for plotting models and sampling
n_points = 50
x1 = np.linspace(lb, ub, n_points)
x2 = np.linspace(lb, ub, n_points)
X1, Y1 = np.meshgrid(x1, x2)

################################################################
# Create file with bounds (needed only once)
bounds_file = Path(Path.cwd().parent,
    "data", "simulated_data", f"{dim}dim_benchmark_f{func_number}_bounds.csv"
)
df = pd.DataFrame(columns=['Variable', 'Min', 'Max', 'Scaling'])
df['Variable'] = ['x_' + str(i) for i in range(1, dim + 1)]
df['Min'] = lb * np.ones(dim)
df['Max'] = ub * np.ones(dim)
df['Scaling'] = np.ones(dim)
df = df.set_index('Variable')
df.to_csv(path_or_buf=bounds_file)

start = time.time()

################################################################
# Next define a dictionary that contains all of the settings that ART will use to compute its
# recommendations.
data_dir = Path(Path.cwd().parent, "data", "simulated_data")
results_dir = Path(Path.cwd().parent, "results", "simulated_data")
if bad_initial:
    data_dir = Path(data_dir, "bad_initial")
    results_dir = Path(results_dir, "bad_initial")
data_file = Path(data_dir, f"{dim}dim_benchmark_f{func_number}.csv")
output_dir = Path(results_dir,
    f"{dim}D_f{func_number}", f"{dim}D_f{func_number}_run{run}"
)

art_params = {
    "bounds_file": bounds_file,
    "input_vars": ["x_" + str(i) for i in range(1, dim + 1)],
    "response_vars": ["y"],
    "objective": "maximize",
    "threshold": 0.2,
    "verbose": 0,
    "seed": None,
    "recommend": False,
    "output_dir": output_dir,
}

if init_cycle == 1:
    mae_score_train = np.zeros(n_cycles)
    mae_score_test = np.zeros(n_cycles)
    best_prediction = np.zeros(n_cycles)
    std = np.zeros(n_cycles)
    prob_success = np.zeros((17, n_cycles))
else:
    mae_score_train = np.genfromtxt(
        str(Path(output_dir, "mae_score_train.csv")), delimiter=","
    )
    mae_score_test = np.genfromtxt(
        str(Path(output_dir, "mae_score_test.csv")), delimiter=","
    )
    best_prediction = np.genfromtxt(
        str(Path(output_dir, "best_prediction.csv")), delimiter=","
    )
    std = np.genfromtxt(str(Path(output_dir, "std.csv")), delimiter=",")
    prob_success = np.genfromtxt(
        str(Path(output_dir, "prob_success.csv")), delimiter=","
    )

results = {
    "mae_score_train": mae_score_train,
    "mae_score_test": mae_score_test,
    "best_prediction": best_prediction,
    "std": std,
    "prob_success": prob_success,
}

for cycle in range(init_cycle - 1, n_cycles):

    print(f"Run {run}\nCycle {cycle + 1}\nAlpha = {alphas[cycle]}")

    if cycle == 0:
        data_file = Path(data_dir, f"{dim}dim_benchmark_f{func_number}.csv")
    else:
        data_file = Path(output_dir, f"Data_Cycle{cycle + 1}.csv")

    art_params["alpha"] = alphas[cycle]

    df = utils.load_study(data_file=data_file)
    art = RecommendationEngine(df, **art_params)

    # Draw recommendations for the next cycle
    draws = art.parallel_tempering_opt()
    art.recommend(draws)
    file_path = Path(art.output_dir, f"recommendations_Cycle{cycle + 1}.csv")
    art.recommendations.to_csv(path_or_buf=file_path)

    # Evaluate models
    art.evaluate_models()
    mae_score_train[cycle] = art.model_df[0]["MAE"]["Ensemble Model"]
    X_test = art.recommendations.values[:, :-1]
    y_test = func(X_test.T, dim).reshape(-1, art.num_response_var)
    art.evaluate_models(X_test, y_test)
    mae_score_test[cycle] = art.model_df[0]["MAE"]["Ensemble Model"]

    # error[run, cycle] = np.abs(art.recommendations.values[0, -1] - global_optimum_value)
    best_prediction[cycle] = np.max(art.recommendations.values[:, -1])
    std[cycle] = art.post_pred_stats(art.recommendations.values[0, :-1])[1][0][0]
    cumulative_success_prob = art.calculate_success_prob(
        current_best=art.find_current_best()
    )
    prob_success[:, cycle] = cumulative_success_prob[0]

    # Save data for the next cycle
    X = np.concatenate((art.X, X_test))
    y = np.concatenate((art.y, y_test))
    file_name = Path(art.output_dir, f"Data_Cycle{cycle + 2}.csv")
    utils.save_edd_csv(X, y, art.input_vars, file_name)

    # For the first run, plot all models for each cycle
    if run == 1:
        plot.all_models_benchmark(
            art,
            func,
            dim,
            X1,
            Y1,
            lb,
            ub,
            global_optimum,
            num_L0_models=8,
            cycle=cycle,
            alpha=alphas[cycle],
        )

    # For each run, plot all models for the first cycle
    if cycle == 0:
        plot.all_models_benchmark(
            art,
            func,
            dim,
            X1,
            Y1,
            lb,
            ub,
            global_optimum,
            num_L0_models=8,
            cycle=cycle + 1,
            alpha=alphas[cycle],
        )

    # For each run and cycle save the metrics
    for name, value in results.items():
        file_name = Path(art.output_dir, f"{name}.csv")
        np.savetxt(file_name, value, fmt="%4.4f", delimiter=",", newline="\n")

# For each run save the metrics
for name, value in results.items():
    file_name = Path(art.output_dir, f"{name}.csv")
    np.savetxt(file_name, value, fmt="%4.4f", delimiter=",", newline="\n")

print("Finished.")

elapsed = time.time() - start
print(str(timedelta(seconds=elapsed)))
