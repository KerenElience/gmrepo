from utils.process import DataProcess, calc_dist_matrix
from src.randomforest import RFModel, RFOptimizer
from src.simulate_annealing import get_initial_guess, SimulatedAnnealing

def main():
    ## loading data
    prcd = DataProcess("./resource/meta.csv")

    ## comfirm model numbers and which model can clearly classifiy different disease/phenotype.
    data = prcd.process({"disease":30, "relative_abundance": 80, "disease": 70})
    label = prcd.label
    disease_names = label.unique()
    x = prcd.transform(prcd.filtration(data))
    y = prcd.encoder.fit_transform(label)

    ## get best disease groups
    dist_matrix = calc_dist_matrix(x, y)
    initial_sol = get_initial_guess(dist_matrix , disease_names, min_partners = 1, max_partners = 3)
    sa = SimulatedAnnealing(x, y, disease_names)
    best_groups, best_recall = sa.solve(initial_sol)

    ## optuna every best disease group model
    
if __name__ == "__main__":
    main()