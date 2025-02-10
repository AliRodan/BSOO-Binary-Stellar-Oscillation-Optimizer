# -*- coding: utf-8 -*-
"""
%___________________________________________________________________________________________%
%  Binary Stellar Oscillation Optimizer (BSOO) source codes demo version 1.0                %
%                                                                                           %                                                             %
%                                                                                           %
%  Author and programmer: Ali Rodan                                                         %
%                         e-Mail: alirodan@gmail.com                                        %
%                         Homepages:                                                        %
%                         1- https://scholar.google.co.uk/citations?user=n8Z3RMwAAAAJ&hl=en %
%                         2- https://www.researchgate.net/profile/Ali-Rodan                 %
%                                                                                           %
%   Paper Title:A Novel Binary Stellar Oscillation Optimizer for Feature Selection          % 
%               Optimization Problems.                                                      %
%                                                                                           %
%___________________________________________________________________________________________%
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def main():
    """
    Main
    ----
    1) Load the dataset dataset
    2) Create a hold-out partition (80% train, 20% test)
    3) Define BSOO settings
    4) Run BSOO multiple times
    5) Print summarized results
    6) Plot the convergence curve
    """

    # 1) Load dataset e.g. "Exactly.mat"
    data_file = 'Exactly.mat'
    loaded_data = scipy.io.loadmat(data_file)
    if 'Exactly' not in loaded_data:
        raise ValueError(f'Variable "Exactly" not found in {data_file}.')
    
    data_matrix = loaded_data['Exactly']
    feat  = data_matrix[:, :-1]
    label = data_matrix[:, -1]

    # 2) Partition for train/test (hold-out = 0.2 => 80% train, 20% test)
    X_train, X_test, y_train, y_test, idx_train, idx_test = custom_train_test_split(feat, label, test_size=0.2)
    
    # 3) Define BSOO settings
    opts = {
        'N':     30,     # population size
        'T':     50,     # max iterations
        'thres': 0.5,    # threshold for binarization
        'k':     5,      # for KNN
        'ws':    np.array([0.99, 0.01])  # [alpha; beta] for cost function
    }
    # We store the partition indices in opts, so "evaluateKNN" can replicate the hold-out
    opts['trainIdx'] = idx_train
    opts['testIdx']  = idx_test

    # 4) Run BSOO multiple times
    num_runs = 3
    all_fitness = np.zeros(num_runs)
    all_accuracy = np.zeros(num_runs)
    all_num_features = np.zeros(num_runs, dtype=int)
    all_curves = []  # store convergence (light curve) from each run

    best_run = -1
    best_fitness = np.inf

    for r in range(num_runs):
        result = binary_stellar_oscillation_optimizer(feat, label, opts)

        # Final fitness = last value in the 'light_curve'
        final_fit = result['c'][-1]
        all_fitness[r] = final_fit

        # Evaluate classification accuracy with the chosen features
        chosen_feats = result['sf']  # indices of selected features
        acc, _, _, _, _ = evaluate_knn(feat[:, chosen_feats], label, opts)
        all_accuracy[r] = acc

        # Number of selected features
        num_sel = len(chosen_feats)
        all_num_features[r] = num_sel

        # Store the convergence curve
        all_curves.append(result['c'])

        # Print run info (fitness)
        print(f'Run {r+1} | Fitness = {final_fit:.9f} | Accuracy = {acc*100:.2f}% | #Features = {num_sel}')

        # Track best run
        if final_fit < best_fitness:
            best_fitness = final_fit
            best_run = r

    # 5) Print results
    mean_fit = np.mean(all_fitness)
    std_fit  = np.std(all_fitness)
    min_fit  = np.min(all_fitness)
    max_fit  = np.max(all_fitness)

    mean_acc = np.mean(all_accuracy)
    std_acc  = np.std(all_accuracy)
    min_acc  = np.min(all_accuracy)
    max_acc  = np.max(all_accuracy)

    mean_num = np.mean(all_num_features)
    std_num  = np.std(all_num_features)
    min_num  = np.min(all_num_features)
    max_num  = np.max(all_num_features)

    print(f'\n=== SUMMARY ACROSS {num_runs} RUNS ===')
    print(f'Fitness: mean={mean_fit:.9f}, std={std_fit:.9f}, min={min_fit:.9f}, max={max_fit:.9f}')
    print(f'Accuracy: mean={mean_acc*100:.2f}%, std={std_acc*100:.2f}%, '
          f'min={min_acc*100:.2f}%, max={max_acc*100:.2f}%')
    print(f'#Features: mean={mean_num:.2f}, std={std_num:.2f}, min={min_num}, max={max_num}')

    # 6) Plot the convergence curve
    plt.figure()
    plt.plot(all_curves[best_run], linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Convergence Curve')
    plt.grid(True)
    plt.show()


def binary_stellar_oscillation_optimizer(feat, label, opts):
    """
    Binary Stellar Oscillation Optimizer (BSOO)
    Returns:
      sf  : indices of selected features
      ff  : the selected subset of features
      nf  : number of selected features
      c   : the light_curve of best fitness across iterations
      f   : original feat
      l   : original label
    """
    # Extract main parameters
    lb    = 0.0
    ub    = 1.0
    thres = opts.get('thres', 0.5)

    star_osc_no = opts.get('N', 10)
    m_iter      = opts.get('T', 100)

    # Initialize population
    star_positions = initialization(star_osc_no, feat.shape[1], lb, ub)
    updated_star_positions = np.copy(star_positions)

    # Best global
    best_phase_position = np.zeros(feat.shape[1])
    best_luminosity = np.inf

    # For tracking top solutions
    top_star_positions = np.zeros((3, feat.shape[1]))
    top_luminosities   = np.full(3, np.inf)
    current_luminosities = np.zeros(star_osc_no)

    # Evaluate initial population
    for i in range(star_osc_no):
        bin_pos = star_positions[i,:] > thres
        current_lum = fitness(feat, label, bin_pos, opts)
        current_luminosities[i] = current_lum

        # Update top 3
        combo_lums = np.concatenate([top_luminosities, [current_lum]])
        combo_pos  = np.vstack([top_star_positions, star_positions[i,:]])
        idx = np.argsort(combo_lums)
        sorted_lums = combo_lums[idx]
        sorted_pos  = combo_pos[idx,:]

        top_luminosities   = sorted_lums[:3]
        top_star_positions = sorted_pos[:3,:]

    best_primary_luminosity = np.inf
    best_primary_position   = np.zeros(feat.shape[1])

    initial_period = 3.0
    light_curve_iter = 0
    light_curve = np.zeros(m_iter)

    # Main loop
    for it in range(m_iter):
        current_period = initial_period + 0.001*(it+1)
        current_angular_frequency = 2.0*np.pi / current_period
        scaling_factor = 2.0 - (it+1)*(2.0/m_iter)

        # (1) Update positions via oscillation
        for i in range(star_osc_no):
            for j in range(feat.shape[1]):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()

                osc_pos1 = (best_phase_position[j]
                            - (r1*r3)*((current_angular_frequency*scaling_factor*r1 - scaling_factor) *
                               (updated_star_positions[i,j] 
                                - abs(r1*np.sin(r2)*abs(r3*best_phase_position[j])))))

                osc_pos2 = (best_phase_position[j]
                            - (r2*r3)*((current_angular_frequency*scaling_factor*r1 - scaling_factor) *
                               (updated_star_positions[i,j] 
                                - abs(r1*np.cos(r2)*abs(r3*best_phase_position[j])))))

                updated_star_positions[i,j] = r3 * (osc_pos1 + osc_pos2/2.0)
            
            # Bound
            updated_star_positions[i,:] = np.clip(updated_star_positions[i,:], lb, ub)

            # Evaluate
            bin_pos = updated_star_positions[i,:] > thres
            current_lum = fitness(feat, label, bin_pos, opts)

            # Update best
            if current_lum < best_primary_luminosity:
                best_primary_luminosity = current_lum
                best_primary_position   = np.copy(updated_star_positions[i,:])

        # (2) Perform oscillatory movement
        for i in range(star_osc_no):
            avg_top_star_position = np.mean(top_star_positions, axis=0)
            # pick 3 random distinct indices
            rand_indices = np.random.choice([x for x in range(star_osc_no) if x!=i], size=3, replace=False)

            rFactor = np.random.rand()
            oscillation_position = avg_top_star_position + 0.5*(
                np.sin(rFactor*np.pi)*(star_positions[rand_indices[0],:] - star_positions[rand_indices[1],:]) +
                np.cos((1-rFactor)*np.pi)*(star_positions[rand_indices[0],:] - star_positions[rand_indices[2],:])
            )

            star_update_position = np.copy(star_positions[i,:])
            for j in range(feat.shape[1]):
                if np.random.rand() <= 0.5:
                    star_update_position[j] = oscillation_position[j]

            star_update_position = np.clip(star_update_position, lb, ub)

            # Evaluate
            bin_pos = star_update_position > thres
            new_lum = fitness(feat, label, bin_pos, opts)

            if new_lum < current_luminosities[i]:
                star_positions[i,:] = star_update_position
                current_luminosities[i] = new_lum

                # update top 3
                combo_lums = np.concatenate([top_luminosities, [new_lum]])
                combo_pos  = np.vstack([top_star_positions, star_update_position])
                idx2 = np.argsort(combo_lums)
                sorted_lums2 = combo_lums[idx2]
                sorted_pos2  = combo_pos[idx2,:]

                top_luminosities   = sorted_lums2[:3]
                top_star_positions = sorted_pos2[:3,:]

        # (3) Compare best vs population
        best_secondary_luminosity = np.min(current_luminosities)
        idx2 = np.argmin(current_luminosities)
        best_secondary_position = star_positions[idx2, :]

        if best_primary_luminosity <= best_secondary_luminosity:
            best_overall_luminosity = best_primary_luminosity
            best_overall_position   = np.copy(best_primary_position)
        else:
            best_overall_luminosity = best_secondary_luminosity
            best_overall_position   = np.copy(best_secondary_position)

        best_luminosity = best_overall_luminosity
        best_phase_position = best_overall_position

        light_curve[it] = best_luminosity
        light_curve_iter += 1

    # Binarize final best
    best_phase_bin = best_phase_position > thres
    selected_feats = np.where(best_phase_bin)[0]
    # sFeat = feat[:, selected_feats]

    BSOO = {
        'sf': selected_feats,        # indices of selected features
        'ff': feat[:, selected_feats],  # the selected features
        'nf': len(selected_feats),
        'c':  light_curve,
        'f':  feat,
        'l':  label
    }
    return BSOO


def initialization(N, dim, lb, ub):

    return lb + (ub - lb)*np.random.rand(N, dim)


def fitness(feat, label, bin_pos, opts):
    """
    Fitness function
    = alpha*(error) + beta*(#selected / #total)
    """
    ws = opts.get('ws', np.array([0.99, 0.01]))
    alpha, beta = ws[0], ws[1]

    if np.sum(bin_pos) == 0:
        # If no features selected, cost = 1 or large number
        return 1.0
    else:
        err_rate = wrapper_knn(feat[:, bin_pos], label, opts)
        num_sel = np.sum(bin_pos)
        max_feat = len(bin_pos)
        cost_val = alpha*err_rate + beta*(num_sel/max_feat)
        return cost_val


def wrapper_knn(sFeat, label, opts):
    """
    KNN error. 
      errorVal = 1 - accuracy
    """
    k_val = opts.get('k', 5)
    idx_train = opts['trainIdx']
    idx_test  = opts['testIdx']

    X_train = sFeat[idx_train,:]
    y_train = label[idx_train]
    X_test  = sFeat[idx_test,:]
    y_test  = label[idx_test]

    mdl = KNeighborsClassifier(n_neighbors=k_val)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    acc = np.sum(y_pred == y_test) / len(y_test)
    return 1.0 - acc


def evaluate_knn(Xfeat, Ylabel, opts):
    """
    Evaluate classification accuracy (and confusion matrix)
    Returns (acc, TP, TN, FP, FN)
    """
    idx_train = opts['trainIdx']
    idx_test  = opts['testIdx']
    
    X_train = Xfeat[idx_train,:]
    y_train = Ylabel[idx_train]
    X_test  = Xfeat[idx_test,:]
    y_test  = Ylabel[idx_test]

    k_val = opts.get('k', 5)
    mdl = KNeighborsClassifier(n_neighbors=k_val)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    # If the test set had only one class, we might get a 1x1 confusion matrix
    if cm.shape[0] < 2:
        if len(y_test) > 1:
            # Means all y_test are the same class
            new_cm = np.array([[cm[0,0], 0],[0,0]])
            cm = new_cm
        else:
            # Single sample in test set
            new_cm = np.zeros((2,2), dtype=int)
            true_label = int(y_test[0])
            pred_label = int(y_pred[0])
            # clip the label to 0/1 if needed
            if true_label>1: true_label=1
            if pred_label>1: pred_label=1
            new_cm[true_label,pred_label] = 1
            cm = new_cm

    TP = cm[0,0]
    FN = cm[0,1]
    FP = cm[1,0]
    TN = cm[1,1]

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    return acc, TP, TN, FP, FN


def custom_train_test_split(feat, label, test_size=0.2, random_state=None):
    N = feat.shape[0]
    indices = np.arange(N)
    X_train, X_test, y_train, y_test, idx_train, idx_test = \
        train_test_split(feat, label, indices, test_size=test_size, 
                         random_state=random_state, stratify=label)
    # X_train, y_train => training set, X_test, y_test => testing set
    # idx_train, idx_test => original row indices
    return X_train, X_test, y_train, y_test, idx_train, idx_test


if __name__ == "__main__":
    main()
