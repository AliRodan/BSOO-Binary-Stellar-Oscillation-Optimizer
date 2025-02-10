%___________________________________________________________________________________________%
%  Binary Stellar Oscillation Optimizer (BSOO) source codes demo version 1.0               %
%                                                                                           %
%  Developed in MATLAB R2024a                                                               %
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


function Main()
    % Main
    % ----------
    % 1) Load Dataset
    % 2) Set up a hold-out partition
    % 3) Run the BSOO multiple times
    % 4) Print summarized results
    % 5) Plot the convergence curve
    
    clear; clc; close all;

    %% 1) Load dataset
    dataFile = 'Exactly.mat';
    loadedData = load(dataFile);
    if ~isfield(loadedData, 'Exactly')
        error('Variable "Exactly" not found in %s.', dataFile);
    end
    dataMatrix = loadedData.Exactly;
    feat  = dataMatrix(:, 1:end-1);
    label = dataMatrix(:, end);

    %% 2) Partition for training/testing
    hoRatio = 0.2;  % 80% train, 20% test
    partitionObj = cvpartition(label, 'HoldOut', hoRatio);

    %% 3) Define BSOO settings
    opts.N     = 30;        % population size
    opts.T     = 50;        % max iterations
    opts.thres = 0.5;       % threshold for binarization
    opts.Model = partitionObj;
    opts.k     = 5;         % for KNN
    opts.ws    = [0.99; 0.01]; % [alpha; beta] for cost function

    %% 4) Multiple runs
    numRuns          = 3;  
    allFitness       = zeros(numRuns,1);
    allAccuracy      = zeros(numRuns,1);
    allSelectedCount = zeros(numRuns,1);
    allCurves        = cell(numRuns,1);  % store convergence curves

    bestRun     = 0;
    bestFitness = inf;

    for r = 1:numRuns
        result = BinaryStellarOscillationOptimizer(feat, label, opts);

        % Final fitness (last value in the 'light_curve')
        finalFit = result.c(end);
        allFitness(r) = finalFit;
        
        % Evaluate classification accuracy with the chosen features
        chosenFeatIdx = result.sf;
        [acc, ~, ~, ~, ~] = evaluateKNN(feat(:, chosenFeatIdx), label, opts);
        allAccuracy(r) = acc;

        % Number of selected features
        numSel = length(chosenFeatIdx);
        allSelectedCount(r) = numSel;

        % Store the convergence curve
        allCurves{r} = result.c;

        % Display run info (fitness with 9 decimal digits)
        fprintf('Run %d | Fitness = %.9f | Accuracy = %.2f%% | #Features = %d\n',...
            r, finalFit, 100*acc, numSel);

        % Track best run
        if finalFit < bestFitness
            bestFitness = finalFit;
            bestRun     = r;
        end
    end

    %% 5) Print Results
    meanFit = mean(allFitness);
    stdFit  = std(allFitness);
    minFit  = min(allFitness);
    maxFit  = max(allFitness);

    meanAcc = mean(allAccuracy);
    stdAcc  = std(allAccuracy);
    minAcc  = min(allAccuracy);
    maxAcc  = max(allAccuracy);

    meanNum = mean(allSelectedCount);
    stdNum  = std(allSelectedCount);
    minNum  = min(allSelectedCount);
    maxNum  = max(allSelectedCount);

    fprintf('\n=== SUMMARY ACROSS %d RUNS ===\n', numRuns);
    fprintf('Fitness: mean=%.9f, std=%.9f, min=%.9f, max=%.9f\n',...
        meanFit, stdFit, minFit, maxFit);
    fprintf('Accuracy: mean=%.2f%%, std=%.2f%%, min=%.2f%%, max=%.2f%%\n',...
        100*meanAcc, 100*stdAcc, 100*minAcc, 100*maxAcc);
    fprintf('#Features: mean=%.2f, std=%.2f, min=%d, max=%d\n',...
        meanNum, stdNum, minNum, maxNum);

    %% 6) Plot the convergence curve
    figure;
    plot(allCurves{bestRun}, 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Fitness');
    title(sprintf('Convergence Curve'));
    grid on;
end

%% ------------------------------------------------------------------------
function [acc, TP, TN, FP, FN] = evaluateKNN(Xfeat, Ylabel, opts)
    %   Evaluate classification accuracy using a KNN on hold-out partition

    cpart = opts.Model;
    trainIdx = cpart.training;
    testIdx  = cpart.test;

    xTrain = Xfeat(trainIdx,:);
    yTrain = Ylabel(trainIdx);
    xTest  = Xfeat(testIdx,:);
    yTest  = Ylabel(testIdx);

    kVal = opts.k;
    mdl  = fitcknn(xTrain, yTrain, 'NumNeighbors', kVal);
    yPred= predict(mdl, xTest);

    % Confusion matrix
    cm = confusionmat(yTest, yPred);
    if size(cm,1) < 2
        if length(yTest) > 1 
            cm = [cm(1,1),0; 0,0];
        else 
            cm = [0,0;0,0];
            cm(yTest+1,yPred+1) = 1;
        end
    end

    TP = cm(1,1);
    FN = cm(1,2);
    FP = cm(2,1);
    TN = cm(2,2);

    acc = (TP + TN) / (TP + TN + FP + FN + eps);
end
