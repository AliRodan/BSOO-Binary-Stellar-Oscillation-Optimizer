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

function BSOO = BinaryStellarOscillationOptimizer(feat, label, opts)

% Extract main parameters
    lb    = 0;
    ub    = 1; 
    thres = 0.5;

    if isfield(opts,'N'),      StarOsc_no = opts.N;     else, StarOsc_no = 10; end
    if isfield(opts,'T'),      m_iter     = opts.T;     else, m_iter     = 100; end
    if isfield(opts,'thres'),  thres      = opts.thres; end

    % Objective (fitness) function handle
    fun = @Fitness;

    % Number of dimensions (features)
    dim = size(feat, 2);

    % BSOO initialization
    best_phase_position = zeros(1, dim); 
    best_luminosity     = inf; 
    star_positions      = initialization(StarOsc_no, dim, ub, lb);
    initial_period      = 3.0;

    light_curve_iter    = 1;  
    light_curve         = zeros(1, m_iter); 

    top_star_positions  = zeros(3, dim);
    top_luminosities    = inf(1, 3);
    current_luminosities= zeros(1, StarOsc_no);

    updated_star_positions = star_positions;

    % Initial evaluation
    for i = 1:StarOsc_no
        binary_position         = star_positions(i,:) > thres;
        current_luminosities(i) = fun(feat, label, binary_position, opts);

        combo_lums  = [top_luminosities, current_luminosities(i)];
        combo_pos   = [top_star_positions; star_positions(i,:)];
        [sortedLums, idx] = sort(combo_lums);
        sortedPos   = combo_pos(idx, :);

        top_luminosities   = sortedLums(1:3);
        top_star_positions = sortedPos(1:3, :);
    end

    best_primary_luminosity = inf;
    best_primary_position   = zeros(1, dim);

    % Main loop
    while light_curve_iter <= m_iter
        %% (1) Update positions (oscillation)
        current_period           = initial_period + 0.001*light_curve_iter;
        current_angular_frequency= 2*pi / current_period;
        scaling_factor           = 2 - light_curve_iter*(2/m_iter);

        for i = 1:StarOsc_no
            for j = 1:dim
                r1 = rand(); r2 = rand(); r3 = rand();

                osc_pos1 = best_phase_position(j) - (r1*r3)* ...
                    ((current_angular_frequency*scaling_factor*r1 - scaling_factor)* ...
                      (updated_star_positions(i, j) - ...
                       abs(r1*sin(r2)*abs(r3*best_phase_position(j)))));

                osc_pos2 = best_phase_position(j) - (r2*r3)* ...
                    ((current_angular_frequency*scaling_factor*r1 - scaling_factor)* ...
                      (updated_star_positions(i, j) - ...
                       abs(r1*cos(r2)*abs(r3*best_phase_position(j)))));

                updated_star_positions(i, j) = r3 * (osc_pos1 + osc_pos2 / 2);
            end

            % Bound
            updated_star_positions(i,:) = min(max(updated_star_positions(i,:), lb), ub);

            % Evaluate
            bin_pos          = updated_star_positions(i,:) > thres;
            current_lum      = fun(feat, label, bin_pos, opts);

            % Update best 
            if current_lum < best_primary_luminosity
                best_primary_luminosity = current_lum;
                best_primary_position   = updated_star_positions(i,:);
            end
        end

        %% (2) Perform oscillatory movement
        for i = 1:StarOsc_no
            avg_top_star_position = mean(top_star_positions, 1);
            random_indices = randperm(StarOsc_no, 3);
            while any(random_indices == i)
                random_indices = randperm(StarOsc_no, 3);
            end

            rFactor = rand();
            oscillation_position = avg_top_star_position + 0.5 * ...
               ( sin(rFactor*pi) * ...
                 (star_positions(random_indices(1),:) - star_positions(random_indices(2),:)) + ...
                 cos((1-rFactor)*pi) * ...
                 (star_positions(random_indices(1),:) - star_positions(random_indices(3),:)) );

            star_update_position = star_positions(i, :);
            for j = 1:dim
                if rand() <= 0.5
                    star_update_position(j) = oscillation_position(j); 
                end
            end

            % Bound
            star_update_position = min(max(star_update_position, lb), ub);

            % Evaluate
            bin_pos      = star_update_position > thres;
            new_lum      = fun(feat, label, bin_pos, opts);

            if new_lum < current_luminosities(i)
                star_positions(i, :)    = star_update_position;
                current_luminosities(i) = new_lum;

                combo_lums = [top_luminosities, new_lum];
                combo_pos  = [top_star_positions; star_update_position];

                [sortedLums2, idx2] = sort(combo_lums);
                sortedPos2          = combo_pos(idx2, :);

                top_luminosities   = sortedLums2(1:3);
                top_star_positions = sortedPos2(1:3, :);
            end
        end

        % (3) Compare best vs population 
        [best_secondary_luminosity, idx2] = min(current_luminosities);
        best_secondary_position = star_positions(idx2, :);

        if best_primary_luminosity <= best_secondary_luminosity
            best_overall_luminosity = best_primary_luminosity;
            best_overall_position   = best_primary_position;
        else
            best_overall_luminosity = best_secondary_luminosity;
            best_overall_position   = best_secondary_position;
        end

        best_luminosity     = best_overall_luminosity;
        best_phase_position = best_overall_position;

        light_curve(light_curve_iter) = best_luminosity;
        light_curve_iter = light_curve_iter + 1;
    end

    % Binarize the global best
    best_phase_position_binary = best_phase_position > thres;
    idxFeat = 1:dim;
    Sf      = idxFeat(best_phase_position_binary == 1);
    sFeat   = feat(:, Sf);

    % Return
    BSOO.sf = Sf;
    BSOO.ff = sFeat;
    BSOO.nf = length(Sf);
    BSOO.c  = light_curve;
    BSOO.f  = feat;
    BSOO.l  = label;
end

%% ------------------------------------------------------------------------
function X = initialization(N, dim, ub, lb)
    X = lb + (ub - lb) * rand(N, dim);
end

%% ------------------------------------------------------------------------
% FITNESS FUNCTION KNN
function cost = Fitness(feat,label,X,opts)

    % By default
    ws = [0.99; 0.01];
    if isfield(opts,'ws'), ws = opts.ws; end

    if sum(X == 1) == 0
        cost = 1;  % or inf
    else
        % KNN error rate
        errorRate = wrapperKNN(feat(:,X == 1), label, opts);

        num_feat = sum(X == 1);
        max_feat = length(X);
        alpha = ws(1);
        beta  = ws(2);

        cost = alpha*errorRate + beta*(num_feat / max_feat);
    end
end

%% ------------------------------------------------------------------------
function errorVal = wrapperKNN(sFeat,label,opts)
    % KNN error
    if isfield(opts,'k'),     kVal  = opts.k;      else, kVal = 5; end
    if isfield(opts,'Model'), Model = opts.Model;
    else, error('No "Model" in opts for train/test partition.'); end

    trainIdx = Model.training;
    testIdx  = Model.test;

    xtrain = sFeat(trainIdx,:);
    ytrain = label(trainIdx);
    xvalid = sFeat(testIdx,:);
    yvalid = label(testIdx);

    mdl   = fitcknn(xtrain,ytrain,'NumNeighbors',kVal);
    ypred = predict(mdl, xvalid);

    acc  = sum(ypred == yvalid) / length(yvalid);
    errorVal = 1 - acc;
end
