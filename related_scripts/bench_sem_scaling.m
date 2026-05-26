%% Multi-scale SEM benchmark — same operations as Python bench
% Measures: pdist, chol/logdet, matmul with NaN handling
% Run: matlab -batch "run('related_scripts/bench_sem_scaling.m')"

sizes = [50, 100, 200, 500, 1000, 1500, 2000, 3000];
T_cal = 60;
nan_frac = 0.15;
n_eval = 15;
n_thresh = 4;

fprintf('%6s | %7s | %7s | %7s | %7s | %7s | %8s | %8s\n', ...
    'n', 'pdist', 'W_bld', 'chol', 'matmul', '1eval', '4th_tot', '12mo');
fprintf('%s\n', repmat('-', 1, 85));

results = struct();
for idx = 1:length(sizes)
    n = sizes(idx);
    rng(42);
    coords = randn(n, 2) * 100000;
    R = randn(n, T_cal);
    mask = rand(n, T_cal) < nan_frac;
    R(mask) = NaN;

    % distance matrix (no toolbox needed)
    tic;
    dx = coords(:,1) - coords(:,1)';
    dy = coords(:,2) - coords(:,2)';
    D = sqrt(dx.^2 + dy.^2);
    dt_pdist = toc;

    % Build W
    tic;
    W = exp(-(D.^2) / (2 * 50000^2));
    W(logical(eye(n))) = 0;
    rs = sum(W, 2);
    rs(rs == 0) = 1;
    W = W ./ rs;
    dt_wbuild = toc;

    A = eye(n) - 0.5 * W;

    % chol (equivalent to slogdet)
    tic;
    try
        U = chol(A);
        logdetA = 2 * sum(log(diag(U)));
    catch
        logdetA = log(abs(det(A)) + eps);
    end
    dt_chol = toc;

    % matmul with NaN handling
    tic;
    eps_mat = NaN(n, T_cal);
    for t = 1:T_cal
        res_t = R(:, t);
        valid = ~isnan(res_t);
        if sum(valid) < 2
            continue;
        end
        A_sub = A(valid, valid);
        eps_mat(valid, t) = A_sub * res_t(valid);
    end
    dt_matmul = toc;

    dt_1eval = dt_chol + dt_matmul;
    dt_4th = n_thresh * (dt_pdist + dt_wbuild + n_eval * dt_1eval);
    dt_12mo = 12 * dt_4th;

    fprintf('%6d | %6.3fs | %6.3fs | %6.3fs | %6.3fs | %6.3fs | %7.1fs | %7.0fs\n', ...
        n, dt_pdist, dt_wbuild, dt_chol, dt_matmul, dt_1eval, dt_4th, dt_12mo);

    results(idx).n = n;
    results(idx).chol = dt_chol;
    results(idx).matmul = dt_matmul;
    results(idx).one_eval = dt_1eval;
    results(idx).four_thresh = dt_4th;
    results(idx).twelve_months = dt_12mo;
end

% Extrapolation
ns = [results.n];
chols = [results.chol];
a_chol = median(chols ./ ns.^3);
n_target = 5214;
chol_pred = a_chol * n_target^3;
mm_pred = median([results.matmul] ./ ns.^2) * n_target^2;
eval_pred = chol_pred + mm_pred;
th_pred = 4 * (15 * eval_pred + 2.0);
mo_pred = 12 * th_pred;

fprintf('\n--- Extrapolation to n=%d ---\n', n_target);
fprintf('  chol(%d) predicted: %.2fs\n', n_target, chol_pred);
fprintf('  matmul(%d) predicted: %.2fs\n', n_target, mm_pred);
fprintf('  1 eval: %.2fs\n', eval_pred);
fprintf('  4 thresholds: %.0fs = %.1f min\n', th_pred, th_pred/60);
fprintf('  12 months: %.0fs = %.1f hours\n', mo_pred, mo_pred/3600);

% Save results
save(fullfile('D:\Dataset\DPastCliM-NA\interim\pcr_station', 'bench_sem_matlab.mat'), 'results');
fprintf('\nDONE\n');
