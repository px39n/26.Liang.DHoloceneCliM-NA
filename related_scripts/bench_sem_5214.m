%% Direct benchmark at n=5214 — single eval
% Measures actual chol + matmul time

n = 5214;
T_cal = 60;
nan_frac = 0.15;
n_eval = 15;
n_thresh = 4;
n_months = 12;

fprintf('n=%d, T=%d, NaN fraction=%.2f\n', n, T_cal, nan_frac);

rng(42);
coords = randn(n, 2) * 100000;
R = randn(n, T_cal);
mask_nan = rand(n, T_cal) < nan_frac;
R(mask_nan) = NaN;

% Distance matrix
fprintf('Building distance matrix...\n');
tic;
dx = coords(:,1) - coords(:,1)';
dy = coords(:,2) - coords(:,2)';
D = sqrt(dx.^2 + dy.^2);
dt_pdist = toc;
fprintf('  pdist: %.3fs\n', dt_pdist);

% Build W
fprintf('Building W matrix...\n');
tic;
W = exp(-(D.^2) / (2 * 50000^2));
W(logical(eye(n))) = 0;
rs = sum(W, 2);
rs(rs == 0) = 1;
W = W ./ rs;
dt_wbuild = toc;
fprintf('  W build: %.3fs\n', dt_wbuild);

A = eye(n) - 0.5 * W;

% chol
fprintf('chol...\n');
tic;
try
    U = chol(A);
    logdetA = 2 * sum(log(diag(U)));
catch
    logdetA = log(abs(det(A)) + eps);
end
dt_chol = toc;
fprintf('  chol: %.3fs (logdet=%.2f)\n', dt_chol, logdetA);

% matmul with NaN
fprintf('matmul with NaN masking...\n');
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
fprintf('  matmul: %.3fs\n', dt_matmul);

dt_1eval = dt_chol + dt_matmul;
fprintf('\n--- Results ---\n');
fprintf('1 eval (chol + matmul): %.3fs\n', dt_1eval);

dt_1thresh = dt_pdist + dt_wbuild + n_eval * dt_1eval;
dt_1month = n_thresh * dt_1thresh;
dt_12months = n_months * dt_1month;

fprintf('\nProjection (assumes %d evals/thresh, %d thresholds):\n', n_eval, n_thresh);
fprintf('  1 threshold: %.0fs = %.1f min\n', dt_1thresh, dt_1thresh/60);
fprintf('  1 month (4 thresh): %.0fs = %.1f min\n', dt_1month, dt_1month/60);
fprintf('  12 months: %.0fs = %.2f hours\n', dt_12months, dt_12months/3600);

fprintf('\nDONE\n');
