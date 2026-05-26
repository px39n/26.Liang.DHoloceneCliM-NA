% Benchmark _profile_neg_loglik factors at production scale
% S=7220 (all stations), n_v=5214 (valid), T=60
% Matches Python factor decomposition for verification

rng(42);
S = 7220; n_v = 5214; T = 60;
fprintf('S=%d, n_v=%d, T=%d\n', S, n_v, T);

% Build synthetic data matching production structure
coords = randn(S, 2) * 100000;
x = coords(:,1); y = coords(:,2);
D = sqrt((x - x').^2 + (y - y').^2);
h = 50000;
W = exp(-(D.^2) / (2*h^2));
W(logical(eye(S))) = 0;
W = W ./ sum(W, 2);
rho = max(abs(eig(W)));
if rho >= 1
    W = W / (rho + 1e-2);
else
    W = W / (1 + 1e-2);
end
fprintf('W built: %dx%d\n', size(W));

% Residual mat with NaN pattern
residual_mat = randn(S, T);
nan_mask = rand(S, T) < 0.15;
residual_mat(nan_mask) = NaN;

vm_idx = sort(randperm(S, n_v));
lam = 0.5;

R_filled = residual_mat;
R_filled(isnan(R_filled)) = 0;

N = 3;
fprintf('\n=== Factor breakdown (avg of %d) ===\n\n', N);

% A: eye - lam*W
times_A = zeros(1, N);
for i = 1:N
    tic; A = eye(S) - lam * W; times_A(i) = toc;
end
fprintf('t_eye:  %.4fs  [eye(S)-lam*W, %dx%d]\n', mean(times_A), S, S);

% B: matmul A * R_filled
times_B = zeros(1, N);
for i = 1:N
    tic; eps_mat = A * R_filled; times_B(i) = toc;
end
fprintf('t_mm:   %.4fs  [A*R, %dx%d @ %dx%d]\n', mean(times_B), S, S, S, T);

% C: NaN restore + sigma2
times_C = zeros(1, N);
for i = 1:N
    tic;
    eps2 = eps_mat; eps2(nan_mask) = NaN;
    nvps = sum(~isnan(eps2), 2);
    eps2sq = eps2.^2; eps2sq(isnan(eps2sq)) = 0;
    s2 = sum(eps2sq, 2) ./ max(nvps, 1);
    times_C(i) = toc;
end
fprintf('t_sig:  %.4fs  [NaN restore + sigma2]\n', mean(times_C));

% D: A_sub extraction
times_D = zeros(1, N);
for i = 1:N
    tic; A_sub = A(vm_idx, vm_idx); times_D(i) = toc;
end
fprintf('t_ix:   %.4fs  [A(vm,vm), %dx%d]\n', mean(times_D), n_v, n_v);

% E: Fortran order copy (MATLAB is native F-order, so this is 0)
fprintf('t_fort: 0.0000s  [native F-order, no copy needed]\n');

% F: chol
times_F = zeros(1, N);
for i = 1:N
    A_sub2 = A(vm_idx, vm_idx);
    tic; [U, flag] = chol(A_sub2); times_F(i) = toc;
end
fprintf('t_chol: %.4fs  [chol(%dx%d), flag=%d]\n', mean(times_F), n_v, n_v, flag);

% G: logdet + terms
times_G = zeros(1, N);
for i = 1:N
    tic;
    logdetA = 2 * sum(log(diag(U)));
    s2v = s2(vm_idx);
    t1 = -(T/2) * sum(log(s2v));
    t2 = T * logdetA;
    ev = eps2(vm_idx, :);
    es = ev; es(isnan(es)) = 0;
    q = sum(sum(es.^2 ./ s2v));
    t3 = -0.5 * q;
    times_G(i) = toc;
end
fprintf('t_term: %.4fs  [logdet + likelihood]\n', mean(times_G));

total_factors = mean(times_A)+mean(times_B)+mean(times_C)+mean(times_D)+mean(times_F)+mean(times_G);
fprintf('\nFactor sum: %.3fs\n', total_factors);

% Full profile eval (5 calls)
fprintf('\n=== Full profile eval (5 calls) ===\n');
lams = [0.1, 0.3, 0.5, 0.7, 0.85];
times_full = zeros(1, 5);
for k = 1:5
    lam_k = lams(k);
    tic;
    A_k = eye(S) - lam_k * W;
    eps_k = A_k * R_filled;
    eps_k(nan_mask) = NaN;
    nvps_k = sum(~isnan(eps_k), 2);
    ek2 = eps_k.^2; ek2(isnan(ek2)) = 0;
    s2_k = sum(ek2, 2) ./ max(nvps_k, 1);
    A_sub_k = A_k(vm_idx, vm_idx);
    [U_k, ~] = chol(A_sub_k);
    ld_k = 2*sum(log(diag(U_k)));
    s2v_k = s2_k(vm_idx);
    t1_k = -(T/2)*sum(log(s2v_k));
    t2_k = T*ld_k;
    ev_k = eps_k(vm_idx,:); ev_k(isnan(ev_k))=0;
    q_k = sum(sum(ev_k.^2 ./ s2v_k));
    nll_k = -(t1_k + t2_k - 0.5*q_k);
    times_full(k) = toc;
end
fprintf('Full eval avg: %.3fs (calls: %.3f %.3f %.3f %.3f %.3f)\n', ...
    mean(times_full), times_full(1), times_full(2), times_full(3), times_full(4), times_full(5));
fprintf('Residual (full - factors): %.3fs\n', mean(times_full) - total_factors);

% Optimizer eval count — use global counter
fprintf('\n=== fminbnd eval count ===\n');
global EVAL_COUNT;
EVAL_COUNT = 0;
obj_fn = @(l) profile_nll_count(l, W, R_filled, nan_mask, vm_idx, S, T);
opts = optimset('TolX', 1e-6, 'Display', 'off');
tic;
[lam_opt, fval] = fminbnd(obj_fn, 0, 0.9, opts);
dt_opt = toc;
N_e = EVAL_COUNT;
fprintf('fminbnd: %d evals in %.1fs (%.3fs/eval), lam=%.6f\n', ...
    N_e, dt_opt, dt_opt/max(N_e,1), lam_opt);

% Projection
t_W_est = 1.0;
T_eval = dt_opt / max(N_e, 1);
T_12mo = 12 * 4 * (t_W_est + dt_opt);
fprintf('\nProjection: T_eval=%.3fs, N_e=%d, T_12mo=%.0fs=%.1fmin\n', ...
    T_eval, N_e, T_12mo, T_12mo/60);
fprintf('DONE\n');

function nll = profile_nll_count(lam, W, R_filled, nan_mask, vm_idx, S, T)
    global EVAL_COUNT;
    EVAL_COUNT = EVAL_COUNT + 1;
    A = eye(S) - lam * W;
    eps_m = A * R_filled;
    eps_m(nan_mask) = NaN;
    nvps = sum(~isnan(eps_m), 2);
    eps_sq = eps_m.^2; eps_sq(isnan(eps_sq)) = 0;
    s2 = sum(eps_sq, 2) ./ max(nvps, 1);
    A_sub = A(vm_idx, vm_idx);
    [U, flag] = chol(A_sub);
    if flag ~= 0
        nll = 1e30; return;
    end
    ld = 2*sum(log(diag(U)));
    s2v = s2(vm_idx);
    if any(s2v <= 0 | isnan(s2v))
        nll = 1e30; return;
    end
    t1 = -(T/2)*sum(log(s2v));
    t2 = T*ld;
    ev = eps_m(vm_idx,:); ev(isnan(ev))=0;
    q = sum(sum(ev.^2 ./ s2v));
    nll = -(t1 + t2 - 0.5*q - 0.1*log(max(1-lam,1e-10)));
end
