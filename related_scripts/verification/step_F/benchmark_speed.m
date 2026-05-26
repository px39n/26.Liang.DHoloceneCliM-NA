%% Benchmark: MATLAB SEM + ARMA speed on 500 stations, 125 years

clear; close all;

ml_dir = 'D:\Dataset\DPastCliM-NA\verification\step_F\matlab';

%% Load data
fid = fopen(fullfile(ml_dir, 'residuals_sub.bin'), 'r');
hdr = fread(fid, 2, 'int32');
n_st = hdr(1); T = hdr(2);
residuals = fread(fid, [T, n_st], 'single')';
fclose(fid);

fid = fopen(fullfile(ml_dir, 'coords_sub.bin'), 'r');
raw = fread(fid, [n_st, 2], 'double');
fclose(fid);
x_proj = raw(:,1); y_proj = raw(:,2);

valid_mask = sum(~isnan(residuals), 2) >= 30;
fprintf('Data: %d stations, %d years, %d valid\n', n_st, T, sum(valid_mask));

%% SEM fitting
tic_sem = tic;
D = sqrt((x_proj - x_proj').^2 + (y_proj - y_proj').^2);
thresholds = linspace(25000, 100000, 4);
best_nLL = Inf; lambda_hat = 0.5; W_best = eye(n_st);
for th_idx = 1:length(thresholds)
    th = thresholds(th_idx);
    W = exp(-(D.^2)/(2*th^2));
    W(eye(n_st)==1)=0; rs=sum(W,2); rs(rs==0)=1; W=W./rs;
    ev=eig(W); rho=max(abs(ev));
    if rho>=1, W=W/(rho+1e-2); end
    options = optimset('TolX',1e-6,'MaxIter',200);
    vm_idx = find(valid_mask);
    fun = @(lam) profileNLL(lam, residuals, W, vm_idx, T, 0.1);
    [lam_opt, fval] = fminbnd(fun, 0, 0.999, options);
    if fval < best_nLL
        best_nLL = fval; lambda_hat = lam_opt; W_best = W;
    end
end
t_sem = toc(tic_sem);
fprintf('SEM fitting: %.2fs (lambda=%.4f)\n', t_sem, lambda_hat);

%% eps_mat computation
tic_eps = tic;
A_best = eye(n_st) - lambda_hat * W_best;
eps_mat = nan(n_st, T);
for t = 1:T
    res_t = residuals(:, t);
    valid = ~isnan(res_t);
    if sum(valid) < 2, continue; end
    eps_mat(valid, t) = A_best(valid,valid) * res_t(valid);
end
t_eps = toc(tic_eps);
fprintf('eps_mat: %.2fs\n', t_eps);

%% ARMA CSS fitting
tic_arma = tic;
ar_ml = zeros(n_st, 1);
ma_ml = zeros(n_st, 1);
var_ml = zeros(n_st, 1);
for i = 1:n_st
    if ~valid_mask(i), continue; end
    y = eps_mat(i,:)'; y = y(~isnan(y)); n = length(y);
    if n < 12, var_ml(i) = var(y); continue; end
    best_css = inf; bp = 0; bt = 0;
    for phi = linspace(-0.95,0.95,39)
        for theta = linspace(-0.95,0.95,39)
            e = zeros(n,1);
            for t = 2:n, e(t) = y(t) - phi*y(t-1) - theta*e(t-1); end
            css = sum(e.^2);
            if css < best_css, best_css=css; bp=phi; bt=theta; end
        end
    end
    for phi = linspace(max(bp-0.1,-0.99),min(bp+0.1,0.99),21)
        for theta = linspace(max(bt-0.1,-0.99),min(bt+0.1,0.99),21)
            e = zeros(n,1);
            for t = 2:n, e(t) = y(t) - phi*y(t-1) - theta*e(t-1); end
            css = sum(e.^2);
            if css < best_css, best_css=css; bp=phi; bt=theta; end
        end
    end
    ar_ml(i)=bp; ma_ml(i)=bt; var_ml(i)=best_css/n;
end
t_arma = toc(tic_arma);
fprintf('ARMA CSS fitting: %.2fs\n', t_arma);

%% ARMA simulation (22ka = 22000 timesteps)
n_sim = 22000;
tic_sim = tic;
rng(42);
eps_arma = zeros(n_st, n_sim);
for i = 1:n_st
    if ~valid_mask(i), continue; end
    sigma = sqrt(var_ml(i));
    eta = sigma * randn(1, n_sim);
    eps_tmp = zeros(1, n_sim);
    phi = ar_ml(i); theta = ma_ml(i);
    for t = 2:n_sim
        eps_tmp(t) = phi*eps_tmp(t-1) + theta*eta(t-1) + eta(t);
    end
    eps_arma(i,:) = eps_tmp;
end
t_sim = toc(tic_sim);
fprintf('ARMA simulation (%d timesteps): %.2fs\n', n_sim, t_sim);

%% Inverse SEM
tic_inv = tic;
A = eye(n_st) - lambda_hat * W_best;
u_mat = A \ eps_arma;
t_inv = toc(tic_inv);
fprintf('Inverse SEM (%d timesteps): %.2fs\n', n_sim, t_inv);

%% Total
total = t_sem + t_eps + t_arma + t_sim + t_inv;
fprintf('\n=== TOTAL MATLAB: %.2fs ===\n', total);
fprintf('  SEM fitting:     %.2fs\n', t_sem);
fprintf('  eps_mat:          %.2fs\n', t_eps);
fprintf('  ARMA CSS fitting: %.2fs\n', t_arma);
fprintf('  ARMA simulation:  %.2fs\n', t_sim);
fprintf('  Inverse SEM:      %.2fs\n', t_inv);

%% helper
function nLL = profileNLL(lam, residual_mat, W, vm_idx, nTime, alpha)
    S = size(W,1); A = eye(S) - lam*W;
    eps_mat = nan(S, nTime);
    for t = 1:nTime
        res_t = residual_mat(:,t); valid = ~isnan(res_t);
        if sum(valid)<2, continue; end
        eps_mat(valid,t) = A(valid,valid)*res_t(valid);
    end
    S_v = length(vm_idx);
    sigma2 = sum(eps_mat.^2,2,'omitnan')./sum(~isnan(eps_mat),2);
    sigma2_v = sigma2(vm_idx);
    if any(sigma2_v<=0|isnan(sigma2_v)), nLL=1e30; return; end
    A_sub = A(vm_idx,vm_idx);
    try U=chol(A_sub); logdetA=2*sum(log(diag(U)));
    catch, logdetA=log(max(abs(det(A_sub)),eps)); end
    term1 = -(nTime/2)*sum(log(sigma2_v));
    term2 = nTime*logdetA;
    quad = 0;
    for t = 1:nTime
        et = eps_mat(vm_idx,t); vt = ~isnan(et);
        quad = quad + sum(et(vt).^2./sigma2_v(vt));
    end
    logLik = term1 + term2 - 0.5*quad;
    nLL = -(logLik + alpha*log(max(1-lam,eps)));
end
