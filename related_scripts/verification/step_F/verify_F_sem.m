%% Step F verification: SEM fitting on PCR residuals (January, tas)
%  Uses Step C/D calibration residuals to fit SEM, then compares with Python.
%  Outputs: lambda_hat, sigma2_hat, eps_mat, W_best

clear; close all;

%% load Step C/D data and compute residuals
out_dir = 'D:\Dataset\DPastCliM-NA\verification\step_F\matlab';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

% load station predictions and observations from Step C/D
ml_dir_D = 'D:\Dataset\DPastCliM-NA\verification\step_D\matlab';
fid = fopen(fullfile(ml_dir_D, 'yhat_full.bin'), 'r');
hdr = fread(fid, 2, 'int32');
T = hdr(1); n_st = hdr(2);
yhat = fread(fid, [n_st, T], 'single')';
fclose(fid);

fid = fopen(fullfile(ml_dir_D, 'station_ids.txt'), 'r');
station_ids = textscan(fid, '%s');
station_ids = station_ids{1};
fclose(fid);

% load observations (same as Step C)
obs = parquetread('D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_obs.parquet');
obs_jan = obs(obs.month == 1, :);
obs_jan = obs_jan(obs_jan.year >= 1875 & obs_jan.year <= 1999, :);

% build obs matrix
[unique_ids, ~, station_idx] = unique(obs_jan.ID);
[unique_years, ~, year_idx] = unique(obs_jan.year);
Y_mat = nan(length(unique_years), length(unique_ids), 'single');
for i = 1:height(obs_jan)
    Y_mat(year_idx(i), station_idx(i)) = obs_jan.value(i);
end

% load ESM time info to align years
trace_tas = 'D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc';
time_ka = double(ncread(trace_tas, 'time'));
cal_year_frac = 1950 + time_ka * 1000;
months_since_0 = round(cal_year_frac * 12);
year_arr = int32(floor(months_since_0 / 12));
month_arr = int32(mod(months_since_0, 12) + 1);
cal_mask = (year_arr >= 1875) & (year_arr <= 1999) & (month_arr == 1);
years_jan_cal = year_arr(cal_mask);

[common_years, ia, ib] = intersect(unique_years, years_jan_cal);
Y_aligned = Y_mat(ia, :);
valid_count = sum(~isnan(Y_aligned), 1);
keep = valid_count >= 30;

% match station IDs
keep_indices = find(keep); % positions of keep=true in original 11326-element unique_ids
[common_ids, ia2, ib2] = intersect(string(station_ids), unique_ids(keep));
Y_obs = Y_aligned(:, keep_indices(ib2))'; % (S, T) — map ib2 through keep_indices
Yhat = yhat(:, ia2)'; % (S, T)

% save intersection station IDs for Python comparison
fid_ids = fopen(fullfile(out_dir, 'common_station_ids.txt'), 'w');
for i = 1:length(common_ids)
    fprintf(fid_ids, '%s\n', common_ids{i});
end
fclose(fid_ids);
n_stations = size(Y_obs, 1);
fprintf('Full dataset: %d stations, %d years\n', n_stations, T);

% compute residuals
residuals = Y_obs - Yhat; % (S, T)

%% load station metadata and project
obs_meta = parquetread('D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_meta.parquet');
id_list = string(obs_meta.ID);
lat_list = obs_meta.lat;
lon_list = obs_meta.lon;

matched_station_ids = station_ids(ia2);
[~, meta_idx] = intersect(id_list, string(matched_station_ids));
st_lat = lat_list(meta_idx);
st_lon = lon_list(meta_idx);
[x_proj, y_proj] = albers_forward(st_lon, st_lat);

%% Subsample for speed
max_stations = 500;
if n_stations > max_stations
    rng(42);
    sub_idx = sort(randperm(n_stations, max_stations));
    residuals = residuals(sub_idx, :);
    x_proj = x_proj(sub_idx);
    y_proj = y_proj(sub_idx);
    n_stations = max_stations;
    fprintf('Subsampled to %d stations for SEM fitting\n', n_stations);
    % save sub_idx so Python can use the exact same subset
    fid_idx = fopen(fullfile(out_dir, 'sub_idx.bin'), 'w');
    fwrite(fid_idx, sub_idx, 'int32');
    fclose(fid_idx);
end

%% SEM fitting (following Guaita's approach)
D = sqrt((x_proj - x_proj').^2 + (y_proj - y_proj').^2);
thresholds = linspace(25000, 100000, 4);
penalty_weight = 0.1;
valid_mask = sum(~isnan(residuals), 2) >= 30;

fprintf('residuals(1,1:5): %.4f %.4f %.4f %.4f %.4f\n', residuals(1,1:5));
fprintf('valid_mask sum: %d\n', sum(valid_mask));
fprintf('NaN fraction: %.4f\n', sum(isnan(residuals(:)))/numel(residuals));
% save residuals and coords for Python comparison
fid_r = fopen(fullfile(out_dir, 'residuals_sub.bin'), 'w');
fwrite(fid_r, [n_stations, T], 'int32');
for s = 1:n_stations
    fwrite(fid_r, residuals(s,:), 'single');
end
fclose(fid_r);

fid_c = fopen(fullfile(out_dir, 'coords_sub.bin'), 'w');
fwrite(fid_c, [x_proj, y_proj], 'double');
fclose(fid_c);

best_nLL = Inf;
lambda_hat = 0.5;
W_best = eye(n_stations);
threshold_best = 50000;

for th_idx = 1:length(thresholds)
    th = thresholds(th_idx);
    h = th;
    W = exp(-(D.^2) / (2 * h^2));
    W(eye(n_stations) == 1) = 0;
    row_sums = sum(W, 2);
    row_sums(row_sums == 0) = 1;
    W = W ./ row_sums;

    eigvals = eig(W);
    rho = max(abs(eigvals));
    fprintf('  th=%d: rho=%.6f\n', th, rho);
    if rho >= 1
        W = W / (rho + 1e-2);
        fprintf('  W rescaled by %.6f\n', rho+1e-2);
    end

    % bounded optimization for lambda
    fun = @(lam) profileNLL(lam, residuals, W, valid_mask, T, penalty_weight);
    options = optimset('TolX', 1e-6, 'MaxIter', 200);
    [lam_opt, fval] = fminbnd(fun, 0, 0.999, options);

    fprintf('  threshold=%.0f, lambda=%.4f, nLL=%.2f\n', th, lam_opt, fval);

    if fval < best_nLL
        best_nLL = fval;
        lambda_hat = lam_opt;
        W_best = W;
        threshold_best = th;
    end
end

fprintf('Best: threshold=%.0f, lambda=%.6f\n', threshold_best, lambda_hat);

% debug: evaluate nLL at fixed lambda values for threshold=100000
h100 = 100000;
W100 = exp(-(D.^2)/(2*h100^2));
W100(eye(n_stations)==1) = 0;
rs = sum(W100,2); rs(rs==0)=1; W100 = W100./rs;
ev = eig(W100); rho100 = max(abs(ev));
if rho100 >= 1, W100 = W100/(rho100+1e-2); end
for test_lam = [0.1, 0.3, 0.5, 0.7, 0.738, 0.800, 0.85, 0.9]
    nll = profileNLL(test_lam, residuals, W100, valid_mask, T, penalty_weight);
    fprintf('  debug lambda=%.3f -> nLL=%.2f\n', test_lam, nll);
end

% save W_best for comparison
fid_w = fopen(fullfile(out_dir, 'W_best.bin'), 'w');
fwrite(fid_w, W_best(:), 'double');
fclose(fid_w);

%% compute eps_mat and sigma2
A_best = eye(n_stations) - lambda_hat * W_best;
eps_mat = nan(n_stations, T);
for t = 1:T
    res_t = residuals(:, t);
    valid = ~isnan(res_t);
    if sum(valid) < 2, continue; end
    A_sub = A_best(valid, valid);
    eps_mat(valid, t) = A_sub * res_t(valid);
end
sigma2_hat = sum(eps_mat.^2, 2, 'omitnan') ./ sum(~isnan(eps_mat), 2);

%% save outputs
fid = fopen(fullfile(out_dir, 'sem_params.bin'), 'w');
fwrite(fid, lambda_hat, 'double');
fwrite(fid, threshold_best, 'double');
fwrite(fid, n_stations, 'int32');
fwrite(fid, T, 'int32');
fwrite(fid, sigma2_hat, 'double');
fclose(fid);

% eps_mat: (S, T) -> write row by row
fid = fopen(fullfile(out_dir, 'eps_mat.bin'), 'w');
fwrite(fid, [n_stations, T], 'int32');
for s = 1:n_stations
    fwrite(fid, eps_mat(s, :), 'single');
end
fclose(fid);

fprintf('Step F outputs saved to: %s\n', out_dir);

%% helper functions
function nLL = profileNLL(lam, residual_mat, W, valid_mask, nTime, alpha)
    S = size(W, 1);
    A = eye(S) - lam * W;
    eps_mat = nan(S, nTime);
    for t = 1:nTime
        res_t = residual_mat(:, t);
        valid = ~isnan(res_t);
        if sum(valid) < 2, continue; end
        eps_mat(valid, t) = A(valid, valid) * res_t(valid);
    end
    sigma2 = sum(eps_mat.^2, 2, 'omitnan') ./ sum(~isnan(eps_mat), 2);
    if any(sigma2(valid_mask) <= 0)
        nLL = 1e30;
        return;
    end
    vm_idx = find(valid_mask);
    S_v = length(vm_idx);
    A_sub = A(vm_idx, vm_idx);
    sigma2_v = sigma2(vm_idx);
    if any(sigma2_v <= 0 | isnan(sigma2_v))
        nLL = 1e30;
        return;
    end
    try
        U = chol(A_sub);
        logdetA = 2 * sum(log(diag(U)));
    catch
        logdetA = log(max(abs(det(A_sub)), eps));
    end
    term1 = -(nTime/2) * sum(log(sigma2_v));
    term2 = nTime * logdetA;
    quad = 0;
    for t = 1:nTime
        et = eps_mat(vm_idx, t);
        vt = ~isnan(et);
        quad = quad + sum(et(vt).^2 ./ sigma2_v(vt));
    end
    term3 = -0.5 * quad;
    logLik = term1 + term2 + term3;
    penalty = -alpha * log(max(1 - lam, eps));
    nLL = -(logLik - penalty);
end

function [x, y] = albers_forward(lon_deg, lat_deg)
    a = 6378137.0; f = 1/298.257222101; e2 = 2*f - f^2; e = sqrt(e2);
    phi1 = 29.5*pi/180; phi2 = 45.5*pi/180; phi0 = 23*pi/180; lam0 = -96*pi/180;
    m1 = cos(phi1)/sqrt(1-e2*sin(phi1)^2);
    m2 = cos(phi2)/sqrt(1-e2*sin(phi2)^2);
    q0 = (1-e2)*(sin(phi0)/(1-e2*sin(phi0)^2) - log((1-e*sin(phi0))/(1+e*sin(phi0)))/(2*e));
    q1 = (1-e2)*(sin(phi1)/(1-e2*sin(phi1)^2) - log((1-e*sin(phi1))/(1+e*sin(phi1)))/(2*e));
    q2 = (1-e2)*(sin(phi2)/(1-e2*sin(phi2)^2) - log((1-e*sin(phi2))/(1+e*sin(phi2)))/(2*e));
    n = (m1^2-m2^2)/(q2-q1); C = m1^2 + n*q1; rho0 = a*sqrt(C-n*q0)/n;
    phi = lat_deg(:)*pi/180; lam = lon_deg(:)*pi/180;
    q = (1-e2)*(sin(phi)./(1-e2*sin(phi).^2) - log((1-e*sin(phi))./(1+e*sin(phi)))/(2*e));
    rho = a*sqrt(C-n*q)/n; theta = n*(lam-lam0);
    x = rho.*sin(theta); y = rho0 - rho.*cos(theta);
end
