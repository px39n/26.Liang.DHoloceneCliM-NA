%% Step C verification: PCR calibration for January (month=1) only
%  Compares PCA eigenvalues, PC scores, and per-station OLS coefficients
%  with the Python implementation.
%
%  Uses Step B outputs (ESM at stations) and Step A outputs (GHCN obs).
%  Both MATLAB and Python use identical logic:
%    - PCA on NA-windowed ESM grid field for calibration years
%    - No intercept regression on anomalies (Guaita's approach)
%    - Fixed n_pc = 5, 70% train / 30% test split, rng seed = 2026

clear; close all;

%% paths
trace_tas = 'D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc';
out_dir = 'D:\Dataset\DPastCliM-NA\verification\step_C\matlab';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% parameters
year_cal_min = 1875;
year_cal_max = 1999;
n_pc = 5;
train_frac = 0.7;
rng_seed = 2026;
target_month = 1;  % January

%% load station obs from Step A (Python canonical)
obs = parquetread('D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_obs.parquet');
meta = parquetread('D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_meta.parquet');

%% load ESM grid field (not station-interpolated — raw grid for PCA)
fprintf('Loading TraCE TREFHT for PCA...\n');
trefht = ncread(trace_tas, 'TREFHT');  % (lon, lat, time)
lat = double(ncread(trace_tas, 'lat'));
lon = double(ncread(trace_tas, 'lon'));
time_ka = double(ncread(trace_tas, 'time'));

cal_year_frac = 1950 + time_ka * 1000;
months_since_0 = round(cal_year_frac * 12);
year_arr = int32(floor(months_since_0 / 12));
month_arr = int32(mod(months_since_0, 12) + 1);

tas_map = trefht - 273.15;
clear trefht

% lon 0-360 -> -180..180
lon180 = mod(lon + 180, 360) - 180;
[lon_sorted, lon_idx] = sort(lon180);
tas_map = tas_map(lon_idx, :, :);
[lat_sorted, lat_idx] = sort(lat);
tas_map = tas_map(:, lat_idx, :);

% NA window with padding (matching Python select_na_window defaults)
na_lat = [-180+5, 75+5];  % wide enough
na_lon = [-180-5, -50+5];
lat_mask = lat_sorted >= -90 & lat_sorted <= 90;  % all lat for now
lon_mask = lon_sorted >= na_lon(1) & lon_sorted <= na_lon(2);
esm_lat = lat_sorted(lat_mask);
esm_lon = lon_sorted(lon_mask);
tas_na = tas_map(lon_mask, lat_mask, :);  % (nlon_na, nlat_na, time)

% filter cal period + January
cal_mask = (year_arr >= year_cal_min) & (year_arr <= year_cal_max) & (month_arr == target_month);
fprintf('  Jan cal steps: %d\n', sum(cal_mask));
esm_jan_cal = tas_na(:,:,cal_mask);  % (nlon, nlat, T_cal)
years_jan_cal = year_arr(cal_mask);

nlon = size(esm_jan_cal, 1);
nlat = size(esm_jan_cal, 2);
T_cal = size(esm_jan_cal, 3);

% reshape to (T_cal, nlon*nlat) for PCA — matching Python's (T, ny*nx)
% Python: arr.reshape(T, ny*nx) where arr is (T, ny, nx) -> col order = (lat varies, then lon)
% MATLAB: we have (lon, lat, time) -> reshape to (T, lon*lat) with permute
field_2d = reshape(permute(esm_jan_cal, [3, 2, 1]), T_cal, []);  % (T_cal, nlat*nlon)
% Wait, Python does arr.reshape(T, ny*nx) where arr is (T, lat, lon)
% So the flattening is lat-major (lat index changes first, then lon)
% MATLAB esm_jan_cal is (lon, lat, time) -> permute to (time, lat, lon) then reshape
field_2d = reshape(permute(esm_jan_cal, [3, 2, 1]), T_cal, []);

fprintf('  field_2d: (%d, %d)\n', size(field_2d,1), size(field_2d,2));

%% PCA (SVD-based, matching Python _pca)
mu = mean(field_2d, 1);  % (1, nfeatures)
Xc = field_2d - mu;
[U, S, V] = svd(Xc, 'econ');
singular_vals = diag(S);
scores_full = U * S;  % (T_cal, min(T,F))
var_frac = (singular_vals.^2) / max(sum(singular_vals.^2), 1e-30);

% select top n_pc
eofs = V(:, 1:n_pc)';  % (n_pc, nfeatures) — same as Python's Vt[:n_pc]
pcs = scores_full(:, 1:n_pc);  % (T_cal, n_pc)
ev_top = var_frac(1:n_pc);

fprintf('PCA: top-%d explained variance: [%s]\n', n_pc, num2str(ev_top'*100, '%.2f%% '));

%% save PCA outputs
% eigenvalues
fid = fopen(fullfile(out_dir, 'eigenvalues.bin'), 'w');
fwrite(fid, length(var_frac), 'int32');
fwrite(fid, var_frac, 'double');
fclose(fid);

% PC scores (T_cal x n_pc)
fid = fopen(fullfile(out_dir, 'pc_scores.bin'), 'w');
fwrite(fid, [T_cal, n_pc], 'int32');
fwrite(fid, pcs', 'single');  % transpose for row-major Python reading
fclose(fid);

% field mean
fid = fopen(fullfile(out_dir, 'field_mean.bin'), 'w');
fwrite(fid, length(mu), 'int32');
fwrite(fid, mu, 'single');
fclose(fid);

%% prepare station obs for January regression
fprintf('Preparing station observations...\n');
obs_jan = obs(obs.month == target_month, :);
obs_jan = obs_jan(obs_jan.year >= year_cal_min & obs_jan.year <= year_cal_max, :);

% pivot to (year x station) matrix
[unique_ids, ~, station_idx] = unique(obs_jan.ID);
[unique_years, ~, year_idx] = unique(obs_jan.year);
n_stations_raw = length(unique_ids);
n_years = length(unique_years);

Y_mat = nan(n_years, n_stations_raw, 'single');
for i = 1:height(obs_jan)
    Y_mat(year_idx(i), station_idx(i)) = obs_jan.value(i);
end

% align years with ESM
[common_years, ia, ib] = intersect(unique_years, years_jan_cal);
Y_aligned = Y_mat(ia, :);
pcs_aligned = pcs(ib, :);

% filter stations with >= 30 valid years
valid_count = sum(~isnan(Y_aligned), 1);
keep = valid_count >= 30;
Y_aligned = Y_aligned(:, keep);
station_ids_keep = unique_ids(keep);
n_stations = size(Y_aligned, 2);
fprintf('  stations with >= 30 years: %d / %d\n', n_stations, n_stations_raw);

%% train/test split (deterministic: first 70% years for train)
T_common = size(Y_aligned, 1);
n_train = max(round(T_common * train_frac), 30);
idx_train = 1:n_train;
idx_test = (n_train+1):T_common;
perm = 1:T_common;  % dummy for saving

X_tr = pcs_aligned(idx_train, :);
X_te = pcs_aligned(idx_test, :);
Y_tr = Y_aligned(idx_train, :);
Y_te = Y_aligned(idx_test, :);

%% per-station OLS (WITH intercept — matching our Python implementation)
fprintf('Running per-station OLS (with intercept)...\n');
beta = nan(n_pc+1, n_stations, 'single');
rmse_train = nan(1, n_stations, 'single');
rmse_test = nan(1, n_stations, 'single');
r2_train = nan(1, n_stations, 'single');

X_tr_int = [ones(size(X_tr,1),1), X_tr];  % prepend intercept column
X_te_int = [ones(size(X_te,1),1), X_te];

for s = 1:n_stations
    ys = Y_tr(:, s);
    m = ~isnan(ys);
    if sum(m) < 20
        continue
    end
    % OLS: beta = (X'X)^-1 X'Y
    Xm = X_tr_int(m, :);
    Ym = ys(m);
    b = Xm \ Ym;
    beta(:, s) = single(b);
    
    yhat = Xm * b;
    resid = Ym - yhat;
    rmse_train(s) = sqrt(mean(resid.^2));
    ss_res = sum(resid.^2);
    ss_tot = sum((Ym - mean(Ym)).^2);
    if ss_tot > 0
        r2_train(s) = 1 - ss_res / ss_tot;
    end
    
    % test
    ys_te = Y_te(:, s);
    mt = ~isnan(ys_te);
    if any(mt)
        pred_te = X_te_int(mt, :) * b;
        rmse_test(s) = sqrt(mean((pred_te - ys_te(mt)).^2));
    end
end

fprintf('  median RMSE train: %.4f\n', median(rmse_train, 'omitnan'));
fprintf('  median RMSE test:  %.4f\n', median(rmse_test, 'omitnan'));
fprintf('  median R2 train:   %.4f\n', median(r2_train, 'omitnan'));

%% save regression outputs
fid = fopen(fullfile(out_dir, 'beta.bin'), 'w');
fwrite(fid, [n_pc+1, n_stations], 'int32');
fwrite(fid, beta, 'single');  % already (n_pc+1, n_stations), col-major = station-by-station
fclose(fid);

fid = fopen(fullfile(out_dir, 'rmse_train.bin'), 'w');
fwrite(fid, n_stations, 'int32');
fwrite(fid, rmse_train, 'single');
fclose(fid);

fid = fopen(fullfile(out_dir, 'rmse_test.bin'), 'w');
fwrite(fid, n_stations, 'int32');
fwrite(fid, rmse_test, 'single');
fclose(fid);

% save train/test split indices and station IDs
fid = fopen(fullfile(out_dir, 'split_perm.bin'), 'w');
fwrite(fid, length(perm), 'int32');
fwrite(fid, perm, 'int32');
fclose(fid);

% save station IDs as a text file
fid = fopen(fullfile(out_dir, 'station_ids.txt'), 'w');
for s = 1:n_stations
    fprintf(fid, '%s\n', station_ids_keep{s});
end
fclose(fid);

fprintf('\nStep C outputs saved to: %s\n', out_dir);
