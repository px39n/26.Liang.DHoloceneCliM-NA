%% Step D verification: PCR prediction on calibration period (January, tas)
%  Uses Step C calibrated models (PCA + OLS) to predict station values.
%  Compares MATLAB vs Python prediction outputs.
%
%  Logic: project ESM onto trained EOFs -> apply per-station beta -> predicted values.

clear; close all;

%% paths and parameters
trace_tas = 'D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc';
out_dir = 'D:\Dataset\DPastCliM-NA\verification\step_D\matlab';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

year_cal_min = 1875; year_cal_max = 1999;
n_pc = 5; train_frac = 0.7; target_month = 1;

%% load ESM field (same as Step C)
fprintf('Loading ESM field...\n');
trefht = ncread(trace_tas, 'TREFHT');
lat = double(ncread(trace_tas, 'lat'));
lon = double(ncread(trace_tas, 'lon'));
time_ka = double(ncread(trace_tas, 'time'));

cal_year_frac = 1950 + time_ka * 1000;
months_since_0 = round(cal_year_frac * 12);
year_arr = int32(floor(months_since_0 / 12));
month_arr = int32(mod(months_since_0, 12) + 1);

tas_map = trefht - 273.15; clear trefht;
lon180 = mod(lon + 180, 360) - 180;
[lon_sorted, lon_idx] = sort(lon180);
tas_map = tas_map(lon_idx, :, :);
[lat_sorted, lat_idx] = sort(lat);
tas_map = tas_map(:, lat_idx, :);

% NA window (matching Python)
lon_mask = lon_sorted >= -185 & lon_sorted <= -45;
lat_mask = lat_sorted >= -90 & lat_sorted <= 90;
esm_lon = lon_sorted(lon_mask);
esm_lat = lat_sorted(lat_mask);
tas_na = tas_map(lon_mask, lat_mask, :);

% cal period January
cal_mask = (year_arr >= year_cal_min) & (year_arr <= year_cal_max) & (month_arr == target_month);
esm_jan_cal = tas_na(:,:,cal_mask);
years_jan_cal = year_arr(cal_mask);
T_cal = size(esm_jan_cal, 3);

% flatten (matching Step C MATLAB: permute to (time,lat,lon) then reshape)
field_2d = reshape(permute(esm_jan_cal, [3, 2, 1]), T_cal, []);

%% PCA (reproduce Step C exactly)
mu = mean(field_2d, 1);
Xc = field_2d - mu;
[U, S, V] = svd(Xc, 'econ');
scores_full = U * S;
pcs = scores_full(:, 1:n_pc);

%% load station obs and reproduce Step C split
obs = parquetread('D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_obs.parquet');
obs_jan = obs(obs.month == target_month, :);
obs_jan = obs_jan(obs_jan.year >= year_cal_min & obs_jan.year <= year_cal_max, :);

[unique_ids, ~, station_idx] = unique(obs_jan.ID);
[unique_years, ~, year_idx] = unique(obs_jan.year);
n_stations_raw = length(unique_ids);

Y_mat = nan(length(unique_years), n_stations_raw, 'single');
for i = 1:height(obs_jan)
    Y_mat(year_idx(i), station_idx(i)) = obs_jan.value(i);
end

[common_years, ia, ib] = intersect(unique_years, years_jan_cal);
Y_aligned = Y_mat(ia, :);
pcs_aligned = pcs(ib, :);
valid_count = sum(~isnan(Y_aligned), 1);
keep = valid_count >= 30;
Y_aligned = Y_aligned(:, keep);
station_ids = unique_ids(keep);
n_stations = size(Y_aligned, 2);

% deterministic split
T_common = size(Y_aligned, 1);
n_train = max(round(T_common * train_frac), 30);

X_full = [ones(T_common, 1), pcs_aligned];
X_tr = X_full(1:n_train, :);
Y_tr = Y_aligned(1:n_train, :);

%% per-station OLS and full-period prediction
fprintf('Predicting on full cal period...\n');
Yhat_full = nan(T_common, n_stations, 'single');

for s = 1:n_stations
    ys = Y_tr(:, s);
    m = ~isnan(ys);
    if sum(m) < 20, continue; end
    b = X_tr(m, :) \ ys(m);
    Yhat_full(:, s) = single(X_full * b);
end

fprintf('  predicted %d stations x %d years\n', n_stations, T_common);

%% save predictions
fid = fopen(fullfile(out_dir, 'yhat_full.bin'), 'w');
fwrite(fid, [T_common, n_stations], 'int32');
fwrite(fid, Yhat_full', 'single');  % transpose for row-major
fclose(fid);

% also save station IDs
fid = fopen(fullfile(out_dir, 'station_ids.txt'), 'w');
for s = 1:n_stations
    fprintf(fid, '%s\n', station_ids{s});
end
fclose(fid);

fprintf('Step D outputs saved to: %s\n', out_dir);
