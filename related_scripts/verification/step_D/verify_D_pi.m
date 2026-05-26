%% Step D PI verification: PI from regression MSE (January, tas)
%  Matches Guaita's ds_ESM_mat_v1.m PI method:
%    sigma_hat = sqrt(MSE); PI = quantile(Yhat + sigma*randn(T,1000), [0.025 0.975])
%  For tas (Gaussian), this is equivalent to Yhat ± 1.96*sigma.
%
%  Saves: sigma2_hat, PI_lo (analytical), PI_hi (analytical) for comparison.

clear; close all;

%% paths and parameters
trace_tas = 'D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc';
out_dir = 'D:\Dataset\DPastCliM-NA\verification\step_D\matlab';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

year_cal_min = 1875; year_cal_max = 1999;
n_pc = 5; train_frac = 0.7; target_month = 1;

%% load ESM field (same as verify_D_prediction.m)
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

lon_mask = lon_sorted >= -185 & lon_sorted <= -45;
lat_mask = lat_sorted >= -90 & lat_sorted <= 90;
tas_na = tas_map(lon_mask, lat_mask, :);

cal_mask = (year_arr >= year_cal_min) & (year_arr <= year_cal_max) & (month_arr == target_month);
esm_jan_cal = tas_na(:,:,cal_mask);
years_jan_cal = year_arr(cal_mask);
T_cal = size(esm_jan_cal, 3);
field_2d = reshape(permute(esm_jan_cal, [3, 2, 1]), T_cal, []);

%% PCA
mu = mean(field_2d, 1);
Xc = field_2d - mu;
[U, S, V] = svd(Xc, 'econ');
scores_full = U * S;
pcs = scores_full(:, 1:n_pc);

%% load station obs
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

T_common = size(Y_aligned, 1);
n_train = max(round(T_common * train_frac), 30);

X_full = [ones(T_common, 1), pcs_aligned];
X_tr = X_full(1:n_train, :);
Y_tr = Y_aligned(1:n_train, :);

%% per-station OLS + sigma2_hat + PI
fprintf('Computing prediction + PI for %d stations...\n', n_stations);
Yhat_full = nan(T_common, n_stations, 'single');
sigma2_hat = nan(1, n_stations, 'single');
PI_lo = nan(T_common, n_stations, 'single');
PI_hi = nan(T_common, n_stations, 'single');

for s = 1:n_stations
    ys = Y_tr(:, s);
    m = ~isnan(ys);
    if sum(m) < 20, continue; end
    b = X_tr(m, :) \ ys(m);
    yhat = single(X_full * b);
    Yhat_full(:, s) = yhat;

    res = ys(m) - single(X_tr(m,:) * b);
    sigma2_hat(s) = mean(res.^2);
    sigma = sqrt(sigma2_hat(s));

    PI_lo(:, s) = yhat - 1.96 * sigma;
    PI_hi(:, s) = yhat + 1.96 * sigma;
end

fprintf('  Done: %d stations\n', n_stations);

%% save
fid = fopen(fullfile(out_dir, 'sigma2_hat.bin'), 'w');
fwrite(fid, n_stations, 'int32');
fwrite(fid, sigma2_hat, 'single');
fclose(fid);

fid = fopen(fullfile(out_dir, 'pi_lo.bin'), 'w');
fwrite(fid, [T_common, n_stations], 'int32');
fwrite(fid, PI_lo', 'single');
fclose(fid);

fid = fopen(fullfile(out_dir, 'pi_hi.bin'), 'w');
fwrite(fid, [T_common, n_stations], 'int32');
fwrite(fid, PI_hi', 'single');
fclose(fid);

fprintf('PI outputs saved to: %s\n', out_dir);
fprintf('  sigma2_hat range: [%.6f, %.6f]\n', min(sigma2_hat(~isnan(sigma2_hat))), max(sigma2_hat(~isnan(sigma2_hat))));
