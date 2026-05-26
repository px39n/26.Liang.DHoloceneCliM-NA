% verify_delta_change.m
% Verify delta-change (30-year moving-mean) computation matches Python.
% Uses synthetic data to ensure deterministic comparison.

out_dir = 'D:\Dataset\DPastCliM-NA\verification\step_D';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

rng(42);

ny = 10; nx = 12;
T_cal = 75;     % calibration timesteps
T_full = 200;   % full transient timesteps (includes cal at the end)

% Synthetic ESM field: smooth spatiotemporal signal + noise
lat = linspace(20, 60, ny)';
lon = linspace(-140, -60, nx)';
[LON, LAT] = meshgrid(lon, lat);

field_full = zeros(ny, nx, T_full);
for t = 1:T_full
    trend = 2 * sin(2*pi*t/T_full) * ones(ny, nx);
    spatial = LAT/30 + LON/100;
    field_full(:,:,t) = trend + spatial + 0.1*randn(ny, nx);
end

% Cal period = last T_cal timesteps (matching typical setup)
i_cal = (T_full - T_cal + 1):T_full;
field_cal = field_full(:,:,i_cal);

% Reshape to (ny*nx, T) for movmean (MATLAB operates along dim 2)
M_full = reshape(field_full, ny*nx, T_full);
M_cal  = reshape(field_cal, ny*nx, T_cal);

n_mov = 30;

% ====== TAS delta-change ======
mu_cal = mean(M_cal, 2, 'omitnan');
mu_mov = movmean(M_full, n_mov, 2, 'omitnan');
mu_adj_tas = mu_mov - mu_cal;

% ====== PR delta-change ======
field_pr = abs(field_full) + 0.5;  % ensure positive for log
M_full_pr = reshape(field_pr, ny*nx, T_full);
field_pr_cal = field_pr(:,:,i_cal);
M_cal_pr = reshape(field_pr_cal, ny*nx, T_cal);

M_t = 1 + min(M_cal_pr, [], 2);
mu_cal_pr = mean(log(M_cal_pr + M_t), 2, 'omitnan');
mu_mov_pr = movmean(log(M_full_pr + M_t), n_mov, 2, 'omitnan');
mu_adj_pr = mu_mov_pr - mu_cal_pr;

% ====== Nearest-neighbor interpolation ======
n_stations = 15;
sta_lat = 25 + (55-25)*rand(n_stations, 1);
sta_lon = -135 + (75)*rand(n_stations, 1);

% MATLAB griddedInterpolant nearest
mu_adj_tas_3d = reshape(mu_adj_tas, ny, nx, T_full);
mu_adj_pr_3d  = reshape(mu_adj_pr, ny, nx, T_full);

mu_adj_at_sta_tas = zeros(n_stations, T_full);
mu_adj_at_sta_pr  = zeros(n_stations, T_full);
[LON_nd, LAT_nd] = ndgrid(lon, lat);  % griddedInterpolant wants ndgrid

for t = 1:T_full
    f_tas = griddedInterpolant(LON_nd, LAT_nd, mu_adj_tas_3d(:,:,t)', 'nearest', 'nearest');
    f_pr  = griddedInterpolant(LON_nd, LAT_nd, mu_adj_pr_3d(:,:,t)', 'nearest', 'nearest');
    for s = 1:n_stations
        mu_adj_at_sta_tas(s, t) = f_tas(sta_lon(s), sta_lat(s));
        mu_adj_at_sta_pr(s, t)  = f_pr(sta_lon(s), sta_lat(s));
    end
end

% ====== Save ======
save(fullfile(out_dir, 'dc_matlab.mat'), ...
    'field_full', 'field_pr', 'lat', 'lon', ...
    'sta_lat', 'sta_lon', 'T_cal', 'T_full', 'n_mov', ...
    'mu_adj_tas', 'mu_adj_pr', ...
    'mu_adj_at_sta_tas', 'mu_adj_at_sta_pr', ...
    '-v7.3');

fprintf('Saved to %s\n', fullfile(out_dir, 'dc_matlab.mat'));
fprintf('mu_adj_tas range: [%.6f, %.6f]\n', min(mu_adj_tas(:)), max(mu_adj_tas(:)));
fprintf('mu_adj_pr range: [%.6f, %.6f]\n', min(mu_adj_pr(:)), max(mu_adj_pr(:)));
fprintf('mu_adj_at_sta_tas range: [%.6f, %.6f]\n', min(mu_adj_at_sta_tas(:)), max(mu_adj_at_sta_tas(:)));
fprintf('Done.\n');
