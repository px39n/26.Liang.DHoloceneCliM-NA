%% Step B verification: Load TraCE-21k II, convert units, interpolate to stations
%  Output: D:\Dataset\DPastCliM-NA\verification\step_B\matlab\
%
%  Adapts Guaita's process_ESM_Data_v1.m logic to TraCE-21k II data.
%  Uses 'nearest' interpolation (matching Guaita's PCR_calibration_v5.m L185).
%
%  Scope: calibration period only (1875-1999) — same window as Python pipeline.

clear; close all;

%% paths
trace_tas = 'D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc';
trace_pr  = 'D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.PRECT.nc';
out_dir   = 'D:\Dataset\DPastCliM-NA\verification\step_B\matlab';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

% station metadata from Step A (Python version — canonical)
meta = parquetread('D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_meta.parquet');
sta_lat = meta.lat;
sta_lon = meta.lon;
sta_id  = string(meta.ID);

%% parameters
year_cal_min = 1875;
year_cal_max = 1999;

%% --- TAS ---
fprintf('Loading TraCE TREFHT...\n');
trefht = ncread(trace_tas, 'TREFHT');  % MATLAB ncread reverses NetCDF dim order
% NetCDF is (time, lat, lon) -> MATLAB gives (lon, lat, time)
fprintf('  ncread size: [%s]\n', num2str(size(trefht)));
lat = double(ncread(trace_tas, 'lat'));
lon = double(ncread(trace_tas, 'lon'));
time_ka = double(ncread(trace_tas, 'time'));  % ka BP
fprintf('  lat: %d values, [%.2f .. %.2f]\n', length(lat), lat(1), lat(end));
fprintf('  lon: %d values, [%.2f .. %.2f]\n', length(lon), lon(1), lon(end));

% convert time to calendar year (CE)
cal_year_frac = 1950 + time_ka * 1000;
% approximate integer year and month
months_since_0 = round(cal_year_frac * 12);
year_arr = floor(months_since_0 / 12);
month_arr = mod(months_since_0, 12) + 1;

% convert to degC
tas_map = trefht - 273.15;  % (lon, lat, time)
clear trefht

% convert lon from 0-360 to -180..180
lon180 = mod(lon + 180, 360) - 180;
[lon_sorted, lon_idx] = sort(lon180);
tas_map = tas_map(lon_idx, :, :);

% also sort lat to ascending (required by griddedInterpolant)
[lat_sorted, lat_idx] = sort(lat);
tas_map = tas_map(:, lat_idx, :);
lat = lat_sorted;

% diagnostic: value at Bermuda grid cell
lat_i = find(abs(lat - 32) == min(abs(lat - 32)), 1);
lon_i = find(abs(lon_sorted - (-64.68)) == min(abs(lon_sorted - (-64.68))), 1);
fprintf('  Bermuda check: lat=%.2f, lon=%.2f, val(t=1)=%.2f C\n', lat(lat_i), lon_sorted(lon_i), tas_map(lon_i, lat_i, 1));

% filter calibration period
cal_mask = (year_arr >= year_cal_min) & (year_arr <= year_cal_max);
fprintf('  cal period: %d months\n', sum(cal_mask));
tas_cal = tas_map(:, :, cal_mask);
year_cal = year_arr(cal_mask);
month_cal = month_arr(cal_mask);

% interpolate to station coords using nearest (matching Guaita)
n_times = size(tas_cal, 3);
n_stations = length(sta_lat);
esm_at_station = nan(n_times, n_stations, 'single');

[lon_grid, lat_grid] = ndgrid(lon_sorted, lat);

fprintf('  interpolating %d times x %d stations (nearest)...\n', n_times, n_stations);
for t = 1:n_times
    f = griddedInterpolant(lon_grid, lat_grid, double(tas_cal(:,:,t)), 'nearest', 'nearest');
    esm_at_station(t,:) = single(f(sta_lon, sta_lat));
end

% save
fprintf('  saving...\n');
T_out = table(year_cal, month_cal, 'VariableNames', {'year','month'});
writetable(T_out, fullfile(out_dir, 'time_cal.csv'));

% ESM at stations: save as parquet (station columns, time rows)
% For large data, save as a binary file
fid = fopen(fullfile(out_dir, 'esm_at_station_tas.bin'), 'w');
fwrite(fid, [n_times, n_stations], 'int32');
fwrite(fid, esm_at_station', 'single');  % transpose: MATLAB col-major -> row-major for Python
fclose(fid);

% also save the full grid for one sample month (Jan of first cal year)
% to compare spatial patterns
idx_sample = find(year_cal == year_cal_min & month_cal == 1, 1, 'first');
if ~isempty(idx_sample)
    sample_field = tas_cal(:,:,idx_sample);  % (lon, lat)
    save(fullfile(out_dir, 'sample_field_tas.mat'), 'sample_field', 'lon_sorted', 'lat', '-v7.3');
end

% ESM cal-period mean per station per month
esm_cal_mean = nan(12, n_stations, 'single');
for m = 1:12
    mi = month_cal == m;
    esm_cal_mean(m,:) = mean(esm_at_station(mi,:), 1, 'omitnan');
end
fid = fopen(fullfile(out_dir, 'esm_cal_mean_tas.bin'), 'w');
fwrite(fid, [12, n_stations], 'int32');
fwrite(fid, esm_cal_mean', 'single');  % transpose
fclose(fid);

fprintf('Step B tas done.\n');

%% --- PR ---
fprintf('Loading TraCE PRECT...\n');
prect = ncread(trace_pr, 'PRECT');  % m/s
pr_map = prect * 86400 * 1000;  % mm/day
clear prect
pr_map = pr_map(lon_idx, :, :);  % reorder lon
pr_map = pr_map(:, lat_idx, :);  % reorder lat (same as tas)

pr_cal = pr_map(:, :, cal_mask);

esm_at_station_pr = nan(n_times, n_stations, 'single');
fprintf('  interpolating pr...\n');
for t = 1:n_times
    f = griddedInterpolant(lon_grid, lat_grid, double(pr_cal(:,:,t)), 'nearest', 'nearest');
    esm_at_station_pr(t,:) = single(max(0, f(sta_lon, sta_lat)));
end

fid = fopen(fullfile(out_dir, 'esm_at_station_pr.bin'), 'w');
fwrite(fid, [n_times, n_stations], 'int32');
fwrite(fid, esm_at_station_pr', 'single');  % transpose
fclose(fid);

esm_cal_mean_pr = nan(12, n_stations, 'single');
for m = 1:12
    mi = month_cal == m;
    esm_cal_mean_pr(m,:) = mean(esm_at_station_pr(mi,:), 1, 'omitnan');
end
fid = fopen(fullfile(out_dir, 'esm_cal_mean_pr.bin'), 'w');
fwrite(fid, [12, n_stations], 'int32');
fwrite(fid, esm_cal_mean_pr', 'single');  % transpose
fclose(fid);

fprintf('Step B pr done.\n');
fprintf('All Step B outputs in: %s\n', out_dir);
