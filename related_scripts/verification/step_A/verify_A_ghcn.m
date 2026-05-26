%% Step A verification: GHCN tas fixed-width parser (corrected)
%
%  Replaces Guaita's textscan with fixed byte-position parsing
%  to match Python's ghcn.py exactly.
%
%  Output: D:\Dataset\DPastCliM-NA\verification\step_A\matlab\

clear; close all;

%% paths
dat_file = 'D:\Dataset\DPastCliM-NA\GHCN\ghcnm.v4.0.1.20260512\ghcnm.tavg.v4.0.1.20260512.qcf.dat';
inv_file = 'D:\Dataset\DPastCliM-NA\GHCN\ghcnm.v4.0.1.20260512\ghcnm.tavg.v4.0.1.20260512.qcf.inv';
out_dir  = 'D:\Dataset\DPastCliM-NA\verification\step_A\matlab';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% parse .inv (metadata)
fprintf('Parsing .inv...\n');
fid = fopen(inv_file, 'r');
lines = {};
while ~feof(fid)
    lines{end+1,1} = fgetl(fid);
end
fclose(fid);
n_meta = length(lines);

meta_ID   = cell(n_meta, 1);
meta_lat  = nan(n_meta, 1);
meta_lon  = nan(n_meta, 1);
meta_elev = nan(n_meta, 1);
meta_name = cell(n_meta, 1);

for i = 1:n_meta
    L = lines{i};
    if length(L) < 38, continue; end
    meta_ID{i}   = L(1:11);
    meta_lat(i)  = str2double(L(13:20));
    meta_lon(i)  = str2double(L(22:30));
    meta_elev(i) = str2double(L(32:38));
    if length(L) >= 39
        meta_name{i} = strtrim(L(39:min(end,100)));
    else
        meta_name{i} = '';
    end
end
fprintf('  %d stations in .inv\n', n_meta);

%% parse .dat (fixed-width, matching Python byte positions)
fprintf('Parsing .dat (fixed-width)...\n');
fid = fopen(dat_file, 'r');
lines_dat = {};
while ~feof(fid)
    lines_dat{end+1,1} = fgetl(fid);
end
fclose(fid);
n_rows = length(lines_dat);
fprintf('  %d rows\n', n_rows);

% preallocate
row_ID   = cell(n_rows, 1);
row_year = zeros(n_rows, 1, 'int32');
row_vals = nan(n_rows, 12, 'single');  % 12 monthly values in degC

MISSING = -9999;

for i = 1:n_rows
    L = lines_dat{i};
    row_ID{i}   = L(1:11);
    row_year(i) = int32(str2double(L(12:15)));
    for m = 0:11
        % byte positions: col 20..24 for month 1, then +8 for each subsequent
        % Python: start = 19 + m*8, end = start+5 (0-indexed)
        % MATLAB: start = 20 + m*8, end = start+4 (1-indexed)
        s = 20 + m*8;
        e = s + 4;
        if e > length(L)
            row_vals(i, m+1) = NaN;
            continue;
        end
        v_str = strtrim(L(s:e));
        if isempty(v_str)
            row_vals(i, m+1) = NaN;
        else
            v = str2double(v_str);
            if v == MISSING || isnan(v)
                row_vals(i, m+1) = NaN;
            else
                row_vals(i, m+1) = single(v / 100);  % 0.01 degC -> degC
            end
        end
    end
end

fprintf('  parsed %d rows\n', n_rows);

%% convert to long format
% expand (n_rows, 12) -> (n_rows*12, 1) with month column
total = n_rows * 12;
long_ID    = cell(total, 1);
long_year  = zeros(total, 1, 'int32');
long_month = zeros(total, 1, 'int32');
long_value = nan(total, 1, 'single');

idx = 0;
for i = 1:n_rows
    for m = 1:12
        idx = idx + 1;
        long_ID{idx}    = row_ID{i};
        long_year(idx)  = row_year(i);
        long_month(idx) = int32(m);
        long_value(idx) = row_vals(i, m);
    end
end

% remove NaN rows (missing values)
valid = ~isnan(long_value);
long_ID    = long_ID(valid);
long_year  = long_year(valid);
long_month = long_month(valid);
long_value = long_value(valid);
fprintf('  %d valid observations (after removing NaN)\n', length(long_value));

%% save as parquet
obsTable = table(string(long_ID), long_year, long_month, long_value, ...
    'VariableNames', {'ID','year','month','value'});
parquetwrite(fullfile(out_dir, 'obs.parquet'), obsTable);

metaTable = table(string(meta_ID), meta_lat, meta_lon, meta_elev, string(meta_name), ...
    'VariableNames', {'ID','lat','lon','elev','name'});
parquetwrite(fullfile(out_dir, 'meta.parquet'), metaTable);

fprintf('Step A outputs saved to: %s\n', out_dir);
fprintf('  obs: %d rows, meta: %d stations\n', height(obsTable), height(metaTable));
