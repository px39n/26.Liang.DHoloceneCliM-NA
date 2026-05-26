% Minimal MATLAB driver that reproduces the GHCN tas parsing portion of
% 25.Guaita.DPastCliM-NA/preprocessing/GHCNm.m up to (but not including)
% the GPR gridding step.  Output is saved to .mat for Python comparison.
%
% Hardcoded to match the Python defaults in
%   related_scripts/prepare_ghcn.py
% so that diffs are meaningful.

clear

path_inv  = 'D:\Dataset\DPastCliM-NA\GHCN\ghcnm.v4.0.1.20260512\ghcnm.tavg.v4.0.1.20260512.qcf.inv';
path_dat  = 'D:\Dataset\DPastCliM-NA\GHCN\ghcnm.v4.0.1.20260512\ghcnm.tavg.v4.0.1.20260512.qcf.dat';
path_out  = 'D:\Dataset\DPastCliM-NA\GHCN\interim\matlab_ghcn_tas.mat';

lim_lat  = [7 75];
lim_lon  = [-180 -50];
lim_year = [1850 2014];
min_years = 20;

%% inv
fid = fopen(path_inv,'r');
formatSpec = '%11s %f %f %f %s';
inv_file = textscan(fid, formatSpec, 'Delimiter','', 'Whitespace','', 'MultipleDelimsAsOne', true);
fclose(fid);

metaTable = table(inv_file{1}, inv_file{2}, inv_file{3}, inv_file{4}, inv_file{5}, ...
                  'VariableNames', {'ID','lat','lon','elev','location'});
clear inv_file

flag_domain = lim_lat(1) <= metaTable.lat & metaTable.lat <= lim_lat(2) & ...
              lim_lon(1) <= metaTable.lon & metaTable.lon <= lim_lon(2);
metaTable = metaTable(flag_domain,:);

%% dat
fid = fopen(path_dat,'r');
formatSpec = '%11s %4d %4s';
for i = 1:12
    formatSpec = [formatSpec ' %5f %*3s'];
end
obs_file = textscan(fid, formatSpec, 'Delimiter','', 'Whitespace','', 'MultipleDelimsAsOne', true);
fclose(fid);

obsTable_tmp = table(obs_file{1}, obs_file{2}, ...
    obs_file{4}/100,  obs_file{5}/100,  obs_file{6}/100,  obs_file{7}/100, ...
    obs_file{8}/100,  obs_file{9}/100,  obs_file{10}/100, obs_file{11}/100, ...
    obs_file{12}/100, obs_file{13}/100, obs_file{14}/100, obs_file{15}/100, ...
    'VariableNames', {'ID','Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'});

flag_station = ismember(string(obsTable_tmp.ID), string(metaTable.ID));
obsTable_tmp = obsTable_tmp(flag_station,:);
flag_station = ismember(string(metaTable.ID), string(obsTable_tmp.ID));
metaTable = metaTable(flag_station,:);
clear obs_file

%% time filter
flag_year = lim_year(1) <= obsTable_tmp.Year & obsTable_tmp.Year <= lim_year(2);
obsTable_tmp = obsTable_tmp(flag_year,:);

%% reshape wide -> long with month_since_0CE
obsTable = table('Size',[0 3], ...
    'VariableTypes',{'string','int64','single'}, ...
    'VariableNames',{'ID','month_since_0CE','Value'});
for i_mth = 1:12
    tmp_table = obsTable_tmp(:, [1 2 2+i_mth]);
    tmp_table.Year = tmp_table.Year * 12 + i_mth;
    tmp_table.Properties.VariableNames{'Year'} = 'month_since_0CE';
    tmp_table.Properties.VariableNames{3} = 'Value';
    if i_mth == 1
        obsTable = tmp_table;
    else
        obsTable = vertcat(obsTable, tmp_table);
    end
end
flag_error = obsTable.Value == -99.99;
obsTable = obsTable(~flag_error,:);

%% attach lat/lon/elev
[~, IDloc] = ismember(categorical(obsTable.ID), categorical(metaTable.ID));
obsTable.lat  = metaTable.lat(IDloc);
obsTable.lon  = metaTable.lon(IDloc);
obsTable.elev = metaTable.elev(IDloc);

%% min-record filter (vectorised: groupcounts on string IDs)
% Caz's version is O(n_stations^2) due to ismember inside a loop -- replaced
% by a single grouped count which gives the same result.
ids_obs = string(obsTable.ID);
[uniq_ids, ~, idx] = unique(ids_obs);
counts = accumarray(idx, 1);
keep_ids_set = uniq_ids(counts >= 12 * min_years);
keep_obs_mask = ismember(ids_obs, keep_ids_set);
obsTable  = obsTable(keep_obs_mask, :);
metaTable = metaTable(ismember(string(metaTable.ID), keep_ids_set), :);

%% derive year/month for easier comparison with Python
obsTable.year  = floor((obsTable.month_since_0CE - 1) / 12);
obsTable.month = mod(obsTable.month_since_0CE - 1, 12) + 1;

fprintf('matlab obs: %d rows, %d stations\n', height(obsTable), height(metaTable));
fprintf('value [degC] min %.3f max %.3f mean %.3f\n', ...
    min(obsTable.Value), max(obsTable.Value), mean(obsTable.Value));

if ~exist(fileparts(path_out),'dir'); mkdir(fileparts(path_out)); end
save(path_out, 'obsTable', 'metaTable', 'lim_lat', 'lim_lon', 'lim_year', 'min_years', '-v7.3');
fprintf('saved %s\n', path_out);

% Also write as Parquet (table-friendly format Python reads natively).
out_pq_obs  = strrep(path_out, '.mat', '_obs.parquet');
out_pq_meta = strrep(path_out, '.mat', '_meta.parquet');
% parquetwrite needs typed columns; convert ID strings explicitly.
obsTable.ID  = string(obsTable.ID);
metaTable.ID = string(metaTable.ID);
parquetwrite(out_pq_obs,  obsTable);
parquetwrite(out_pq_meta, metaTable);
fprintf('wrote %s and %s\n', out_pq_obs, out_pq_meta);
