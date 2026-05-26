%% Step E verification: station-to-grid interpolation (January, tas)
%  Uses Guaita's exact approach: Albers projection + natural neighbor interpolation.
%  Grid: NE-US 0.5 deg test region.

clear; close all;

%% paths
out_dir = 'D:\Dataset\DPastCliM-NA\verification\step_E\matlab';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% load Step D predictions from MATLAB
ml_dir = 'D:\Dataset\DPastCliM-NA\verification\step_D\matlab';
fid = fopen(fullfile(ml_dir, 'yhat_full.bin'), 'r');
hdr = fread(fid, 2, 'int32');
T = hdr(1); n_st = hdr(2);
yhat = fread(fid, [n_st, T], 'single')';
fclose(fid);

ml_ids_file = fullfile(ml_dir, 'station_ids.txt');
fid = fopen(ml_ids_file, 'r');
station_ids = textscan(fid, '%s');
station_ids = station_ids{1};
fclose(fid);

%% load station metadata
obs_meta = parquetread('D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_meta.parquet');
id_list = string(obs_meta.ID);
lat_list = obs_meta.lat;
lon_list = obs_meta.lon;

[~, ia, ib] = intersect(string(station_ids), id_list);
st_lat = lat_list(ib);
st_lon = lon_list(ib);
yhat_matched = yhat(:, ia);
fprintf('Matched %d / %d stations\n', length(ia), n_st);

%% define target grid (NE-US, 0.5 deg)
lat_min = 35; lat_max = 50;
lon_min = -90; lon_max = -65;
res = 0.5;
grid_lat = (lat_min:res:lat_max)';
grid_lon = (lon_min:res:lon_max)';
[glon, glat] = meshgrid(grid_lon, grid_lat);
ny = length(grid_lat); nx = length(grid_lon);

%% filter stations in region (+5 deg buffer)
in_region = st_lat >= lat_min-5 & st_lat <= lat_max+5 & ...
            st_lon >= lon_min-5 & st_lon <= lon_max+5;
fprintf('Stations in buffered region: %d\n', sum(in_region));
st_lat_r = st_lat(in_region);
st_lon_r = st_lon(in_region);
yhat_r = yhat_matched(:, in_region);

%% Project to Albers (NAD83 / Conus Albers, EPSG:5070) — manual implementation
% EPSG:5070 parameters: GRS80 ellipsoid, std parallels 29.5°N and 45.5°N,
% origin 23°N 96°W, false easting 0, false northing 0
[x_proj, y_proj] = albers_forward(st_lon_r, st_lat_r);
[xgrid_proj, ygrid_proj] = albers_forward(glon(:), glat(:));

%% Gridding: natural neighbor (Guaita's method)
gridded = nan(T, ny, nx, 'single');

for t = 1:T
    z = yhat_r(t, :)';
    valid = ~isnan(z);
    if sum(valid) < 4, continue; end
    F = scatteredInterpolant(x_proj(valid), y_proj(valid), z(valid), 'natural', 'nearest');
    vals = F(xgrid_proj, ygrid_proj);
    gridded(t, :, :) = single(reshape(vals, ny, nx));
end

fprintf('Gridded %d timesteps to %dx%d grid\n', T, ny, nx);

%% save — write time-slice by time-slice (row-major for Python)
fid = fopen(fullfile(out_dir, 'gridded.bin'), 'w');
fwrite(fid, [T, ny, nx], 'int32');
for t = 1:T
    slice = squeeze(gridded(t,:,:));  % (ny, nx)
    fwrite(fid, slice', 'single');    % transpose: nx-varies-first in col-major
end
fclose(fid);

% also save projected coordinates for debugging
fid = fopen(fullfile(out_dir, 'proj_coords.bin'), 'w');
fwrite(fid, length(x_proj), 'int32');
fwrite(fid, x_proj, 'double');
fwrite(fid, y_proj, 'double');
fwrite(fid, length(xgrid_proj), 'int32');
fwrite(fid, xgrid_proj, 'double');
fwrite(fid, ygrid_proj, 'double');
fclose(fid);

fprintf('Step E outputs saved to: %s\n', out_dir);

function [x, y] = albers_forward(lon_deg, lat_deg)
    % Albers Equal-Area Conic, EPSG:5070 (NAD83 / Conus Albers)
    % GRS80 ellipsoid
    a = 6378137.0;
    f = 1/298.257222101;
    e2 = 2*f - f^2;
    e = sqrt(e2);
    
    phi1 = 29.5 * pi/180;
    phi2 = 45.5 * pi/180;
    phi0 = 23.0 * pi/180;
    lam0 = -96.0 * pi/180;
    x0 = 0; y0 = 0;
    
    m1 = cos(phi1) / sqrt(1 - e2*sin(phi1)^2);
    m2 = cos(phi2) / sqrt(1 - e2*sin(phi2)^2);
    
    q0 = (1-e2) * (sin(phi0)/(1-e2*sin(phi0)^2) - log((1-e*sin(phi0))/(1+e*sin(phi0)))/(2*e));
    q1 = (1-e2) * (sin(phi1)/(1-e2*sin(phi1)^2) - log((1-e*sin(phi1))/(1+e*sin(phi1)))/(2*e));
    q2 = (1-e2) * (sin(phi2)/(1-e2*sin(phi2)^2) - log((1-e*sin(phi2))/(1+e*sin(phi2)))/(2*e));
    
    n = (m1^2 - m2^2) / (q2 - q1);
    C = m1^2 + n*q1;
    rho0 = a * sqrt(C - n*q0) / n;
    
    phi = lat_deg(:) * pi/180;
    lam = lon_deg(:) * pi/180;
    
    q = (1-e2) * (sin(phi)./(1-e2*sin(phi).^2) - log((1-e*sin(phi))./(1+e*sin(phi)))/(2*e));
    rho = a * sqrt(C - n*q) / n;
    theta = n * (lam - lam0);
    
    x = x0 + rho .* sin(theta);
    y = y0 + rho0 - rho .* cos(theta);
end
