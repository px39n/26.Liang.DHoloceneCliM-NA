% Save field values as simple binary for Python comparison.
out_dir = 'D:\Dataset\DPastCliM-NA\verification\step_D';

rng(42);
ny = 10; nx = 12;
T_cal = 75; T_full = 200;

lat = linspace(20, 60, ny)';
lon = linspace(-140, -60, nx)';
[LON, LAT] = meshgrid(lon, lat);

field_full = zeros(ny, nx, T_full);
for t = 1:T_full
    trend = 2 * sin(2*pi*t/T_full) * ones(ny, nx);
    spatial = LAT/30 + LON/100;
    field_full(:,:,t) = trend + spatial + 0.1*randn(ny, nx);
end

% Save raw field as binary (column-major)
fid = fopen(fullfile(out_dir, 'field_full.bin'), 'w');
fwrite(fid, [ny, nx, T_full], 'int32');
fwrite(fid, field_full(:), 'float64');
fclose(fid);

% Compute mu_adj
i_cal = (T_full - T_cal + 1):T_full;
M_full = reshape(field_full, ny*nx, T_full);
M_cal  = reshape(field_full(:,:,i_cal), ny*nx, T_cal);

mu_cal = mean(M_cal, 2, 'omitnan');
mu_mov = movmean(M_full, 30, 2, 'omitnan');
mu_adj = mu_mov - mu_cal;

% Save mu_adj as binary
fid = fopen(fullfile(out_dir, 'mu_adj_tas.bin'), 'w');
fwrite(fid, mu_adj(:), 'float64');
fclose(fid);

% Print some values for sanity check
fprintf('field(1,1,1) = %.10f\n', field_full(1,1,1));
fprintf('field(2,1,1) = %.10f\n', field_full(2,1,1));
fprintf('field(1,2,1) = %.10f\n', field_full(1,2,1));
fprintf('field(1,1,2) = %.10f\n', field_full(1,1,2));
fprintf('mu_adj(1,1) = %.10f\n', mu_adj(1,1));
fprintf('mu_adj(1,100) = %.10f\n', mu_adj(1,100));
fprintf('mu_adj(60,100) = %.10f\n', mu_adj(60,100));
fprintf('Done.\n');
