% Benchmark MATLAB scatteredInterpolant gridding speed.
% Pair with benchmark_gridding.py for Python comparison.

fprintf('=== Gridding Benchmark (MATLAB scatteredInterpolant) ===\n\n');

% Use raw coordinates (no projection) since Mapping Toolbox unavailable.
% Timing comparison is still valid: same grid size and interpolation method.
lat_range = [15 75];
lon_range = [-170 -50];
res = 0.20;

lat = lat_range(1):res:lat_range(2);
lon = lon_range(1):res:lon_range(2);
ny = length(lat);
nx = length(lon);
[longrid, latgrid] = meshgrid(lon, lat);
xgrid = longrid(:);
ygrid = latgrid(:);
M = length(xgrid);
fprintf('Grid: %dx%d = %d query points\n\n', ny, nx, M);

rng(42);
sizes = [800, 7000];
for si = 1:length(sizes)
    n_st = sizes(si);
    x_st = -165 + 110*rand(n_st, 1);
    y_st = 20 + 50*rand(n_st, 1);
    vals = randn(n_st, 1);

    fprintf('--- %d stations ---\n', n_st);

    % Single interpolation (scatteredInterpolant creation + evaluation)
    tic;
    F = scatteredInterpolant(x_st, y_st, vals, 'natural', 'nearest');
    result = F(xgrid, ygrid);
    t_single = toc;
    fprintf('  Single interp (create+eval): %.3f s\n', t_single);

    % 125 interpolations (same stations, different values)
    V = randn(n_st, 125);
    tic;
    for k = 1:125
        F.Values = V(:,k);
        result = F(xgrid, ygrid);
    end
    t_batch = toc;
    fprintf('  125x interp (reuse F): %.3f s\n', t_batch);
    fprintf('  Per-timestep: %.3f s\n\n', t_batch/125);
end

fprintf('Done.\n');
