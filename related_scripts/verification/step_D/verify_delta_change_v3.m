% Debug intermediate values for comparison.
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

i_cal = (T_full - T_cal + 1):T_full;
M_full = reshape(field_full, ny*nx, T_full);
M_cal  = reshape(field_full(:,:,i_cal), ny*nx, T_cal);

mu_cal = mean(M_cal, 2, 'omitnan');
mu_mov = movmean(M_full, 30, 2, 'omitnan');

fprintf('M_full(1,1) = %.10f\n', M_full(1,1));
fprintf('mu_cal(1) = %.10f\n', mu_cal(1));
fprintf('mu_mov(1,1) = %.10f\n', mu_mov(1,1));
fprintf('mu_adj(1,1) = %.10f\n', mu_mov(1,1) - mu_cal(1));
fprintf('\n');

% Check movmean window at t=1
% For k=30: kb=14, kf=15 -> window at t=1: [max(1,1-14):min(200,1+15)] = [1:16]
win_vals = M_full(1, 1:16);
manual_mean = mean(win_vals);
fprintf('Manual movmean at t=1, pixel=1: mean of M(1,1:16) = %.10f\n', manual_mean);
fprintf('movmean result: %.10f\n', mu_mov(1,1));
fprintf('Match: %d\n', abs(manual_mean - mu_mov(1,1)) < 1e-15);

% Print cal period values for pixel 1
fprintf('\nCal period M(1, 126:130) = ');
fprintf('%.4f ', M_full(1, 126:130));
fprintf('\n');
fprintf('Cal period size: %d\n', length(i_cal));
