% Verify variance correction (movstd ratio scaling)
% Generates synthetic test data, applies MATLAB's correction, saves for Python comparison.

rng(42);

n_cells = 50;   % grid cells
n_time  = 200;  % time steps
n_year_mov = 30;

% Synthetic PCR output: smooth trend + noise
t = (1:n_time)';
pcr = repmat(sin(2*pi*t/100)', n_cells, 1) + 0.5*randn(n_cells, n_time);

% Synthetic ESM field: different trend + different noise amplitude
esm = repmat(0.8*sin(2*pi*t/100 + 0.3)', n_cells, 1) + 1.2*randn(n_cells, n_time);

% Save inputs
writematrix(pcr, 'test_pcr_input.csv');
writematrix(esm, 'test_esm_input.csv');

% Apply variance correction (exact copy of Guaita's code)
std_correction = movstd(esm - movmean(esm, n_year_mov, 2, 'omitnan'), n_year_mov, 0, 2, 'omitnan') ./ ...
    movstd(pcr - movmean(pcr, n_year_mov, 2, 'omitnan'), n_year_mov, 0, 2, 'omitnan');
pcr_adjusted = movmean(pcr, n_year_mov, 2, 'omitnan') + ...
    (pcr - movmean(pcr, n_year_mov, 2, 'omitnan')) .* std_correction;

% Save outputs
writematrix(pcr_adjusted, 'test_pcr_adjusted_matlab.csv');

% Also save intermediate values for debugging
pcr_movmean = movmean(pcr, n_year_mov, 2, 'omitnan');
esm_movmean = movmean(esm, n_year_mov, 2, 'omitnan');
pcr_movstd = movstd(pcr - pcr_movmean, n_year_mov, 0, 2, 'omitnan');
esm_movstd = movstd(esm - esm_movmean, n_year_mov, 0, 2, 'omitnan');

writematrix(pcr_movmean, 'test_pcr_movmean_matlab.csv');
writematrix(esm_movmean, 'test_esm_movmean_matlab.csv');
writematrix(pcr_movstd, 'test_pcr_movstd_matlab.csv');
writematrix(esm_movstd, 'test_esm_movstd_matlab.csv');
writematrix(std_correction, 'test_std_correction_matlab.csv');

fprintf('Done. Saved test data and MATLAB results.\n');
fprintf('PCR adjusted: size = %d x %d\n', size(pcr_adjusted));
fprintf('Any NaN in adjusted: %d\n', any(isnan(pcr_adjusted(:))));
fprintf('Any Inf in std_correction: %d\n', any(isinf(std_correction(:))));
