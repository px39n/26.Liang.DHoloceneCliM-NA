%% Step F2 verification: ARMA(1,1) coefficient comparison
%  Manual ARMA(1,1) fitting via CSS (Conditional Sum of Squares)
%  to compare with Python's statsmodels ARIMA MLE.

clear; close all;

ml_dir = 'D:\Dataset\DPastCliM-NA\verification\step_F\matlab';
py_dir = 'D:\Dataset\DPastCliM-NA\verification\step_F\python';

%% Load eps_mat from SEM fitting
fid = fopen(fullfile(ml_dir, 'eps_mat.bin'), 'r');
hdr = fread(fid, 2, 'int32');
n_st = hdr(1); T = hdr(2);
eps_mat = fread(fid, [T, n_st], 'single')';
fclose(fid);

% Load sem_params for valid_mask
fid = fopen(fullfile(ml_dir, 'sem_params.bin'), 'r');
lambda_hat = fread(fid, 1, 'double');
threshold_best = fread(fid, 1, 'double');
n_st2 = fread(fid, 1, 'int32');
T2 = fread(fid, 1, 'int32');
fclose(fid);

valid_mask = sum(~isnan(eps_mat), 2) >= 30;
fprintf('Loaded: %d stations, %d years, %d valid\n', n_st, T, sum(valid_mask));

%% Fit ARMA(1,1) via CSS for each valid station
ar_ml = zeros(n_st, 1);
ma_ml = zeros(n_st, 1);
var_ml = zeros(n_st, 1);

for i = 1:n_st
    if ~valid_mask(i), continue; end
    y = eps_mat(i, :)';
    y = y(~isnan(y));
    n = length(y);
    if n < 12, var_ml(i) = var(y); continue; end
    
    % CSS: minimize sum of squared residuals
    % ARMA(1,1): y(t) = phi*y(t-1) + theta*e(t-1) + e(t)
    best_css = inf;
    best_phi = 0; best_theta = 0;
    
    for phi_try = linspace(-0.95, 0.95, 39)
        for theta_try = linspace(-0.95, 0.95, 39)
            e = zeros(n, 1);
            for t = 2:n
                e(t) = y(t) - phi_try*y(t-1) - theta_try*e(t-1);
            end
            css = sum(e.^2);
            if css < best_css
                best_css = css;
                best_phi = phi_try;
                best_theta = theta_try;
            end
        end
    end
    
    % Refine with finer grid around best
    for phi_try = linspace(max(best_phi-0.1,-0.99), min(best_phi+0.1,0.99), 21)
        for theta_try = linspace(max(best_theta-0.1,-0.99), min(best_theta+0.1,0.99), 21)
            e = zeros(n, 1);
            for t = 2:n
                e(t) = y(t) - phi_try*y(t-1) - theta_try*e(t-1);
            end
            css = sum(e.^2);
            if css < best_css
                best_css = css;
                best_phi = phi_try;
                best_theta = theta_try;
            end
        end
    end
    
    ar_ml(i) = best_phi;
    ma_ml(i) = best_theta;
    var_ml(i) = best_css / n;
end

fprintf('MATLAB CSS fit done.\n');
fprintf('AR: mean=%.4f, std=%.4f\n', mean(ar_ml(valid_mask)), std(ar_ml(valid_mask)));
fprintf('MA: mean=%.4f, std=%.4f\n', mean(ma_ml(valid_mask)), std(ma_ml(valid_mask)));
fprintf('Var: mean=%.4f, std=%.4f\n', mean(var_ml(valid_mask)), std(var_ml(valid_mask)));

%% Load Python's ARMA coefficients
fid = fopen(fullfile(py_dir, 'arma_coeffs.bin'), 'r');
n_st_py = fread(fid, 1, 'int32');
ar_py = fread(fid, n_st_py, 'double');
ma_py = fread(fid, n_st_py, 'double');
var_py = fread(fid, n_st_py, 'double');
vm_py = fread(fid, n_st_py, 'int32');
fclose(fid);

%% Compare
vm = valid_mask & logical(vm_py);
ar_diff = abs(ar_ml(vm) - ar_py(vm));
ma_diff = abs(ma_ml(vm) - ma_py(vm));
var_diff = abs(var_ml(vm) - var_py(vm));

fprintf('\n=== ARMA(1,1) Coefficient Comparison ===\n');
fprintf('Python AR: mean=%.4f, std=%.4f\n', mean(ar_py(vm)), std(ar_py(vm)));
fprintf('Python MA: mean=%.4f, std=%.4f\n', mean(ma_py(vm)), std(ma_py(vm)));
fprintf('Python Var: mean=%.4f, std=%.4f\n', mean(var_py(vm)), std(var_py(vm)));
fprintf('AR: max|diff|=%.4f, mean|diff|=%.4f\n', max(ar_diff), mean(ar_diff));
fprintf('MA: max|diff|=%.4f, mean|diff|=%.4f\n', max(ma_diff), mean(ma_diff));
fprintf('Var: max|diff|=%.4f, mean|diff|=%.4f\n', max(var_diff), mean(var_diff));
% manual correlation
r_ar = sum((ar_ml(vm)-mean(ar_ml(vm))).*(ar_py(vm)-mean(ar_py(vm))))/(std(ar_ml(vm))*std(ar_py(vm))*sum(vm));
r_var = sum((var_ml(vm)-mean(var_ml(vm))).*(var_py(vm)-mean(var_py(vm))))/(std(var_ml(vm))*std(var_py(vm))*sum(vm));
fprintf('Correlation: AR=%.4f, Var=%.4f\n', r_ar, r_var);

% Save MATLAB coefficients
fid = fopen(fullfile(ml_dir, 'arma_coeffs_css.bin'), 'w');
fwrite(fid, n_st, 'int32');
fwrite(fid, ar_ml, 'double');
fwrite(fid, ma_ml, 'double');
fwrite(fid, var_ml, 'double');
fclose(fid);

fprintf('MATLAB CSS coefficients saved.\n');
