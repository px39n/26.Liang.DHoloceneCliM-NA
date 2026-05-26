% compare_vc_matlab.m
% Apply Pier's exact VC formula in MATLAB on the same input grids,
% compare with Python VC output.
% KEY FINDING: Pier uses window=30 (months), Python uses window=360 (months).

clear; close all; clc;

grid_dir = 'D:\Dataset\DPastCliM-NA\interim\grid_cal';

for iv = 1:2
    if iv == 1, var = 'tas'; else, var = 'pr'; end
    fprintf('\n========== %s ==========\n', upper(var));

    pcr_raw_file = fullfile(grid_dir, ['grid_pcr_raw_' var '.nc']);
    esm_cal_file = fullfile(grid_dir, ['grid_esm_cal_' var '.nc']);
    pcr_vc_py_file = fullfile(grid_dir, ['grid_pcr_cal_' var '.nc']);

    fprintf('Loading data...\n');
    pcr_raw = ncread(pcr_raw_file, var);
    esm_cal = ncread(esm_cal_file, var);
    pcr_vc_py = ncread(pcr_vc_py_file, var);

    [nx, ny, nt] = size(pcr_raw);
    fprintf('Grid: %d x %d, %d timesteps\n', nx, ny, nt);

    pcr_2d = double(reshape(pcr_raw, nx*ny, nt));
    esm_2d = double(reshape(esm_cal, nx*ny, nt));

    % --- Apply VC with BOTH window sizes ---
    for win = [30, 360]
        fprintf('\n--- Window = %d months (%.1f years) ---\n', win, win/12);
        tic;

        std_corr = movstd(esm_2d - movmean(esm_2d, win, 2, 'omitnan'), win, 0, 2, 'omitnan') ./ ...
            movstd(pcr_2d - movmean(pcr_2d, win, 2, 'omitnan'), win, 0, 2, 'omitnan');

        vc_ml = movmean(pcr_2d, win, 2, 'omitnan') + ...
            (pcr_2d - movmean(pcr_2d, win, 2, 'omitnan')) .* std_corr;

        vc_ml = single(reshape(vc_ml, nx, ny, nt));
        t_vc = toc;
        fprintf('  MATLAB VC done in %.1fs\n', t_vc);

        % Compare with Python VC (which uses window=360)
        diff = double(vc_ml) - double(pcr_vc_py);
        valid = ~isnan(diff);
        fprintf('  vs Python VC (win=360): max|diff|=%.2e, mean|diff|=%.2e\n', ...
            max(abs(diff(valid))), mean(abs(diff(valid))));

        % P10/P90 comparison
        obs_test_file = fullfile(grid_dir, ['grid_obs_test_' var '.nc']);
        if isfile(obs_test_file)
            obs_test = ncread(obs_test_file, var);
            year_test = ncread(obs_test_file, 'year');
            year_cal = ncread(pcr_raw_file, 'year');
            test_years = unique(year_test);
            test_mask = ismember(year_cal, test_years);

            vc_ml_test = vc_ml(:,:,test_mask);
            pcr_raw_test = pcr_raw(:,:,test_mask);

            obs_mean = mean(obs_test, 3, 'omitnan');
            obs_p10 = prctile(obs_test, 10, 3);
            obs_p90 = prctile(obs_test, 90, 3);

            for src_idx = 1:3
                if src_idx == 1
                    src = pcr_raw_test; src_name = 'PCR_raw';
                elseif src_idx == 2
                    src = pcr_vc_py(:,:,test_mask); src_name = 'Python_VC_360';
                else
                    src = vc_ml_test; src_name = sprintf('MATLAB_VC_%d', win);
                end

                s_mean = mean(src, 3, 'omitnan');
                s_p10 = prctile(src, 10, 3);
                s_p90 = prctile(src, 90, 3);

                b_mean = s_mean - obs_mean;
                b_p10 = s_p10 - obs_p10;
                b_p90 = s_p90 - obs_p90;

                v = ~isnan(b_mean);
                v10 = ~isnan(b_p10);
                v90 = ~isnan(b_p90);
                fprintf('  %s:\n', src_name);
                fprintf('    Mean bias: median=%.4f, MAE=%.4f\n', median(b_mean(v)), mean(abs(b_mean(v))));
                fprintf('    P10 diff:  median=%.4f, MAE=%.4f\n', median(b_p10(v10)), mean(abs(b_p10(v10))));
                fprintf('    P90 diff:  median=%.4f, MAE=%.4f\n', median(b_p90(v90)), mean(abs(b_p90(v90))));
            end
        end
    end
    fprintf('\n');
end

fprintf('Done.\n');
