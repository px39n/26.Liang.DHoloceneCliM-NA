% Benchmark variance correction at production scale
% Compare MATLAB movmean/movstd speed on realistic grid sizes.

fprintf('=== Variance Correction Benchmark (MATLAB) ===\n\n');

rng(42);
n_year_mov = 30;

sizes = {
    'Small (50x200)',      50,   200;
    'Medium (18030x1500)', 18030, 1500;
    'Full (180901x1500)',  180901, 1500;
};

for k = 1:size(sizes, 1)
    label   = sizes{k, 1};
    n_cells = sizes{k, 2};
    n_time  = sizes{k, 3};

    fprintf('--- %s ---\n', label);

    if n_cells > 100000
        fprintf('  (Full grid: %.1f GB RAM needed, skipping if insufficient)\n', ...
            n_cells * n_time * 8 * 8 / 1e9);
        try
            pcr = randn(n_cells, n_time);
            esm = randn(n_cells, n_time);
        catch ME
            fprintf('  SKIPPED: %s\n\n', ME.message);
            continue
        end
    else
        pcr = randn(n_cells, n_time);
        esm = randn(n_cells, n_time);
    end

    % Warm up
    if n_cells <= 18030
        tmp = movmean(pcr(1:min(100,n_cells), :), n_year_mov, 2, 'omitnan'); %#ok<NASGU>
    end

    % Benchmark movmean
    tic;
    pcr_mm = movmean(pcr, n_year_mov, 2, 'omitnan');
    t_mm = toc;
    fprintf('  movmean:  %.3f s\n', t_mm);

    % Benchmark movstd
    tic;
    pcr_ms = movstd(pcr - pcr_mm, n_year_mov, 0, 2, 'omitnan');
    t_ms = toc;
    fprintf('  movstd:   %.3f s\n', t_ms);

    % Benchmark full variance correction
    tic;
    pcr_mm2 = movmean(pcr, n_year_mov, 2, 'omitnan');
    esm_mm2 = movmean(esm, n_year_mov, 2, 'omitnan');
    std_c = movstd(esm - esm_mm2, n_year_mov, 0, 2, 'omitnan') ./ ...
            movstd(pcr - pcr_mm2, n_year_mov, 0, 2, 'omitnan');
    std_c(~isfinite(std_c)) = 1;
    adjusted = pcr_mm2 + (pcr - pcr_mm2) .* std_c;
    t_full = toc;
    fprintf('  Full varcorr: %.3f s\n', t_full);
    fprintf('  Peak RAM est: %.1f GB\n\n', n_cells * n_time * 8 * 8 / 1e9);
end

fprintf('Done.\n');
