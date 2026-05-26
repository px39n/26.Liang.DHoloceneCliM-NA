% Gap-free timing of profileNLL on REAL SEM matrix
% Mirrors Python's perf_counter gap-free diagnostic exactly
% Run: cd related_scripts; test_chol_det_timing

data = load('test_chol_det.mat');
W_sub = data.W_sub;
n_v = double(data.n_v);
fprintf('W_sub: %dx%d\n', size(W_sub));

% Build pre-allocated buffers (same as Python cached path)
I_v = eye(n_v);
neg_W_sub = -W_sub;
T = 60;
R_filled = randn(n_v, T);
nan_mask_v = false(n_v, T);
nan_mask_v(rand(n_v, T) < 0.15) = true;
R_filled(nan_mask_v) = 0;
R_v = R_filled;
n_valid_v = sum(~nan_mask_v, 2);

lambdas = [0.3, 0.5, 0.7, 0.85, 0.9];
n_trials = 3;

for li = 1:length(lambdas)
    lam = lambdas(li);
    
    for trial = 1:n_trials
        % ====== ENTRY ======
        t_entry = tic;
        
        % --- S1: unpack (no-op in MATLAB) ---
        t1 = tic;
        
        % --- S2: A = I - lam*W_sub ---
        A = I_v + lam * neg_W_sub;
        s2 = toc(t1);
        
        % --- S3: eps = R_v - lam*(W_sub @ R_filled) ---
        t3 = tic;
        eps_mat = R_v - lam * (W_sub * R_filled);
        s3 = toc(t3);
        
        % --- S4: NaN mask + sigma2 ---
        t4 = tic;
        eps_mat(nan_mask_v) = NaN;
        eps_sq = eps_mat.^2;
        eps_sq(nan_mask_v) = 0;
        sigma2_v = sum(eps_sq, 2) ./ max(n_valid_v, 1);
        s4 = toc(t4);
        
        % --- S5: sigma2 validity check ---
        t5 = tic;
        bad = any(sigma2_v <= 0 | isnan(sigma2_v));
        s5 = toc(t5);
        
        if bad
            total = toc(t_entry);
            fprintf('  lam=%.2f t%d: BAD sigma total=%.4fs\n', lam, trial, total);
            continue;
        end
        
        % --- S6: no copy needed (MATLAB is column-major native) ---
        t6 = tic;
        s6 = toc(t6);
        
        % --- S7: chol attempt ---
        t7 = tic;
        [R_chol, p] = chol(A);
        s7 = toc(t7);
        
        if p == 0
            % --- S8a: logdet from chol ---
            t8 = tic;
            logdetA = 2*sum(log(diag(R_chol)));
            s8 = toc(t8);
            s8_label = 's8_chol_det';
        else
            % --- S8b: det fallback (MATLAB approach) ---
            t8 = tic;
            d = det(A);
            logdetA = log(abs(d) + eps);
            s8 = toc(t8);
            s8_label = 's8_det_fb ';
            
            if d <= 0
                total = toc(t_entry);
                fprintf('  lam=%.2f t%d: det<=0 total=%.4fs\n', lam, trial, total);
                continue;
            end
        end
        
        % --- S9: likelihood terms ---
        t9 = tic;
        term1 = -(T/2) * sum(log(sigma2_v));
        term2 = T * logdetA;
        eps_safe = eps_mat;
        eps_safe(isnan(eps_safe)) = 0;
        quad = sum(sum(eps_safe.^2 ./ sigma2_v));
        term3 = -0.5 * quad;
        logLik = term1 + term2 + term3;
        penalty = -0.1 * log(max(1 - lam, 1e-10));
        nll = -(logLik - penalty);
        s9 = toc(t9);
        
        % ====== EXIT ======
        total = toc(t_entry);
        parts_sum = s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;
        gap = abs(total - parts_sum);
        
        if trial == n_trials
            fprintf('  lam=%.2f: total=%.4fs  gap=%.1es  chol_p=%d\n', lam, total, gap, p);
            fprintf('    s2_A=%.5f  s3_eps=%.5f  s4_sig=%.5f  s5_chk=%.5f\n', s2, s3, s4, s5);
            fprintf('    s6_copy=%.5f  s7_chol=%.5f  %s=%.5f  s9_terms=%.5f\n', s6, s7, s8_label, s8, s9);
        end
    end
end

fprintf('\nDone.\n');
