% Debug movmean window behavior
rng(42);
x = randn(1, 20);
writematrix(x, 'debug_x.csv');

% movmean with k=6 (even)
y6 = movmean(x, 6, 2, 'omitnan');
writematrix(y6, 'debug_movmean_k6.csv');

% movmean with k=5 (odd)
y5 = movmean(x, 5, 2, 'omitnan');
writematrix(y5, 'debug_movmean_k5.csv');

% movmean with k=30 on larger data
x30 = randn(1, 200);
writematrix(x30, 'debug_x30.csv');
y30 = movmean(x30, 30, 2, 'omitnan');
writematrix(y30, 'debug_movmean_k30.csv');

% movstd with k=30
ys30 = movstd(x30, 30, 0, 2, 'omitnan');
writematrix(ys30, 'debug_movstd_k30.csv');

fprintf('x = '); disp(x);
fprintf('movmean(x,6) = '); disp(y6);
fprintf('movmean(x,5) = '); disp(y5);
fprintf('First 5 of movmean(x30,30) = '); disp(y30(1:5));
fprintf('Last 5 of movmean(x30,30) = '); disp(y30(end-4:end));
