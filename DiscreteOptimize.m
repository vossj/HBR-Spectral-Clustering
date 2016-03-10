function [A_inv] = DiscreteOptimize( X, function_choice, delta )
%   function [A_inv] = DiscreteOptimize( X, function_choice )
%   inputs:
%       X -- contains data points in its columns
%       function_choice -- see selectFunction.m for the possible values.
%         This determines which function is actually used as the contrast.
%       delta (optional) -- This is the no-go angle which determines that
%         once a cluster is isolated, no points within a delta angle of the
%         cluster center may be used to create a new cluster.  The default
%         value is pi/4.
%   return:
%       A_inv -- each row will correspond to the center of a clusters
    if exist('delta', 'var')
        assert(delta >= 0 && delta < pi/2);
        assert( isreal( delta ) );
        no_go_angle = delta;
    else
        no_go_angle = pi/4;
    end
    
    no_go_cos = cos(no_go_angle);

    N = size(X, 2);
    k = size(X, 1);
    [f, grad_f, g, dg, maxFlag] = selectFunction( function_choice );
    
    % fprintf(1, '\t\tDimensionality of eigenvectors:  (%d, %d)\n', N, k);

    A_inv = zeros(k, k);
    go_points = logical(ones(1, N));
    
    % Collect norms of columns of X
    X_norms = sqrt(sum(X.^2, 1));
    
%      % normalize X
%      X_norm = zeros(size(X));
%      for i = 1:N
%          X_norm(:, i) = X(:, i) / norm(X(:, i));
%      end
    
    % Calculate the function values at all points
    % tic;
    f_vals = zeros(N, 1);
    for i = 1:N
        u = X(:, i) / X_norms(i);
        f_vals(i) = f(u, X, N);
    end
    % fprintf(1, 'calculate function values:  ');
    % toc
    
    % Normalize the columns of X -- Probably faster but uses too much memory
    % U = X / diag(sparse(X_norms));
    % f_vals = f(U, X, N)';
    
    % tic;
    for i = 1:k
        if i > 1
            go_points = go_points & (abs(A_inv(i-1, :)*X) < no_go_cos * X_norms);
        end
        if sum(sum(go_points)) == 0
            warning('Not all clusters were identified');
            A_inv = A_inv(1:i-1, :);
            break;
        end
        
        if maxFlag
            [maxVal, ind] = max(abs(f_vals(go_points)));
        else
            [minVal, ind] = min(abs(f_vals(go_points)));
        end
        I = find(go_points, ind, 'first');
        id = I(end);
        A_inv(i, :) = X(:, id)' / norm(X(:, id));
    end
    % fprintf(1, 'find cluster centers:  ');
    % toc
end

