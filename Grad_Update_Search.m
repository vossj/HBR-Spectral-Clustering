function [A_inv, totalSteps, f] = Grad_Update_Search( ...
    X, whitenFlag, f_choice, alg_choice, enforceOrthogonality )
% function [A_inv, totalSteps, f] = Grad_Update_Search( ...
%     X, whitenFlag, f_choice, alg_choice, enforceOrthogonality )
% Inputs:
%   X -- K x N matrix of embedded data points.  Each column of X is assumed
%       to be a datum.
%   

    % X should contain each "data point" in a column.
    debug = 0;
    show_graph = 0;
    verbosity = 0;
    if ~exist( 'enforceOrthogonality', 'var' )
        enforceOrthogonality = true;
    end
    maxIterations = 1000;
    epsilon = 1e-5;
%     alg = 'fast descent';
%     alg = 'grad it';
%     alg = 'grad descent';
%     alg = 'grad ascent';

    %% Constants and Flags
    N = size(X, 2);
    d = size(X, 1);
    totalSteps = 0;
        
    %% preprocessing
    % Compute the second moment matrix and perform "whitening" (optional)
    L = eye(d);
    if whitenFlag
        N = size(X, 2);
        M2 = 1/N * X*X';
        L = M2^(-1/2);
        X = L*X;
        for i = 1:size(X, 2)
            X(:, i) = X(:, i) / norm(X(:, i));
        end
    end
    
    %% Define function required for the iterative update step
    [f, grad_f, g, dg, maxFlag, ddg] = selectFunction(f_choice);
    
    if maxFlag
%         alg = 'grad ascent';
        if strcmp(alg_choice, 'grad descent')
            alg = 'grad ascent';
        else
            alg = alg_choice;
        end
        maximizeSign = +1;
    else
        if strcmp(alg_choice, 'grad ascent')
            alg = 'grad descent';
        else
            alg = alg_choice;
        end
%         alg = 'grad descent';
        maximizeSign = -1;
    end
    
    lambda = 0.05;  % learning rate.
    
    A_inv = zeros(d, d);
    for iteration=1:d
        orthFlag = 1;  % Always start by enforcing orthogonality
        
        if verbosity >= 3
            fprintf('Searching for component %d:\n', iteration);
        end
        rowsFound = iteration - 1;
        
        %% choose a random starting vector and orthogonalize
        u = normrnd(0, 1, 1, d)';
        u_prev = zeros(d, 1);
        if rowsFound > 0
            u = u - A_inv(1:rowsFound, :)' * A_inv(1:rowsFound, :) * u;
        end
        u = u / norm(u);

        % Run update algorithm
        % NOTE:  During testing of algorithm, extra_iterations was not yet a concept.
        steps = 0;
        extra_iterations = 0;
        while ( 1 ) % Loop exit condition in the following if block
            if ( steps >= maxIterations+extra_iterations || norm(u_prev - u) <= epsilon || ...
                norm(u_prev + u) <= epsilon )
                if orthFlag && ~enforceOrthogonality && iteration > 1
                    orthFlag = 0; % relax orthogonality constraint.
                    % allow maxIterations steps with orthogonality not enforced.
                    extra_iterations = steps; 
                    if verbosity >= 3
                        fprintf('Orthogonality constraint relaxed')
                    end
                else
                    break;  % Exit condition
                end
            end
            
            if verbosity >= 3
                fprintf('.');
            end
            u_prev = u;

            if debug
                % uni-directional visualization
                gradient = grad_f(u, X, N);
                z = gradient - (u'*gradient)*u;
                theta = 0:.001:2*pi;
                ftheta = theta;
                for i = 1:size(ftheta, 2)
                    ftheta(i) = 1/N * sum((u'*X * cos(theta(i)) + z'*X * sin(theta(i))).^3);
                end
                figure(6);
                plot(theta, ftheta);
            end
            if (strcmp(alg, 'grad it'))
                % Apply the gradient iteration update.
                u = grad_f(u, X, N);
            elseif (strcmp(alg, 'fast descent'))
                % Calculate z the gradient direction projected into the
                % tangent space of the sphere at u.
                gradient = grad_f(u, X, N);
                z = gradient - (u'*gradient)*u;
                z_tilde = z / norm(z);
                
                % Calculate the derivatives along the curve on S^(d-1) at u
                % with tangent direction z (the great circle) under the parameterization
                % of polar coordinates.
                d_Gamma_0 = 1/N * sum(dg(X'*u).*(X'*z));
                dd_Gamma_0 = 1/N * sum( ddg(X'*u).*(X'*z).^2 - dg(X'*u).*(X'*u) );
                
                theta_new = 0 - d_Gamma_0 / dd_Gamma_0;
%                 if sign(theta_new) * maximizeSign ~= sign(z)
%                     % When Newton iteration goes in the wrong direction, fall back to a 
%                     % gradient decent variant.
% %                     theta_new = 0 + lambda * maximizeSign * norm(z);
%                     theta_new = lambda*maximizeSign*d_Gamma_0;
%                 end

                u = u*cos(theta_new) + z_tilde * sin(theta_new);
                
            elseif (strcmp(alg, 'grad descent'))
                raw_grad = grad_f(u, X, N);
                tan_grad = raw_grad - (u'*raw_grad)*u;
                u = u - lambda * tan_grad;
            elseif (strcmp(alg, 'grad ascent'))
                raw_grad = grad_f(u, X, N);
                tan_grad = raw_grad - (u'*raw_grad)*u;
                u = u + lambda * tan_grad;
            else
                fprintf(2, 'error')
            end

            % enforce orthogonality constraint
            % Note: If the enforcement of orthogonality is turned off by
            % default, then orthogonality is still enforced during the
            % early runs through this loop.  It is only relaxed after an
            % initial convergence (or time out in terms of number of steps)
            % is acheived.
            if orthFlag && iteration > 1
                u = u - A_inv(1:rowsFound, :)' * A_inv(1:rowsFound, :) * u;
            end
            
            u = u / norm(u);
            steps = steps + 1;
        end
        if (verbosity >= 2)
            if verbosity >= 3
                fprintf('\n')
            end
            fprintf('component %d found in %d iteration(s).\n', iteration, steps);
        end
        A_inv(iteration, :) = u;
        totalSteps = totalSteps + steps;
    end    
%     %% Graph the resulting points
    %% Plot the value of f at various points on the unit sphere.
    if show_graph && size(X, 1) == 3
        ww = [];
        [XX, YY, ZZ] = sphere(100);
        [MM, NN] = size(XX);
        W = [reshape(XX, [MM*NN, 1]) reshape(YY, [MM*NN, 1]), reshape(ZZ, [MM*NN, 1])];
        ff = zeros(size(W, 1), 1);
        for i = 1:size(W, 1)
            ff(i) = 1/N * sum( g(X'*W(i, :)') );
        end
%         ff = 1/N * sum(g(X'*W'), 1);
        ff = reshape(ff, [MM, NN]);

        figure(4);
        scatter3(X(1, :), X(2, :), X(3, :));
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        title('data points');
        hold on;
        scatter3(A_inv(:, 1), A_inv(:, 2), A_inv(:, 3), 'rx');

        surf(XX, YY, ZZ, ff);
        
        A = A_inv';
        scatter3(A(1, :), A(2, :), A(3, :), 'ks');
        Line3D(-2*A(:, 1)', 2*A(:, 1)', 'k');
        Line3D(-2*A(:, 2)', 2*A(:, 2)', 'k');
        Line3D(-2*A(:, 3)', 2*A(:, 3)', 'k');

        axis equal;
        hold off;
    end
   
    if verbosity >= 1
        fprintf(1, '\t\tTotal Steps:  %d\td:  %d\n', totalSteps, d);
    end

    %% return results
%      S = A_inv * X; % Recall that X has already been whitened / preprocessed.
    A_inv = A_inv * L;
end

function Line3D(a, b, op)
    plot3([a(1), b(1)], [a(2) b(2)], [a(3), b(3)], op);
end

