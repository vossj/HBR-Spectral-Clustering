function varargout = SpectralAlg(W, K, alg, normalization, f_choice, varargin)
%   function vargout = SpectralAlg(W, K, alg, normalization, f_choice, ...)
%   Required Inputs:
%       W -- An affinity matrix of graph vertex similarities.  If X is
%          passed in, then this is ignored and can be passed in as []
%          without any ill effects.
%       K -- The number of clusters to form.
%       alg -- Choice of algorithm to use for spectral clustering in the
%              spectral space.  Options are:
%           'k-means' -- Not a part of the research, but it is here anyway.
%           'grad descent' -- gradient descent algorithm for optimizing
%                 using arbitrary contrast function.
%           'discrete optimize' -- A fully deterministic algorithm which
%                 finds somewhat localized maxima of the contrast function
%                 in directions corresponding to data points.
%           'grad it' -- Gradient iteration.
%               Note:  This will only provide convergence to clusters for
%               certain choices of f_choice, most notably the moments (and
%               absolute moments) of order strictly greater than 2 should
%               work.  This is a faster converging algorithm than gradient
%               descent, and is based on the work in the paper "Basis
%               Learning as an Algorithmic Primitive" by Belkin,
%               Rademacher, and Voss.
%       normalization -- Specifies the choice of the graph Laplacian used
%               to create the spectral points.  The definitions of the 
%               various graph Laplacians come from Von Luxberg's Tutorial
%               on spectral clustering.  The options are:
%           'none' -- corresponds to L the unnormalized graph Laplacian
%           'rw' -- corresponds to L_rw
%           'sym' -- corresponds to L_sym
%       f_choice -- choice of contrast function to be used which using any 
%              algorithm other than k-means for the alg choice.  Current
%               options (possibly not all good) are:
%           'abs' -- absolute value function:  |x|
%           'abs m3' --  |x^3|
%           'm3' --     x^3
%           'm4' --     x^4
%           'log cosh' -- log cosh(x)
%           'log cosh sq' -- (log cosh(x))^2
%           'gauss' --  exp(-x^2)
%   Optional Inputs (appended to the end):
%       Optional inputs are passed in as pairs as:
%       SpectralAlg(W, K, alg, normalization, f_choice, op1_descr, op1_val,
%          op2_descr, op2_val, ...);
%       Descripritions:
%       'X' -- op_val should be a previously computed set of spectral
%           embedded data points passed back by this function.  If passed
%           in, then the computation of the embedding is skipped.  The
%           input for W is ignored and can be [] in this case.
%       'showPics' -- op_val should be a boolean value.  If op_val == 1 and
%           K==3, then it will display figures of the embedding process.
%       'enforce orth flag' -- op_val should be a boolean value determining
%           whether orthogonality is enforced between the recovered cluster
%           centers when running the grad descent algorithm.
%       'delta' -- op_val should be an angle between 0 and pi/2 in radians.
%           This angle is the parameter for the discrete optimize algorithm
%           and controls the minimum possible angle between cluster
%           centers.
%   Outputs:
%      idx -- Output when called with one output (the default).
%             Array of integer numbers giving the generated class labels of
%             the vertices given by the rows of the affinity matrix W.
%      X -- Output only if two outputs are requested.
%           Array of the embedded points.  Each collumn of X is an embedded
%           data point from the graph Laplacian eigenspace.
%      A -- Output only if at least three outputs are requested
%           Each column of A gives one of the recovered directions
%           corresponding to a class center (up to sign).

%     if ~exist('normalization', 'var')
%         normalization = 'rw';
% %         normalization = 'none';
%     end
%     if ~exist('f_choice', 'var')
%         f_choice = 'log cosh sq';
% %         f_choice = 'abs';
% %         f_choice = 'combo';
% %         f_choice = 'log cosh';
% %         f_choice = 'gauss';
% %          f_choice = 'abs m3';
%     end
    
    % default values
    X = [];
    showPics = false;
    delta = pi / 4;
    enforceOrthFlag = true;
    
    nVarargs = length(varargin);
    for indx = 2:2:nVarargs
        s = varargin{indx - 1};
        op_val = varargin{indx};
        if strcmp( s, 'X' )
            X = op_val;
        elseif strcmp( s, 'showPics' )
            showPics = op_val;
        elseif strcmp( s, 'enforce orth flag' )
            enforceOrthFlag = op_val;
        elseif strcmp( s, 'delta' )
            delta = op_val;
        end
    end

    %% SPECTRAL CLUSTERING -- IMPORTANT CODE STARTS HERE.
    if size(X, 1) == 0
        % tic;
        X = LaplacianProjection( W, K, normalization );
%         fprintf(1, '\t\tspectral projection:  ');
%         toc
    end
    
%     %% Project all data onto the unit sphere (optional, works for fewer functions)
%     D = 1 ./ sqrt(sum(X.^2, 1)); % Collects the inverse column norms;
%     D = sparse( 1:size(X, 2), 1:size(X, 2), D);
%     X = X * D; % Projects columns of X onto the unit sphere (divides by norms)
    

    %% Find the means
    % tic;  % Measure time to actually perform the algorithmic step in the embedded space
    
    whitenFlag = 0;
    calcA = 1;
    if ( strcmp(lower(alg), 'grad descent') || ...
            strcmp(lower(alg), 'grad ascent') || ...            
            strcmp(lower(alg), 'grad it') )
        
        [A_inv, totSteps] = Grad_Update_Search( ...
            X, whitenFlag, f_choice, alg, enforceOrthFlag);
    elseif (strcmp(lower(alg), 'grad ascent simul') || ...
            strcmp(lower(alg), 'grad descent simul') || ...
            strcmp(lower(alg), 'grad it simul' ) )
    
        [A_inv, totSteps] = Grad_Update_Search2( ...
            X, whitenFlag, f_choice, alg, enforceOrthFlag);
    elseif strcmp(lower(alg), 'spherical k-means') 
        % normalize the data points to the unit sphere. -- Not necessary if
        % using cosine distance.
        % X_norms = sqrt(sum(X.^2, 1));
        % X = X / diag(sparse(X_norms));
        
        
        try
            [id, C] = kmeans(X', K, 'emptyaction', 'singleton', ...
                             'Distance', 'cosine');
        catch
            warning(['Falling back on k-means with standard Euclidean ' ...
                     'distance.  This is probably a bad thing.  ' ...
                     'Vectors are not unit normalized.']);
            [id, C] = kmeans(X', K, 'emptyaction', 'singleton' );
        end
%         id = kmeans(X', K, 'emptyaction', 'drop');
        
        A = C';
        calcA = 0; % A (cluster centers) is implititly calculated by K-Means

    elseif strcmp(lower(alg), 'k-means') 
        [id, C] = kmeans(X', K, 'emptyaction', 'singleton' );
        A = C';
        calcA = 0; % A (cluster centers) is implititly calculated by K-Means
    elseif strcmp(lower(alg), 'unit k-means') 
        % normalize the data points to the unit sphere.
        X_norms = sqrt(sum(X.^2, 1));
        Xunit = X / diag(sparse(X_norms));
        
        [id, C] = kmeans(Xunit', K, 'emptyaction', 'singleton' );
        
        A = C';
        calcA = 0; % A (cluster centers) is implititly calculated by K-Means        
    else
        % tic;
        A_inv = DiscreteOptimize(X, f_choice, delta);
%         fprintf(1, '\t\tDiscreteOptimize:  ');
    end
%     fprintf('time to classify points in embedded space:\t');
%     toc

    %% Perform clustering
    if calcA
        for i = 1:size(A_inv, 1)
            s = sign(sum(A_inv(i, :)*X));
            if s ~= 0
                A_inv(i, :) = s*A_inv(i, :);
            end
        end

        if whitenFlag
            A = inv(A_inv);
            for i = 1:size(A, 2)
                A(:, i) = A(:, i) / norm(A(:, i));
            end
        else
            A = A_inv';
        end
        clear('A_inv');
        [M, id] = max( abs(X'*A), [], 2 );
    end    
            
    %% Prepare the outputs
    nOutputs = nargout;
    if nargout < 1
        nOutputs = 1;
    end
    varargout = cell(1, nOutputs);
    varargout{1} = id;
    if nOutputs >= 2
        varargout{2} = X;
    end
    if nOutputs >= 3
        varargout{3} = A;
    end
    
    %% SPECTRAL CLUSTERING -- IMPORTANT CODE ENDS HERE.
    
    %% Plotting stuff in the 3-D case
    if showPics % only works with 3-D data
        [f, grad_f, g, dg, maximizeFlag] = selectFunction( f_choice );
        %% Plot the projections
        scatter3(X(1, :), X(2, :), X(3, :), [], (id-1) / 2);
%        scatter3(X(1, :), X(2, :), X(3, :));
        title('Rows of the Eigenspace');
        scale = 2.5;
        set(gca, 'FontSize', 16);
        xlabel('X');
        ylabel('Y');
        zlabel('Z');

        % Add A_inv to previous scatter plot
        hold on;
        % scatter3(A(1, :), A(2, :), A(3, :), 'ks');
        Line3D(zeros(1, 3), scale*A(:, 1)', 'k');
        Line3D(zeros(1, 3), scale*A(:, 2)', 'k');
        Line3D(zeros(1, 3), scale*A(:, 3)', 'k');
        
        % Visualize f on the unit sphere
        [XX, YY, ZZ] = sphere(50);
        [MM, NN] = size(XX);
        WW = [reshape(XX, [MM*NN, 1]) reshape(YY, [MM*NN, 1]), reshape(ZZ, [MM*NN, 1])];
        ff = 1/size(X, 2) * sum(g(X'*WW'), 1);
        ff = reshape(ff, [MM, NN]);
        ff = (ff - min(min(ff))) / (max(max(ff)) - min(min(ff)));

        surf(XX, YY, ZZ, ff);

        hold off;
        axis('equal', 'tight');
        
    end
end

function Line3D(a, b, op)
    plot3([a(1), b(1)], [a(2) b(2)], [a(3), b(3)], op, 'LineWidth', 2);
end




