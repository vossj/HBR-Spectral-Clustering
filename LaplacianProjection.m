function X = LaplacianProjection( W, d, normalization, SparseFlag )
%   inputs:
%       W -- Similarity matrix
%       d -- number of dimensions on which to project (d smalles
%           eigenvectors)
%       normalization -- choose from 'none', 'sym' (default), or 'rw'
%       SparseFlag -- Either true or false.  Determines whether
%           the eigendecomposition used during the spectral embedding step
%           uses sparse or dense algorithms.  By default, this will be true
%           if W is sparse and false if W is a dense matrix.
%   return:
%       X -- Each row is an eigenvector of the Laplacian, making each
%           collumn a point projected into the "simplex space"

    % Set default parameters
    if ~exist('normalization', 'var')
        normalization = 'sym'
    end
    if ~exist('SparseFlag', 'var')
        SparseFlag = true;
    end

    % Compute the standard Laplacian
    DD = sum(W, 2);
    D = diag(DD);
    L = D - W;
    
    % Note:  addition of the identity matrix does not change the eigen
    % vectors, though it does change the eigenvalues.  It does help with
    % matrix conditioning for the eigs routine.
    if strcmp( lower(normalization), 'none' )
        opts.issym = 1;  opts.isreal = 1;
        X = EmbedLaplacian( L, d, SparseFlag, opts );
    elseif strcmp( lower(normalization), 'sym' )
        opts.issym = 1;  opts.isreal = 1;
        D_inv_half = diag( DD.^(-1/2) );
        L_sym = D_inv_half * L * D_inv_half;
        clear('L');
        X = EmbedLaplacian( L_sym, d, SparseFlag, opts );
    elseif strcmp( lower(normalization), 'rw' )
        opts.issym = 0;  opts.isreal = 1;
        D_inv = diag( DD.^(-1) );
        L_rw = D_inv * L;
        clear('L')
        X = EmbedLaplacian( L_rw, d, SparseFlag, opts );
    else
        error('Bad choice for input parameter normalization');
    end    
end

function X = EmbedLaplacian( L, d, SparseFlag, opts )
    n = size(L, 1);
    
    if SparseFlag
        I_shift = 100*eps*speye( size(L) );
        [X, Lambda] = eigs(L + I_shift, d, 'SM', opts);
        if (~isreal( X )) || (~isreal( Lambda))
            warning('Ignoring imaginary components in the eigendecomposition.');
            Lambda = real(Lambda);
            X = real(X);
        end
        %      Lambda = Lambda - I_shift(1:d, 1:d);
    else
        % I_shift = 100*eps*eye( size(L) );
        [V, Lambda] = eig( full(L) );
        % Only use real information as any imaginary information should not
        % have arisen.
        if (~isreal( V )) || (~isreal( Lambda))
            warning('Ignoring imaginary components in the eigendecomposition.');
            Lambda = real(Lambda);
            V = real(V);
        end
        
        [eig_vals, I] = sort( diag( Lambda ) );
        
        N = size(V, 2);
        X = V(:, I(1:d));
    end
    X = sqrt(n) * X';
end
