function [f, grad_f, g, dg, maximizeFlag, ddg] = selectFunction(s)
%   function [grad_f, g, dg, maximizeFlag, ddg] = selectFunction(s)
%   input: 
%   s -- a string containing the choice of function to be used as
%        the ICA contrast.  The options are a follows:  
%
%       'abs' -- absolute value function:  |x|
%       'abs m2.5' --  |x|^(2.5) (a fractional absolute moment)
%       'abs m3' --  |x|^3
%       'm3' --     x^3
%       'm4' --     x^4
%       'log cosh' -- log cosh(x)
%       'log cosh sq' -- (log cosh(x))^2
%       'gauss' --  exp(-x^2)
%		'sigmoid' -- 1/(1 + exp(-|x|))
%   	'sigmoid sq' -- 1/(1+exp(-x^2))
%
%   NOTE:  not all function selections can be used for each update
%   algorithm.
%
%   output:
%   All outputs are function handles which give ant estimate of the
%   contrast choice over the data.  f refers to the contrast function from
%   which the ICA implementation is derived.  Note that only f's
%   derivatives are used in the update step.
%
%   g -- the base function choice.
%   dg -- the first univariate derivative.
    grad_f = @(u, X, N) ...
        error('INVALID FUNCTION HANDLE USED FOR ALGORITHM CHOICE.');

    
    switch lower(s)
        case 'abs'
            g = @(z) -abs(z);
            dg = @(z) -sign(z);
            ddg = @(z) 0;
        case 'abs m2.5'
            g = @(z) abs(z).^2.5;
            dg = @(z) 2.5 * abs(z).^1.5 .* sign(z);
            ddg = @(z) 3.75 * abs(z).^0.5;
        case 'abs m3'
            g = @(z) abs(z).^3;
            dg = @(z) 3 * z.^2 .* sign(z);
            ddg = @(z) 6 * abs(z);
        case 'm3'
            % Note:  The third cumulant and third central moment happen to 
            % be the same.  Since we will renormalize and there is only one
            % term, we do not need to worry about the bias.
            g = @(z) z.^3;
            dg = @(z) 3 * z.^2;
            ddg = @(z) 6 * z;
        case 'm4'
            g = @(z) z.^4;
            dg = @(z) 4 * z.^3;
            ddg = @(z) 12 * z.^2
        case 'log cosh'
            g = @(z) -(log( cosh( z ) ));
            dg = @(z) - tanh(z);
            ddg = @(z) -(1-tanh(z).^2);
        case 'log cosh sq'
            g = @(z) (log( cosh( z ) )).^2;
            dg = @(z) 2 * log(cosh(z)) .* tanh(z);
            ddg = @(z) 2*(tanh(z).^2 + log(cosh(z)).*(1-tanh(z).^2));
        case 'gauss'
            g = @(z) exp(-z.^2);
            dg = @(z) -2 * z .* exp( -z.^2 );
            ddg = @(z) -2* exp(-z.^2) + 4*z.^2 .* exp(-z.^2);
        case 'sigmoid sq'
            g = @(z) 1 ./ ( 1 + exp( -z.^2 ) );
            dg = @(z) 2*exp(z.^2) .* z ./ (exp(z.^2) + 1).^2;
            ddg = @(z) 2*exp(z.^2).*(-2*z.^2 + exp(z.^2).*(2*z.^2 - 1) - 1) ./ (exp(z.^2) + 1).^3;
        case 'sigmoid'
            g = @(z) - 1 ./ ( 1 + exp( -abs(z) ) );
            dg = @(z) sign(z) .* exp(abs(z)) ./ (exp(abs(z).^2) + 1);
            ddg = @(z) (exp( abs(z) ) .* (exp(abs(z)) - 1)) ./ (exp(abs(z)) + 1).^3;
        case 'combo'    
            g = @(z) exp(-z.^2) + 0.25*abs(z).^3;
            dg = @(z) -2 * z .* exp( -z.^2 ) + 0.75*abs(z).^2 .* sign(z);
            ddg = @(z) -2* exp(-z.^2) + 4*z.^2 .* exp(-z.^2) + 1.5 * abs(z);
        otherwise
            error('INVALID FUNCTION CHOICE');
    end
    
    grad_f = @(u, X, N) 1/N * X*dg(X'*u);
    
    f = @(u, X, N) 1/N * sum(g(X'*u), 1);
    maximizeFlag = (dg(0.5) < 0.5*dg(1));
end
