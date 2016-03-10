function demo( algorithm, emb_type, f_choice,pert )
%     algorithm = 'k-means'
%     algorithm = 'discrete optimize';
%    algorithm = 'grad descent';
%     algorithm = 'grad it';

    if ~exist('pert', 'var')
        pert = 0.55;
    end
    if ~exist('emb_type', 'var')
        emb_type = 'rw';
    end
    if ~exist('algorithm', 'var')
        algorithm = {'grad descent'};
    end
    if ~exist('f_choice', 'var')
        f_choice = 'sigmoid';
    end
    if ~exist('descriptors', 'var')
        descriptors = {['FindOpt1-' f_choice], 'k-means'}
    end

    % Demo test
    [W, X, d] = GenTestData1( pert );
    figure(1);
    h = gcf;
    scatter(X(:,1), X(:, 2));
    set(gca, 'FontSize', 16);
    title('Original Data');
    axis('equal');
    xlim([-7.5, 7.5]);
    ylim([-7.5, 7.5]);
%         axis('equal', 'tight');
    xlabel('X');
    ylabel('Y');

    [f, grad_f, g, dg, maximizeFlag] = selectFunction( f_choice );

    [idx, EmbData, A] = SpectralAlg( W, d, algorithm, emb_type, ...
                                     f_choice, [], 1 );

    %% Split the data into small, medium, and large sets
    EmbData = EmbData';
    Counts = zeros(1, 3);
    for i = 1:3
        Counts(i) = sum( idx == i );
    end
    
    SortCounts = sort(Counts);
    SizeIndexes = zeros(1, 3);
    for i = 1:3
        % Note that this does assume that no two classes end up having the
        % same number of points.
        SizeIndexes(i) = find(Counts==SortCounts(i));
    end
    
    %% plot the categorical data
    msize = 7;
    figure(2);
    set(gca, 'FontSize', 16);
    plot(X(idx==SizeIndexes(1), 1), X(idx==SizeIndexes(1), 2), ' rx', 'MarkerSize', msize);
    hold on;
    plot(X(idx==SizeIndexes(2), 1), X(idx==SizeIndexes(2), 2), 'b +', 'MarkerSize', msize);
    plot(X(idx==SizeIndexes(3), 1), X(idx==SizeIndexes(3), 2), 'm d', 'MarkerSize', msize);
    hold off;
    %         tstring = sprintf( 'Clustered Data\n%s, \\gamma=%0.2f', ...
    %             descriptors{algIndex}, pert )
    tstring = 'Data Clustered';
    title(tstring);
    axis('equal');
    xlim([-7.5, 7.5]);
    ylim([-7.5, 7.5]);
    set(gca, 'xtick', [-5 0 5]);
    set(gca, 'ytick', [-5 0 5]);
    
    %% EMBEDDED DATA PLOT STUFF
    
    %% Redo signs of A in order to be near data with consistency rather
    %% than having a 1/6th chance.
    for i = 1:3
        % Select which cluster the current column of A belongs to and the
        % correct sign correction for said column
        ipMax = -1;
        ipSign = 0;
        for j = 1:3
            ip = sum(EmbData(idx==j, :))/Counts(j) * A(:, i);
            if abs(ip) > ipMax
                ipMax = abs(ip);
                ipSign = sign(ip);
            end
        end
        
        % Perform the sign correction
        A(:, i) = ipSign * A(:, i);
    end
    
    %% Plot the projections
    figure(3);
    set(gca, 'FontSize', 16);
    scatter3(EmbData(idx==SizeIndexes(1), 1), ...
             EmbData(idx==SizeIndexes(1), 2), ...
             EmbData(idx==SizeIndexes(1), 3), 10*msize, ' rx');
    hold on;
    scatter3(EmbData(idx==SizeIndexes(2), 1), ...
             EmbData(idx==SizeIndexes(2), 2), ...
             EmbData(idx==SizeIndexes(2), 3), 10*msize, 'b +');
    scatter3(EmbData(idx==SizeIndexes(3), 1), ...
             EmbData(idx==SizeIndexes(3), 2), ...
             EmbData(idx==SizeIndexes(3), 3), 10*msize, 'm d');
    
    %         tstring = sprintf( 'Embedded Data\n%s, \\gamma=%0.2f', ...
    %             descriptors{algIndex}, pert );
    tstring = 'Embedded Data Clustered'
    title(tstring);
    scale = 2.25;
    
    % Add A_inv lines to previous scatter plot
    Line3D(zeros(1, 3), scale*A(:, 1)', 'k');
    Line3D(zeros(1, 3), scale*A(:, 2)', 'k');
    Line3D(zeros(1, 3), scale*A(:, 3)', 'k');
    
    % Visualize f on the unit sphere
    [XX, YY, ZZ] = sphere(200);
    [MM, NN] = size(XX);
    WW = [reshape(XX, [MM*NN, 1]) reshape(YY, [MM*NN, 1]), reshape(ZZ, [MM*NN, 1])];
    ff = 1/size(EmbData, 1) * sum(g(EmbData*WW'), 1);
    ff = reshape(ff, [MM, NN]);
    ff = (ff - min(min(ff))) / (max(max(ff)) - min(min(ff)));
    ff = -ff;
    
    if strcmp(algorithm, 'k-means')
        colormap(gray)
        surf(XX, YY, ZZ)
    else
        colormap('default');
        surf(XX, YY, ZZ, ff, 'LineStyle', 'none');
    end
    hold off;
    axis('equal', 'tight');
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Line3D(a, b, op)
    plot3([a(1), b(1)], [a(2) b(2)], [a(3), b(3)], op, 'LineWidth', 2);
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, X, k] = GenTestData1(pert)
    %% Constants
    k = 3;  % number of regions
    if ~exist('pert', 'var')
        pert = 0.55;
    end
    
    %% Generate random data
    X = genShellPoints(1, 300, 2, pert);
    X = [X; genShellPoints(3, 450, 2, pert)];
    X = [X; genShellPoints(5, 600, 2, pert)];
    

    % similarity matrix W
    W = zeros(size(X, 1));
    W = exp(-5 * L2_distance(X', X').^2);
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = genShellPoints(radius, numPoints, d, pert)
    Sigma = eye(d);
    mu = zeros(numPoints, d);
    X = mvnrnd(mu, Sigma);
    for i = 1:size(X, 1)
        err = 1 + (2*rand*pert - pert)/radius;
        X(i, :) = (X(i, :) / norm(X(i, :))) * err * radius;
    end

end



