function [ X, y, X_xval, y_xval ] = extractCrossValidationSet(DATA)
%PICKCROSSVALIDATIONSET Summary of this function goes here
%   Detailed explanation goes here

numberOfBins = max(DATA(:,6));

[~,~,Z] = unique(DATA(:,6));
C = accumarray(Z,1:size(DATA,1),[],@(r){DATA(r,:)});

X_xval = [];
y_xval = [];
X = [];
y = [];


for i = 1:numberOfBins
    
    rand_index = [];
    min_val = 1;
    max_val = size(X,1);
    
    X_temp = C{i}(:, 1:5);
    y_temp = C{i}(:, 6);
    X_xval_temp = [];
    y_xval_temp = [];
    
    %Cross-validation matrix size
    size_xvalid = floor(size(X_temp,1)*0.2);
    
    for j = 1:size_xvalid
        r = randi([1, size(X_temp,1)]);
        rand_index = [rand_index; r];
    end
    
    for j = 1:size(rand_index)
        X_xval_temp = [X_xval_temp; X_temp(rand_index(j),:)];
        y_xval_temp = [y_xval_temp; y_temp(rand_index(j))];
    end
    
    [X_temp,PS] = removerows(X_temp,'ind',rand_index);
    [y_temp,PS] = removerows(y_temp,'ind',rand_index);
    
    X_xval = [X_xval; X_xval_temp];
    y_xval = [y_xval; y_xval_temp];
    X = [X; X_temp];
    y = [y; y_temp];
end

end

