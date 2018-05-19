DATA = csvread('../resources/recipeData_mod.csv',1,0);

X = DATA(:, 1:5);
y = DATA(:,6);


numberOfBins = max(y(:))
countsPercentage = 100 * hist(y(:), numberOfBins) / numel(y)
max(countsPercentage(:))