DATA = csvread('../resources/recipeData_mod.csv',1,0);

[X, y, X_xval, y_xval] = extractCrossValidationSet(DATA);

%STEP 1: PREDICTION USING LOGREG WITH ONE VS ALL APPROACH FOR LBP FEATURES
theta = OneVsAll(X, y, 177, 0);

%For xval
p = predictOneVsAll(theta, X_xval);

%Where it found the right answer
hits = (p == y_xval);

%Sum number of hits
sumHits = sum(hits(:) == 1);

%Accuracy value using xval
accVal = (sumHits/size(y_xval, 1)) * 100;
fprintf('\nLogReg OvA accuracy(%): %d \n', accVal*100);