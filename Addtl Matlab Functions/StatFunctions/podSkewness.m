function statObj = podSkewness(~, Y, y_hat, valList, ~, settingsSet)
%podSkewness Calculate the skewness of the distribution of concentrations
%for estimates and reference values for each fold

%Convert reference to an array for functionality
Y = table2array(Y(:,1));

%Number of folds that were fitted on
nfolds = settingsSet.nFoldRep;

%Initialize vector to hold R2 values
statObj = zeros(nfolds,4);

%Loop through each validation/cablibration set to calculate the skewness
for kk = 1:nfolds
    %Get the reference values for calibration and validation sets
    temp_Y_cal = Y(valList~=kk,1);
    temp_Y_val = Y(valList==kk,1);
    
    %Get the estimated values for calibration and validation sets
    temp_yhat_cal = y_hat.cal{settingsSet.loops.m,settingsSet.loops.k,kk};
    temp_yhat_val = y_hat.val{settingsSet.loops.m,settingsSet.loops.k,kk};
    
    try
        %Calculate mean values
        skew_ycal = skewness(temp_Y_cal);
        skew_yval = skewness(temp_Y_val);
        
        %Calculate sums
        skew_yhat_cal = skewness(temp_yhat_cal);
        skew_yhat_val = skewness(temp_yhat_val);
    catch
        continue
    end
    %Calculate put skewness values into object to return
    statObj(kk,1) = skew_ycal;
    statObj(kk,2) = skew_yval;
    statObj(kk,3) = skew_yhat_cal;
    statObj(kk,4) = skew_yhat_val;
end

end

