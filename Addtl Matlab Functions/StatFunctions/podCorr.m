function podcorr = podCorr(~, Y, y_hat, valList, ~, settingsSet)
%podR2 Calculate the Spearman's Rho rank correlation of the estimate and reference 

%Convert reference to an array for functionality
Y = table2array(Y);

%Number of folds that were fitted on
nfolds = settingsSet.nFoldRep;

%Initialize vector to hold R2 values
podcorr = zeros(nfolds,2);

%Loop through each validation/cablibration set to calculate the RMSE
for kk = 1:nfolds
    %Get the reference values for calibration and validation sets
    temp_Y_cal = Y(valList~=kk,1);
    temp_Y_val = Y(valList==kk,1);
    
    %Get the estimated values for calibration and validation sets
    temp_yhat_cal = y_hat.cal{settingsSet.loops.m,settingsSet.loops.k,kk};
    temp_yhat_val = y_hat.val{settingsSet.loops.m,settingsSet.loops.k,kk};
    
    %Calculate rank correlation and input into matrix (to be returned into the cell array "mdlStats")
    podcorr(kk,1) = corr(temp_Y_cal, temp_yhat_cal,'Type','Spearman');
    podcorr(kk,2) = corr(temp_Y_val, temp_yhat_val,'Type','Spearman');
end

end

