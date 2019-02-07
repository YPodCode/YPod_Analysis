function p_r2 = podR2(~, Y, y_hat, valList, ~, settingsSet)
%podR2 Calculate the coefficient of determination (R^2) of the estimated pollutants

%Convert reference to an array for functionality
Y = table2array(Y);

%Number of folds that were fitted on
nfolds = settingsSet.nFoldRep;

%Initialize vector to hold R2 values
p_r2 = zeros(nfolds,2);

%Loop through each validation/cablibration set to calculate the RMSE
for kk = 1:nfolds
    %Get the reference values for calibration and validation sets
    temp_Y_cal = Y(valList~=kk,1);
    temp_Y_val = Y(valList==kk,1);
    
    %Get the estimated values for calibration and validation sets
    temp_yhat_cal = y_hat.cal{settingsSet.loops.m,settingsSet.loops.k,kk};
    temp_yhat_val = y_hat.val{settingsSet.loops.m,settingsSet.loops.k,kk};
    
%     %Calculate mean values
%     mean_ycal = mean(temp_Y_cal,'omitnan');
%     mean_yval = mean(temp_Y_val,'omitnan');
%     
%     %Calculate sums
%     ss_res_cal = sum((temp_Y_cal - temp_yhat_cal).^2);
%     ss_res_val = sum((temp_Y_val - temp_yhat_val).^2);
%     ss_tot_cal = sum((temp_Y_cal - mean_ycal).^2);
%     ss_tot_val = sum((temp_Y_val - mean_yval).^2);
%     
%     %Calculate R2 and input into matrix (to be returned into the cell array "mdlStats")
%     p_r2(kk,1) = 1 - (ss_res_cal/ss_tot_cal);
%     p_r2(kk,2) = 1 - (ss_res_val/ss_tot_val);

    %Calculate R2 and input into matrix (to be returned into the cell array "mdlStats")
    calfit = fitlm(temp_Y_cal,temp_yhat_cal);
    valfit = fitlm(temp_Y_val,temp_yhat_val);
    p_r2(kk,1) = calfit.Rsquared.Ordinary;
    p_r2(kk,2) = valfit.Rsquared.Ordinary;
    
end

end

