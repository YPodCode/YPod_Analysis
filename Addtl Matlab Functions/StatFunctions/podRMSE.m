function p_RMSE = podRMSE(~, Y, y_hat, valList, ~, settingsSet)
%podRMSE Calculate the RMSE of the estimated pollutants

%Convert reference to an array for functionality
Y = table2array(Y);

%Number of folds that were fitted on
nfolds = settingsSet.nFoldRep;

%Get the index of the current validation and regression to get the right estimates
k = settingsSet.loops.k;
m=settingsSet.loops.m;

%Initialize vector to hold error values
p_RMSE = zeros(nfolds,2);

%Loop through each validation/cablibration set to calculate the RMSE
for kk = 1:nfolds
    %Get the reference values for calibration and validation sets
    temp_Y_cal = Y(valList~=kk,1);
    temp_Y_val = Y(valList==kk,1);
    
    %Get the estimated values for calibration and validation sets
    if isa(y_hat.cal{m,k,kk},'table')
        temp_yhat_cal = table2array(y_hat.cal{m,k,kk});
        temp_yhat_val = table2array(y_hat.val{m,k,kk});
    else
        temp_yhat_cal = y_hat.cal{m,k,kk};
        temp_yhat_val = y_hat.val{m,k,kk};
    end
    
    %Calculate RMSE and input into matrix (to be returned into the cell array "mdlStats")
    p_RMSE(kk,1) = sqrt(mean( (temp_Y_cal-temp_yhat_cal).^2 ));
    p_RMSE(kk,2) = sqrt(mean( (temp_Y_val-temp_yhat_val).^2 ));
end

end

