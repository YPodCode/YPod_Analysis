function F1_list = podF1Score(~, Y, y_hat, valList, ~, settingsSet)
%The F1 score is a balanced measure of precision and recall.  The Wikipedia
%page on the topic does a good job illustrating what those mean and what
%the F1 score indicates.

%Convert to arrays
Y = table2array(Y);

%Get the number of folds that were validated on
nFolds = settingsSet.nFoldRep;

%Get the index of the current validation and regression to get the right estimates
k = settingsSet.loops.k;
m=settingsSet.loops.m;

%Initialize Matrix of scores
F1_list = zeros(nFolds,2);

F1_temp = NaN(size(Y,2),2);

%Calculate F1 Score for each fold
for kk=1:nFolds
    temp_Y_cal = Y(valList~=kk,1);
    temp_Y_val = Y(valList==kk,1);
    temp_yhat_cal = y_hat.cal{m,k,kk};
    temp_yhat_val = y_hat.val{m,k,kk};
    
    %Loop through each category/column of Y
    if iscategorical(Y) || isa(Y,'numeric')
        for yy = categories(Y)
            F1_temp(yy,1) = calcF1(temp_Y_cal(temp_Y_cal==yy,1),temp_yhat_cal(temp_yhat_cal==yy,1));
            F1_temp(yy,2) = calcF1(temp_Y_val(temp_Y_val==yy,1),temp_yhat_val(temp_yhat_val==yy,1));
        end
    else
        for yy = 1:size(Y,2)
            F1_temp(yy,1) = calcF1(temp_Y_cal(:,yy),temp_yhat_cal(:,yy));
            F1_temp(yy,2) = calcF1(temp_Y_val(:,yy),temp_yhat_val(:,yy));
        end
    end
    F1_list(kk,1) = mean(F1_temp(:,1),'omitnan');%Calibrated
    F1_list(kk,2) = mean(F1_temp(:,2),'omitnan');%Validation
end
end

function f1score = calcF1(Y_cat,y_hat_cat)
%Calculate precision assuming estimates >=0.5 are positives
precision = sum((Y_cat==1)&y_hat_cat>=0.5)/sum(y_hat_cat>=0.5);

%Calculate recall assuming estimates >=0.5 are positives
recall = sum((Y_cat==1)&y_hat_cat>=0.5)/sum(Y_cat==1);

%Correct for instance where precision could be NaN when no predictions were right
if (sum(Y_cat==1)>0) && (sum(y_hat_cat>=0.5)==0)
    precision = 0;
end

%Calculate the F1 Score
f1score = 2*(precision*recall)/(precision+recall);

%Correct for instance where both recall and precision are 0
if (precision + recall)==0
    f1score=0;
end

end

