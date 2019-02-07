function acfPlot(~,X,Y,~,~,~,settingsSet)
%Plots the correlation between X variables and the Y matrix

%Get the time averaging to guess appropriate lags
deltaT = settingsSet.timeAvg;
nlags = hours(12)/minutes(deltaT);
%Assume that first column (only column) of Y has data
yarray = table2array(Y(:,1));
%Remove NaNs
yarray(isnan(yarray))=0;
%Get pollutant name
pollutant = settingsSet.fileList.colocation.reference.files.pollutants{settingsSet.loops.i};

nplots = 0;
for i = 1:size(X,2)
    %Get variable name
    VarName = X.Properties.VariableNames{i};
    %Only plot this for designated sensors (not T, P, Rh, etc)
    isSensor = 0;
    for j = 1:length(settingsSet.podSensors)
        if ~isempty(regexpi(VarName,settingsSet.podSensors{j}))
            isSensor = 1;
        end
    end
    if isSensor == 1
        nplots = nplots+1;
    end
end

plotn = 1;
figure('Position',get( groot, 'Screensize' ));
for i = 1:size(X,2)
    %Get variable name
    VarName = X.Properties.VariableNames{i};
    %Get pod name
    podName = settingsSet.podList.podName{settingsSet.loops.j};
    
    %Only plot this for designated sensors (not T, P, Rh, etc)
    isSensor = 0;
    for j = 1:length(settingsSet.podSensors)
        if ~isempty(regexpi(VarName,settingsSet.podSensors{j}))
            isSensor = 1;
        end
    end
    if isSensor == 1
        %Get X as array
        xarray = table2array(X(:,i));
        %Remove NaNs
        xarray(isnan(xarray))=0;
        
        %Plot the cross correlation
        ncol = round(sqrt(nplots),0);
        subplot(ceil(nplots/ncol),ncol,plotn);
        crosscorr(yarray, xarray,nlags);
        
        %Give the plot a better title
        ylabel('XCF');ylim([-1 1])
        if plotn==1
            title([pollutant ': ' VarName])
        else
            title(VarName)
        end
        %Display the max and min correlations
        [xcf,lags,~] = crosscorr(yarray, xarray,nlags);
        line([0 0],[floor(min(xcf)*10)/10 ceil(max(xcf)*10)/10])
        displaytext = ['\leftarrow Max XCF: ' num2str(round(max(xcf),3)) ' at lag: ' num2str(lags(max(xcf)==xcf)*deltaT) 'min.'];
        text(lags(max(xcf)==xcf),max(xcf),displaytext);
        displaytext = ['\leftarrow Min XCF: ' num2str(round(min(xcf),3)) ' at lag: ' num2str(lags(min(xcf)==xcf)*deltaT) 'min.'];
        text(lags(min(xcf)==xcf),min(xcf),displaytext);
        
        %Increment subplot counter
        plotn=plotn+1;
    end
end

end

