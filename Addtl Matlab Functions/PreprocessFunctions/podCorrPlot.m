function [X,t] = podCorrPlot(X, t, settingsSet)
%{
Plot the correlation of all variables
%}

%Group the data including the date
plotDat = X;
plotDat.datetime = datenum(t);
%Temporarily disable annoying warnings and then make the correlation plot
warning('off','MATLAB:polyfit:RepeatedPointsOrRescale')
corrplot(plotDat,'testR','on');
warning('on','MATLAB:polyfit:RepeatedPointsOrRescale')

%Save image
if settingsSet.savePlots
    currentPod = settingsSet.podList.podName{settingsSet.loops.j};
    temppath = [currentPod '_CorrelationPlot'];
    temppath = fullfile(settingsSet.outpath,temppath);
    saveas(gcf,temppath,'jpeg');
    clear temppath
    close(gcf)
end
end %Function





