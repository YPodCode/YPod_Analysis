function func = williamsFit(a)
%From Williams, D. E.; Henshaw, G. S.; Bart, M.; Laing, G.; Wagner, J.; Naisbitt, S.; Salmond, J. A. Validation of low-cost ozone measurement instruments suitable for use in an air-quality monitoring network. Meas. Sci. Technol. 2013, 24, 065803, doi:10.1088/0957-0233/24/6/065803.


switch a
    case 1; func = @williamsGen;
    case 2; func = @williamsApply;
    case 3; func = @williamsReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlobj = williamsGen(Y,X,settingsSet)

%P = (Rg ? Rb)/(ab*Rb*Rg ? bg*(Rg ? Rb))
%Where Rb is the baseline resistance, Rg is the gas resistance, 
%and a* and b* are fitted constants


end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = williamsApply(X,mdlobj,~)

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function williamsReport(mdlobj,~)
try
    figure;
    subplot(2,2,1);plotResiduals(mdl);
    subplot(2,2,2);plotDiagnostics(mdl,'cookd');
    subplot(2,2,3);plotResiduals(mdl,'probability');
    subplot(2,2,4);plotResiduals(mdl,'lagged');
    plotSlice(mdl);
catch err
    disp('Error reporting this model');
end

end
%--------------------------------------------------------------------------
