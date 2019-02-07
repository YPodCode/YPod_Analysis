function func = a_genericModel(a)
%This main function returns a handle for the relative
%fitting/application/reporting functions
switch a
    %NOTE: You should change the name of the subfunctions as well as the
    %main function, or you may have conflicts with other models.
    %E.g.: Change "genericGenerate" to "lineXGenerate", where "lineX" is
    %your new model name
    case 1; func = @genericGenerate;
    case 2; func = @genericApply;
    case 3; func = @genericReport;
end
end

%-------------Generate/fit the model-------------
function fittedMdl = genericGenerate(Y,X,settingsSet)
%Accepts X and Y as tables of dimensions nx1 and nxp respectively
fittedMdl = fit(table2array(X),table2array(Y),'poly1');

end

%-------------Apply the model to new data-------------
function y_hat = genericApply(X,fittedMdl,settingsSet)
%Accepts X as a table of dimensions mxp and fittedMdl in the same format as
%is output by the fitting function above
y_hat = fittedMdl(table2array(X));
end

%-------------Report relevant stats (coefficients, etc) about the model-------------
function genericReport(fittedMdl,mdlStats,settingsSet)
fittedMdl
end