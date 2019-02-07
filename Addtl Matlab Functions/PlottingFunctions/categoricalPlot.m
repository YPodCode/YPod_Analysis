function categoricalPlot(~,~,Y,Y_hat,valList,~,settingsSet)
%Plots the validation and calibration estimates for each fold


%Apply the same categories to Y_hat as in the original dataset (m=nRegs,k=nVal,kk=nFold)
Y = table2array(Y);
ycats = categories(Y);
nrefcats = size(ycats,1);
figure
for m=1:settingsSet.loops.m
    for k=1:settingsSet.loops.k
        for kk=1:settingsSet.loops.kk
            Y_hat_cal = Y_hat.cal{m,k,kk};
            Y_hat_val = Y_hat.val{m,k,kk};
            if kk==1
                yplot = [Y(valList{k}~=kk,:); Y(valList{k}==kk,:)];
                yhat = [Y_hat_cal; Y_hat_val];
            else
                yhat = [yhat; Y_hat_cal; Y_hat_val];
                yplot = [yplot; Y(valList{k}~=kk,:); Y(valList{k}==kk,:)];
            end
            
        end
        if ~iscategorical(yhat)
            yhat = categorical(yhat,ycats);
        end
        %Calculate normalized confusion matrix
        %cm=confusionmat(yplot,yhat);
        predcats = categories(yhat);
        nhatcats = size(predcats,1);
        cm = zeros(nrefcats, nhatcats);
        countlist = ones(size(yplot,1),1);
        for zz = 1:nrefcats
            for xx = 1:nhatcats
                cm(zz,xx) = sum(countlist(yplot==ycats{zz} & yhat==predcats{xx}));
            end
        end
        cumcm = sum(cm,2);
        for z = 1:size(cm,1)
            cm(z,:) = cm(z,:)/cumcm(z);
        end
        subplot(settingsSet.loops.m, settingsSet.loops.k, (m-1)*settingsSet.loops.k+k)
        imagesc(cm); colormap('bone');
        colorbar('Ticks',[0,0.25,0.5,0.75,1]);yticks(1:length(ycats));xticks(1:length(categories(yhat)));grid on
        title(['Reg: ' settingsSet.modelList{m} ', Val: ' settingsSet.valList{k}])
        ylabel('True Category');xlabel('Prediction')
        hold on
        clear yhat yplot
    end
end

end

