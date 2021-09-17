dm.splitAsSourceTargetStreams2(500, 1/4);

crSource = {};
crTarget             = ones(dm.nMinibatches, 1);
trainTime            = ones(dm.nMinibatches, 1);
testTime             = ones(dm.nMinibatches, 1);
CMDLoss      = ones(dm.nMinibatches, 1);
classificationLoss   = ones(dm.nMinibatches, 1);
nodeEvolution        = zeros(dm.nMinibatches, 1);
discriminativeLoss   = ones(dm.nMinibatches, 1);
generativeLossTarget = ones(dm.nMinibatches, 1);
sgSourceDisc = {};
gmmTargetBatch      = ones(dm.nMinibatches, 1);
gmmSourceBatch      = ones(dm.nMinibatches, 1);

finalCRtarget = [];
finalCRsource = [];
finalTime = [];
finalnodes = [];
trueClassesSet    = [];
predictClassesSet = [];
for iRun = 1: nRun
    fprintf('No of run: %d of %d\n', iRun ,nRun);
    nn = NeuralNetwork([dm.nFeatures 1 dm.nClasses]);
    ae = DenoisingAutoEncoder([nn.layers(1) nn.layers(2) nn.layers(1)]);
    
    % I am building the greedyLayerBias
    x = dm.getXs2(1,1);
    ae.greddyLayerWiseTrain(x(1, :), 1, 0.1);
    % I am building the greedyLayerBias
    for iSource = 1:dm.nSource
        sgSourceDisc{iSource} = singleGaussian();
    end
    originalLearningRate = ae.learningRate;
    for i = 1 : dm.nMinibatches
        Xt = dm.getXt2(1,i);
        yt = dm.getYt2(1,i);
        
        %% Evaluation ~ Test Target
        tic
        nn.test(Xt, yt);
        crTarget(i) = nn.classificationRate;
        classificationLoss(i) = nn.lossValue;
        testTime(i) = toc;
        discriminativeLoss(i) = nn.lossValue;
        trueClassesSet = [trueClassesSet; nn.trueClasses];
        predictClassesSet = [predictClassesSet; nn.outputedClasses];
        
        
        ae.test(Xt);
        generativeLossTarget(i) = ae.lossValue;
        
        
        tic
        for epoch = 1 : epochs
            %% Discriminative phase on Source        
            for iSource = 1: dm.nSource
                Xs = dm.getXs2(iSource, i);
                ys = dm.getYs2(iSource, i);
                nn.test(Xs(max(Xs, [], 2) ~= 0, :), ys(max(Xs, [], 2) ~= 0, :));
                crSource{iSource}(i) = nn.classificationRate;
                for j = 1 : size(Xs, 1)
                    x = Xs(j, :);
                    ae.forwardpass(x);
                    ae.greddyLayerWiseTrain(x, 1, 0.1);
                end
                for j = 1 : numel(ae.layers)-2
                    nn.weight{j} = ae.weight{j};
                    nn.bias{j}   = ae.bias{j};
                end
                for j = 1 : size(Xs, 1)
                    x = Xs(j, :);
                    y = ys(j, :);
                    if max(y) == 0
                        continue
                    end
                    sgSourceDisc{iSource}.runGM(x);                
                    nn.forwardpass(x);
                    if epoch == 1
                        nn.widthAdaptationStepwise(y, sgSourceDisc{iSource});
                    else
                        nn.nSamplesFeed = nn.nSamplesFeed - 1;
                        nn.nSamplesLayer(lastHiddenLayerNo) = nn.nSamplesLayer(lastHiddenLayerNo) - 1;
                        nn.widthAdaptationStepwise(y, sgSourceDisc{iSource});
                    end
                    if nn.growable
                        nn.grow(2);
                        ae.grow(2);                       
                    elseif nn.prunable(1) ~= 0
                        nodeToPrune = nn.prunable(1);
                        ae.prune(2,nodeToPrune);
                        nn.prune(2,nodeToPrune);                                     
                    end
                    nn.train(x, y);
                end
            end
            for j = 1 : numel(nn.layers)-2
                ae.weight{j} = nn.weight{j};
                ae.bias{j}   = nn.bias{j};
            end
            for j = 1 : size(Xt, 1)
                x = Xt(j, :);
                y = x;
                lastHiddenLayerNo = numel(nn.layers) - 1;
                ae.greddyLayerWiseTrain(x, 1, 0.1);
            end
            for j = 1 : numel(ae.layers)-2
                nn.weight{j} = ae.weight{j};
                nn.bias{j}   = ae.bias{j};
            end
            
            nn.SimilarityTrain(Xt);              
            for j = 1 : numel(ae.layers)-2
                nn.weight{j} = ae.weight{j};
                nn.bias{j}   = ae.bias{j};
            end
% %             CMD 
            for iSource = 1: dm.nSource
                Xs = dm.getXs2(iSource, i);
                CMDLoss = ae.updateWeightsByCMD(Xs, Xt, 2);
            end
            for j = 1 : numel(ae.layers)-2
                nn.weight{j} = ae.weight{j};
                nn.bias{j}   = ae.bias{j};
            end
        end
        trainTime(i) = toc;
       
        
        
        %% Print metrics
        nodeEvolution(i, :) = nn.layers(2 : end - 1);
        if mod(i,round(0.3*dm.nMinibatches)) == 0 || i == 2 || i == dm.nMinibatches
            fprintf('Minibatch: %d/%d\n', i, dm.nMinibatches);
            fprintf('Total of samples: %d Source | %d Target\n', size(Xs,1), size(Xt,1));
            fprintf('Max Mean Min Now Accu Training time: %f %f %f %f %f\n', max(trainTime(1:i)), mean(trainTime(1:i)), min(trainTime(1:i)), trainTime(i), sum(trainTime(1:i)));
            fprintf('Max Mean Min Now Accu Testing time: %f %f %f %f %f\n', max(testTime(1:i)), mean(testTime(1:i)), min(testTime(1:i)), testTime(i), sum(testTime(1:i)));
            fprintf('Max Mean Min Now CR: %f%% %f%% %f%% %f%%\n', max(crTarget(2:i)) * 100., mean(crTarget(2:i)) * 100., min(crTarget(2:i)) * 100., crTarget(i) * 100.);
            fprintf('Max Mean Min Now Classification Loss: %f %f %f %f\n', max(classificationLoss(2:i)), mean(classificationLoss(2:i)), min(classificationLoss(2:i)), classificationLoss(i));
            fprintf('Max Mean Min Now Nodes: %d %f %d %d\n', max(nodeEvolution(2:i)), mean(nodeEvolution(2:i)), min(nodeEvolution(2:i)), nodeEvolution(i));
            fprintf('Max Mean Min Now Source 1 CR: %f%% %f%% %f%% %f%%\n', max(crSource{1}(2:i)) * 100., mean(crSource{1}(2:i)) * 100., min(crSource{1}(2:i)) * 100., crSource{1}(i) * 100.);
%             fprintf('Max Mean Min Now Source 2 CR: %f%% %f%% %f%% %f%%\n', max(crSource{2}(2:i)) * 100., mean(crSource{2}(2:i)) * 100., min(crSource{2}(2:i)) * 100., crSource{2}(i) * 100.);
%             fprintf('Max Mean Min Now Source 3 CR: %f%% %f%% %f%% %f%%\n', max(crSource{3}(2:i)) * 100., mean(crSource{3}(2:i)) * 100., min(crSource{3}(2:i)) * 100., crSource{3}(i) * 100.);
            fprintf('Network structure: %s (Discriminative) | %s (Generative)\n', num2str(nn.layers(:).'), num2str(ae.layers(:).'));       
            fprintf('\n');
        end
    end
    
end
[fMeasure,gMean,recall,precision,error] = performanceMeasure(trueClassesSet, predictClassesSet, dm.nClasses);
fprintf('Precision: %s\n recall: %s', num2str(precision), num2str(recall));
%% Plots

subplot(2,1,1);
plot(nodeEvolution)
ylim([0 max(nodeEvolution, [], 'all') * 1.1]);
xlim([1 size(nodeEvolution, 1)]);
title('Number of nodes');
xlabel('Minibatches');
hold on;
subplot(2,1,2);
plot(crTarget)
ylim([0  max(crTarget) * 1.1]);
xlim([1 size(crTarget, 1)]);
title('Classification Rate');
xlabel('Minibatches');
hold off;

%% Performance measure
% This function is developed from Gregory Ditzler. It functions to measure
% the precision and recall
% https://github.com/gditzler/IncrementalLearning/blob/master/src/stats.m
function [fMeasure,gMean,recall,precision,error] = performanceMeasure(trueClass, rawOutput, nClass)
label           = index2vector(trueClass, nClass);
predictedLabel  = index2vector(rawOutput, nClass);

recall      = calculate_recall(label, predictedLabel, nClass);
error       = 1 - sum(diag(predictedLabel'*label))/sum(sum(predictedLabel'*label));
precision   = calculate_precision(label, predictedLabel, nClass);
gMean       = calculate_g_mean(recall, nClass);
fMeasure    = calculate_f_measure(label, predictedLabel, nClass);


    function gMean = calculate_g_mean(recall, nClass)
        gMean = (prod(recall))^(1/nClass);
    end

    function fMeasure = calculate_f_measure(label, predictedLabel, nClass)
        fMeasure = zeros(1, nClass);
        for iClass = 1:nClass
            fMeasure(iClass) = 2*label(:, iClass)'*predictedLabel(:, iClass)/(sum(predictedLabel(:, iClass)) + sum(label(:, iClass)));
        end
        fMeasure(isnan(fMeasure)) = 1;
    end

    function precision = calculate_precision(label, predictedLabel, nClass)
        precision = zeros(1, nClass);
        for iClass = 1:nClass
            precision(iClass) = label(:, iClass)'*predictedLabel(:, iClass)/sum(predictedLabel(:, iClass));
        end
        precision(isnan(precision)) = 1;
    end

    function recall = calculate_recall(label, predictedLabel, nClass)
        recall = zeros(1, nClass);
        for iClass = 1:nClass
            recall(iClass) = label(:, iClass)'*predictedLabel(:, iClass)/sum(label(:, iClass));
        end
        recall(isnan(recall)) = 1;
    end

    function output = index2vector(input, nClass)
        output = zeros(numel(input), nClass);
        for iData = 1:numel(input)
            output(iData, input(iData)) = 1;
        end
    end
end
