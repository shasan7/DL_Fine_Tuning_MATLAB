clc; clear all; close all; warning off;

DatasetPath = 'E:\SER\RAVDESS\'; % location to your dataset, subfolders should be labels of corresponding image categories

% reading images from the image database folder
images = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numClasses = numel(categories(images.Labels));

    net=darknet53;
    lgraph = layerGraph(net);
    
    newLearnableLayer = fullyConnectedLayer(numClasses, 'Name','new_fc', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'conv53',newLearnableLayer);
    
    newsoftmaxLayer = softmaxLayer('Name','new_softmax');
    lgraph = replaceLayer(lgraph,'softmax',newsoftmaxLayer);
    
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'output',newClassLayer);

[TrainImages, TestImages] = splitEachLabel(images, 0.8, 'randomized');

augTrainImages = augmentedImageDatastore([256 256 3], TrainImages);
augTestImages = augmentedImageDatastore([256 256 3], TestImages);

miniBatchSize = 8;
numObservationsTrain = numel(augTrainImages.Files); % Total Training Observations
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize); % Validation Frequency

% training options
options = trainingOptions('adam', 'MiniBatchSize', miniBatchSize, 'MaxEpochs', 20, 'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', 'ValidationData', augTestImages, 'ValidationFrequency', numIterationsPerEpoch, 'Verbose', false, 'Plots', 'training-progress', 'OutputNetwork', 'best-validation');

% training the DarkNet53
netTransfer = trainNetwork(augTrainImages, layers, options);

% Classifying images
YPred = classify(netTransfer, augTestImages);
YValidation = TestImages.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

% plotting Confusion Matrix
figure;
plotconfusion(YValidation, YPred)
title('Confusion Matrix Using DarkNet53')

% Optional: Save the training curve and the fine-tuned model
% currentfig = findall(groot,'Type','Figure'); savefig(currentfig,'DarkNet53 Learning Curve.fig')
% Folder = 'D:\Trained Networks';
% File = 'Dark53.mat';
% save(fullfile(Folder,File),'netTransfer');