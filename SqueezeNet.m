clc; clear all; close all; warning off;

DatasetPath = 'E:\SER\RAVDESS\';  % location to your dataset, subfolders should be labels of corresponding image categories

% reading images from the image database folder
images = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numClasses = numel(categories(images.Labels));

    sqz = squeezenet;
    lgraphSqz = layerGraph(sqz);

    tmpLayer = lgraphSqz.Layers(end-5);
    newDropoutLayer = dropoutLayer(0.6,'Name','new_dropout');
    lgraphSqz = replaceLayer(lgraphSqz,tmpLayer.Name,newDropoutLayer);

    tmpLayer = lgraphSqz.Layers(end-4);
    newLearnableLayer = convolution2dLayer(1,numClasses, 'Name','new_conv', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10);
    lgraphSqz = replaceLayer(lgraphSqz,tmpLayer.Name,newLearnableLayer);

    tmpLayer = lgraphSqz.Layers(end);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraphSqz = replaceLayer(lgraphSqz,tmpLayer.Name,newClassLayer);

[TrainImages, TestImages] = splitEachLabel(images, 0.8, 'randomized');

augTrainImages = augmentedImageDatastore([227 227 3], TrainImages);
augTestImages = augmentedImageDatastore([227 227 3], TestImages);

miniBatchSize = 8;
numObservationsTrain = numel(augTrainImages.Files); % Total Training Observations
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize); % Validation Frequency

% training options
options = trainingOptions('adam', 'MiniBatchSize', miniBatchSize, 'MaxEpochs', 20, 'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', 'ValidationData', augTestImages, 'ValidationFrequency', numIterationsPerEpoch, 'Verbose', false, 'Plots', 'training-progress', 'OutputNetwork', 'best-validation');

% training the SqueezeNet
netTransfer = trainNetwork(augTrainImages, layers, options);

% Classifying images
YPred = classify(netTransfer, augTestImages);
YValidation = TestImages.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

% plotting Confusion Matrix
figure;
plotconfusion(YValidation, YPred)
title('Confusion Matrix Using SqueezeNet')

% Optional: Save the training curve and the fine-tuned model
% currentfig = findall(groot,'Type','Figure'); savefig(currentfig,'SqueezeNet Learning Curve.fig')
% Folder = 'D:\Trained Networks';
% File = 'Squeeze.mat';
% save(fullfile(Folder,File),'netTransfer');