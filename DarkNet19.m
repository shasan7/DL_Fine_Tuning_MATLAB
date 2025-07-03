clc; clear all; close all; warning off;

DatasetPath = 'E:\SER\RAVDESS\';  % location to your dataset, subfolders should be labels of corresponding image category

% reading images from the image database folder
images = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numClasses = numel(categories(images.Labels));

    net=darknet19;
    layersTransfer = net.Layers(1:end-4);

    layers = [
        layersTransfer
        globalAveragePooling2dLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];

[TrainImages, TestImages] = splitEachLabel(images, 0.8, 'randomized');

augTrainImages = augmentedImageDatastore([256 256 3], TrainImages);
augTestImages = augmentedImageDatastore([256 256 3], TestImages);

miniBatchSize = 8;
numObservationsTrain = numel(augTrainImages.Files); % Total Training Observations
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize); % Validation Frequency

%training options
options = trainingOptions('adam', 'MiniBatchSize', miniBatchSize, 'MaxEpochs', 20, 'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', 'ValidationData', augTestImages, 'ValidationFrequency', numIterationsPerEpoch, 'Verbose', false, 'Plots', 'training-progress', 'OutputNetwork', 'best-validation');

% training the DarkNet19
netTransfer = trainNetwork(augTrainImages, layers, options);

%Classifying images
YPred = classify(netTransfer, augTestImages);
YValidation = TestImages.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

%plotting Confusion Matrix
figure;
plotconfusion(YValidation, YPred)
title('Confusion Matrix Using DarkNet19')

% Optional: Save the training curve and the fine-tuned model
% currentfig = findall(groot,'Type','Figure'); savefig(currentfig,'DarkNet19 Learning Curve.fig') %
% Folder = 'D:\Trained Networks';
% File = 'Dark19.mat';
% save(fullfile(Folder,File),'netTransfer');