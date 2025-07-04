clc; clear all; close all; warning off;

DatasetPath = 'E:\SER\RAVDESS\';  % location to your dataset, subfolders should be labels of corresponding image categories

% reading images from the image database folder
images = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numClasses = numel(categories(images.Labels));

net = resnet18;
lgraph = layerGraph(net);

newConnectedLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph, 'fc1000', newConnectedLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

[TrainImages, TestImages] = splitEachLabel(images, 0.8, 'randomized');

augTrainImages = augmentedImageDatastore([224 224 3], TrainImages);
augTestImages = augmentedImageDatastore([224 224 3], TestImages);

miniBatchSize = 8;
numObservationsTrain = numel(augTrainImages.Files); % Total Training Observations
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize); % Validation Frequency

% training options
options = trainingOptions('adam', 'MiniBatchSize', miniBatchSize, 'MaxEpochs', 20, 'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', 'ValidationData', augTestImages, 'ValidationFrequency', numIterationsPerEpoch, 'Verbose', false, 'Plots', 'training-progress', 'OutputNetwork', 'best-validation');

% training the ResNet18
netTransfer = trainNetwork(augTrainImages, layers, options);

% Classifying images
YPred = classify(netTransfer, augTestImages);
YValidation = TestImages.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

% plotting Confusion Matrix
figure;
plotconfusion(YValidation, YPred)
title('Confusion Matrix Using ResNet18');

% Optional: Save the training curve and the fine-tuned model
% currentfig = findall(groot,'Type','Figure'); savefig(currentfig,'ResNet18 Learning Curve.fig')
% Folder = 'D:\Trained Networks';
% File = 'Res18.mat';
% save(fullfile(Folder,File),'netTransfer');