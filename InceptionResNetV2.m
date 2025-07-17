clc; clear all; close all; warning off;

DatasetPath = 'E:\SER\RAVDESS\';  % location to your dataset, subfolders should be labels of corresponding image categories

% reading images from the image database folder
images = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numClasses = numel(categories(images.Labels));

    net = inceptionresnetv2;
    lgraph = layerGraph(net);
    
    newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'predictions',newFCLayer);
    
    newClassLayer = softmaxLayer('Name','new_softmax');
    lgraph = replaceLayer(lgraph,'predictions_softmax',newClassLayer);
    
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

[TrainImages, TestImages] = splitEachLabel(images, 0.8, 'randomized');

augTrainImages = augmentedImageDatastore([299 299 3], TrainImages);
augTestImages = augmentedImageDatastore([299 299 3], TestImages);

miniBatchSize = 8;
numObservationsTrain = numel(augTrainImages.Files); % Total Training Observations
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize); % Validation Frequency

% training options
options = trainingOptions('adam', 'MiniBatchSize', miniBatchSize, 'MaxEpochs', 20, 'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', 'ValidationData', augTestImages, 'ValidationFrequency', numIterationsPerEpoch, 'Verbose', false, 'Plots', 'training-progress', 'OutputNetwork', 'best-validation');

% training the InceptionResnNet-V2
netTransfer = trainNetwork(augTrainImages, lgraph, options);

% Classifying images
YPred = classify(netTransfer, augTestImages);
YValidation = TestImages.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

% plotting Confusion Matrix
figure;
plotconfusion(YValidation, YPred)
title('Confusion Matrix Using InceptionResNet-V2')

% Optional: Save the training curve and the fine-tuned model
% currentfig = findall(groot,'Type','Figure'); savefig(currentfig,'InceptionResNet-V2 Learning Curve.fig')
% Folder = 'D:\Trained Networks';
% File = 'InceptionRes.mat';
% save(fullfile(Folder,File),'netTransfer');
