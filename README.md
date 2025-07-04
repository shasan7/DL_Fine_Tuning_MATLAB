Fine-tune pretrained image classification models in MATLAB. The dataset location is supposed to contain subfolders with folder names as labels of corresponding image categories.
Due to different final layers structure for different networks, i had to provide separate codes for each of them.
In my MATLAB version (R2023a), I can only save the model with "best-validation" where the "Metric' is automatically assumed validation loss. If you version is compatible, try "best-validation" with "accuracy" Metric in the trainingOptions for higher performance.
If you require the cross-validation version of these codes, you'll find them in the "Cross Validation" folder.
