%%% PRINT VARIABLES %%%
print_training_set = 0; % Print random images of the training set
print_test_set = 0; % Print random images of the predicted set
print_conf_matr = 1; % Print the confusion matrix

%%% CLASSIFICATION VERSION %%%
% classification_version = "matlab"; % Uncomment to use the matlab version of the svm classifier
classification_version = "liblinear"; % Uncomment to use the liblinear version of the svm classifier

%%% NETWORK SELECTION %%%
network = "alexnet"; % Uncomment to use alexnet
% network = "resnet"; % Uncomment to use resnet18
% network = "vgg"; % Uncomment to use vgg16

%%% LOAD DATA %%%
imdsTrain = imageDatastore('TrainingSet','IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest = imageDatastore('ValidationSet','IncludeSubfolders',true,'LabelSource','foldernames');
numTrainImages = numel(imdsTrain.Labels);

%%% PRINT SOME IMAGES OF THE TRAINING SET %%%
if print_training_set == 1
    idx = randperm(numTrainImages,16);
    figure
    for i = 1:16
        subplot(4,4,i)
        I = readimage(imdsTrain,idx(i));
        imshow(I)
    end
end

%%% LOAD PRETRAINED NETWORK %%%
tic;
if network == "alexnet"
    net = alexnet;
elseif network == "resnet"
    net = resnet18;
else
    net = vgg16;
end

%%% RESIZE THE IMAGES %%%
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%%% SELECT ACTIVATION LAYER FOR FEATURE EXTRACTION AND EXTRACT FEATURES %%%
if network == "resnet"
    layer = 'pool5';
else
    layer = 'fc7';
end
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%%% CLASSIFICATION MATLAB VERSION %%%
if classification_version == "matlab"
    classifier = fitcecoc(featuresTrain,YTrain);
    YPred = predict(classifier,featuresTest);
end

%%% CLASSIFICATION LIBLINEAR VERSION %%%
if classification_version == "liblinear"
    YTrain = double(YTrain(:,1)) -1;
    YTest = double(YTest(:,1)) -1;
    featuresTrain = sparse(double(featuresTrain));
    featuresTest = sparse(double(featuresTest));
    model = train(YTrain, featuresTrain, '-s 2');
    YPred = predict(YTest, featuresTest, model);
end

%%% SHOW IMAGES %%%
if print_test_set == 1
    idx = randi([1 3077],1,12);
    % idx = [];
    figure
    for i = 1:numel(idx)
        subplot(3,4,i)
        I = readimage(imdsTest,idx(i));
        label = YPred(idx(i));
        corr = YTest(idx(i));
        imshow(I)
        title("#: "+idx(i)+" / Predicted: "+label+" / Correct: "+corr)
    end
end

%%% CONFUSION MATRIX %%%
if print_conf_matr == 1
    cm = confusionmat(YTest,YPred);
    labels = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    figure
    plotConfMat(cm,labels)
end

%%% RESULTS %%%
time = toc;
diff = numel(find(YPred~=YTest));
[M,N] = size(YPred);
tp = M-diff;
accuracy = round(mean(YPred == YTest)*100,2);
disp('Accuracy: '+string(accuracy)+"% - Time Elapsed: "+time+" s - True Positive vs Total: "+tp+"/"+M);