%%
clear all;
clc;
close all;
%% Load dataset in a image store and then split it into train and test image store
close all; clear all;
rootFolder1 = 'AR_Database\Unoccluded_TrainingSet';
categories = {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7','P8', 'P9','P10',...
    'P11','P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19','P20',...
    'P21','P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29','P30','P31'};

TrainFace = imageDatastore(fullfile(rootFolder1, categories), 'LabelSource',...
    'foldernames');

% Chose one at a time. Either rootFolder2 or rootFolder3 and generate
% TestFace1 or TestFace2. For glass images use first one while for scarf
% use second one.

rootFolder2 = 'AR_Database\Glass_TestSet';
% rootFolder3 = 'AR_Database\Scarf_TestSet';

TestFace = imageDatastore(fullfile(rootFolder2, categories), 'LabelSource',...
   'foldernames');
% TestFace = imageDatastore(fullfile(rootFolder3, categories), 'LabelSource',...
%    'foldernames');

% shuffle the data in image stores
TrainFace = shuffle(TrainFace);
TestFace = shuffle(TestFace);

Y =  TrainFace.Labels; %Train set target labels
Y1 = TestFace.Labels; % test set target labels

L  = length(Y); % Number of samples in training set
L1 = length(Y1); % Number of samples in the test set


%% implementation of random ternary measurement matrix and per-column CS
target_size = [500, 470];  % size of the image sensor
nr = target_size(1);   % Number of rows
nc = target_size(2);   % Number of columns

Row = zeros(nr,nc);    % store the indices of pixel to be made to zero

Xs = zeros(L,nc);   % Training samples: Each samples consists of number of features equal to number of columns 
XsT = zeros(L1,nc); % Test samples

C = [0.5]; % percentage of pixels set to zero

K = 10;  % running the simulation 10 times
Accuracy = zeros(K,1);  % storing the accuracy for each run
ref= 127; % reference value for comparator of sigma-delta university

for p=1:K
    
    Modu = randsrc(nr,nc,[-1 1]); % Matrix for Random modulation by +1 and -1

    for i=1:nr
        Row(i,:) = randperm(nc); % Each column has values(indices for selection) 
        %in the range of number of rows but randomly permuted
    end

    
    c1 = ceil(c*nr);   %Number of pixels to be set to zero
    for i=1:L
        img = imresize(rgb2gray(readimage(TrainFace,i)), target_size);  % read the train image

        img1 = double(img); % change the dataformat for further operations        
        img1 = img1.*Modu;  % multiply each pixel with + 1 or -1.
        
        Zero_ind = zeros(c1,nc); % this hold the indices used to make pixels zero in each column
        for k=1:nc
            sel=Row(1:c1,k);
            Zero_ind(:,k) = sel;
            img1(sel,k)= 0; % selected pixels are made to zeros
        end

        % Following is the function for Sigma-Delta ADC implementation
        % feeding Zero_ind matrix to sigma delta ADC is to disbale the
        % processing the when the pixel is set to zero
        
        m = sigma_delta_UD_Counter_col_not_selected_skipped(img1,ref,Zero_ind);   % This returns the count value of each image which constitute the feature vectore of support vector machine
        Xs(i,:) = m;  % Count vaue stored in training matrix


    end
   % Similar operation for test set
    for i=1:L1
        img = imresize(rgb2gray(readimage(TestFace,i)), target_size);

        img1 = double(img);

        img1 = img1.*Modu; 
        Zero_ind = zeros(c1,nc);
        for k=1:nc
            sel=Row(1:c1,k);
            Zero_ind(:,k) = sel;
            img1(sel,k)= 0; % selected pixels are made to zeros
        end

        m = sigma_delta_UD_Counter_col_not_selected_skipped(img1,ref,Zero_ind); 

        XsT(i,:) = m;


     end

     t = templateSVM('KernelFunction','Linear', 'Standardize',true); % create SVM template keeping the data standardization enabled
     mdl = fitcecoc(Xs,Y,'Coding','onevsall','Learners',t);  % train the classifier
     pred = predict(mdl,XsT);    % predict the Test set
     acc = sum(pred==Y1)/L1      % calculate the accuracy
     Accuracy(p,1) = acc;        % store the evaluated accuracy in the matrix
    
end
Avg_Accuracy = sum(Accuracy)/K   % Average the accuracy
