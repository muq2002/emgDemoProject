%% Pattern Recognition of Electromyography Signal  
%% Objective 
% The aim of this project is to apply pattern recognition steps to several EMG signals, which are normal, myopathy, and neuropathy signals by using Matlab.
%%
% 
% <<fig1.png>>
% 
%% 1) Load and show electromyography (EMG) Signals
normalSignal = load('data/emg_healthy.txt');
myopathySignal = load('data/emg_myopathy.txt');
neuropathySignal = load('data/emg_neuropathy.txt');

% Normal Signal
figure(1);

subplot(2,2,[1 3]);
plot(normalSignal(:,1), normalSignal(:,2));
title('Normal Signal'); xlabel('Time'); ylabel('Amplitude');

subplot(2,2,2);
plot(normalSignal(1:6000,1), normalSignal(1:6000,2), '-r');
title('Zooming on Normal Signal');  xlabel('Time'); ylabel('Amplitude');

subplot(2,2,4);
plot(normalSignal(1000:1400,1), normalSignal(1000:1400,2), '-r');
title('Zooming Normal Signal'); xlabel('Time'); ylabel('Amplitude');
suptitle('Normal Signal');

% Myopathy Signal
figure(2);

subplot(2,2,[1 3]);
plot(myopathySignal(:,1), myopathySignal(:,2));
title('Myopathy Signal'); xlabel('Time'); ylabel('Amplitude');

subplot(2,2,2);
plot(myopathySignal(1:6000,1), myopathySignal(1:6000,2), '-r');
title('Zooming on Myopathy Signal');  xlabel('Time'); ylabel('Amplitude');

subplot(2,2,4);
plot(myopathySignal(1000:1400,1), myopathySignal(1000:1400,2), '-r');
title('Zooming Myopathy Signal'); xlabel('Time'); ylabel('Amplitude');
suptitle('Myopahty Signal');


% Neuropathy Signal
figure(3);

subplot(2,2,[1 3]);
plot(neuropathySignal(:,1), neuropathySignal(:,2));
title('Neuropathy Signal'); xlabel('Time'); ylabel('Amplitude');

subplot(2,2,2);
plot(neuropathySignal(1:6000,1), neuropathySignal(1:6000,2), '-r');
title('Zooming on Neuropathy Signal');  xlabel('Time'); ylabel('Amplitude');

subplot(2,2,4);
plot(neuropathySignal(1000:1400,1), neuropathySignal(1000:1400,2), '-r');
title('Zooming Neuropathy Signal'); xlabel('Time'); ylabel('Amplitude');
suptitle('Neuropathy Signal');
%% 2) Determine the specifications of electromyography (EMG) Signals

% Normal Signal
samples_normal = length(normalSignal);
size_normal = normalSignal(end,1);
Fs_normal = samples_normal / size_normal
Ts_normal = 1/Fs_normal


% Myopathy Signal
samples_myopathy = length(myopathySignal);
size_myopathy = myopathySignal(end,1);
Fs_myopathy = ceil(samples_myopathy / size_myopathy)
Ts_myopathy = 1/Fs_myopathy


% Neuropathy Signal
samples_neuropathy = length(neuropathySignal);
size_neuropathy = neuropathySignal(end,1);
Fs_neuropathy = ceil(samples_neuropathy / size_neuropathy)
Ts_neuropathy = 1/Fs_neuropathy


%% 3) Segmentation of electromyography (EMG) Signals

window_Size = 0.25; % second

% Normal Signal
samples_window_normal =  floor(window_Size/Ts_normal)
number_of_windows_normal = floor(size_normal/window_Size)
segments_normal = segmentationSignal(normalSignal, samples_normal,samples_window_normal,number_of_windows_normal);


% Myopathy Signal
samples_window_myopathy = floor(window_Size/Ts_myopathy)
number_of_windows_myopathy = floor(size_myopathy/window_Size)
segments_myopathy = segmentationSignal(myopathySignal, samples_myopathy,samples_window_myopathy,number_of_windows_myopathy);

% Neuropathy Signal
samples_window_neuropathy = floor(window_Size/Ts_neuropathy)
number_of_windows_neuropathy = floor(size_neuropathy/window_Size)
segments_neuropathy = segmentationSignal(neuropathySignal, samples_neuropathy,samples_window_neuropathy,number_of_windows_neuropathy);

%% 4) Features extraction of electromyography (EMG) Signals

% * Root Mean Square (RMS)
% * Mean Absolute Value (MAV)
% * Integrated EMG (IEMG)
% * Variance of EMG (VAR)


% Normal Signal
feature_extraction_data_normal = ones(number_of_windows_normal,4);
for i=1:number_of_windows_normal
   signal_RMS =  rms(segments_normal{i,1}(:,2));
   signal_VAR = var(segments_normal{i,1}(:,2));
   signal_Mean =  mean(abs(segments_normal{i,1}(:,2)));
   siganl_IEMG =  sum(abs(segments_normal{i,1}(:,2)));
   
   feature_extraction_data_normal(i,:) = [signal_RMS signal_VAR signal_Mean siganl_IEMG];
end
head_table_of_normal = feature_extraction_data_normal(1:5,:) % Show head of Data
xlswrite('feature_extraction_normal.xlsx',feature_extraction_data_normal);

% Myopathy Signal 
feature_extraction_data_myopathy = ones(number_of_windows_myopathy,4);
for i=1:number_of_windows_myopathy
   signal_RMS =  rms(segments_myopathy{i,1}(:,2));
   signal_VAR = var(segments_myopathy{i,1}(:,2));
   signal_Mean =  mean(abs(segments_myopathy{i,1}(:,2)));
   siganl_IEMG =  sum(abs(segments_myopathy{i,1}(:,2)));
   
   feature_extraction_data_myopathy(i,:) = [signal_RMS signal_VAR signal_Mean siganl_IEMG];
end
head_table_of_myopathy = feature_extraction_data_myopathy(1:5,:) % Show head of Data
xlswrite('feature_extraction_myopathy.xlsx',feature_extraction_data_myopathy);

% Neuropathy Signal 
feature_extraction_data_neuropathy = ones(number_of_windows_neuropathy,4);
for i=1:number_of_windows_neuropathy
   signal_RMS =  rms(segments_neuropathy{i,1}(:,2));
   signal_VAR = var(segments_neuropathy{i,1}(:,2));
   signal_Mean =  mean(abs(segments_neuropathy{i,1}(:,2)));
   siganl_IEMG =  sum(abs(segments_neuropathy{i,1}(:,2)));
   
   feature_extraction_data_neuropathy(i,:) = [signal_RMS signal_VAR signal_Mean siganl_IEMG];
end
head_table_of_neuropathy = feature_extraction_data_neuropathy(1:5,:) % Show head of Data
xlswrite('feature_extraction_neuropathy.xlsx',feature_extraction_data_neuropathy);

%% 5) Classification of electromyography (EMG) Signals
% Classification In Machine Learning. 
% Classification is a process of categorizing a given set of data into classes, It can be performed on both structured or unstructured data. The process starts with predicting the class of given data points. The classes are often referred to as target, label or categories.
%% A. Correlation of Data
% What is correlation? 
% Correlation is a statistical measure that expresses the extent to which two variables are linearly related (meaning they change together at a constant rate). It's a common tool for describing simple relationships without making a statement about cause and effect.
figure(4);

% 1) RMS & IEMG
subplot(3,1,1);

plot(feature_extraction_data_normal(:,1),feature_extraction_data_normal(:,4),'o','Color',[0, 0.4470, 0.7410],'LineWidth',2); hold on; % Normal Signal
plot(feature_extraction_data_myopathy(:,1),feature_extraction_data_myopathy(:,4),'^','Color',[0.4660, 0.6740, 0.1880],'LineWidth',2); hold on;% Mypopathy
plot(feature_extraction_data_neuropathy(:,1),feature_extraction_data_neuropathy(:,4),'s','Color',[0.6350, 0.0780, 0.1840],'LineWidth',2); hold on; % Neuropathy
xlabel('RMS') ; ylabel('IEMG');
legend('Normal','Myopathy','Neuropathy');

RMS_Data = [feature_extraction_data_normal(:,1).' feature_extraction_data_myopathy(:,1).' feature_extraction_data_neuropathy(:,1).'];
IEMG_Data = [feature_extraction_data_normal(:,4).' feature_extraction_data_myopathy(:,4).' feature_extraction_data_neuropathy(:,4).'];
RMS_IEMG_Correlation  = corrcoef(RMS_Data,IEMG_Data)

% 2) RMS & MEAN_ABSOLUTE
subplot(3,1,2);

plot(feature_extraction_data_normal(:,1),feature_extraction_data_normal(:,3),'o','Color',[0, 0.4470, 0.7410],'LineWidth',2); hold on; % Normal Signal
plot(feature_extraction_data_myopathy(:,1),feature_extraction_data_myopathy(:,3),'^','Color',[0.4660, 0.6740, 0.1880],'LineWidth',2); hold on;% Mypopathy
plot(feature_extraction_data_neuropathy(:,1),feature_extraction_data_neuropathy(:,3),'s','Color',[0.6350, 0.0780, 0.1840],'LineWidth',2); hold on; % Neuropathy
xlabel('RMS') ; ylabel('Mean_ABS');
legend('Normal','Myopathy','Neuropathy');

MEAN_ABS_Data = [feature_extraction_data_normal(:,3).' feature_extraction_data_myopathy(:,3).' feature_extraction_data_neuropathy(:,3).'];
RMS_MEAN_ABS_Correlation  = corrcoef(RMS_Data,MEAN_ABS_Data)

% 3) RMS & VARINCE
subplot(3,1,3);

plot(feature_extraction_data_normal(:,1),feature_extraction_data_normal(:,2),'o','Color',[0, 0.4470, 0.7410],'LineWidth',2); hold on; % Normal Signal
plot(feature_extraction_data_myopathy(:,1),feature_extraction_data_myopathy(:,2),'^','Color',[0.4660, 0.6740, 0.1880],'LineWidth',2); hold on;% Mypopathy
plot(feature_extraction_data_neuropathy(:,1),feature_extraction_data_neuropathy(:,2),'s','Color',[0.6350, 0.0780, 0.1840],'LineWidth',2); hold on; % Neuropathy
xlabel('RMS') ; ylabel('VARINCE');
legend('Normal','Myopathy','Neuropathy');

VARINCE_Data = [feature_extraction_data_normal(:,2).' feature_extraction_data_myopathy(:,2).' feature_extraction_data_neuropathy(:,2).'];
RMS_VARINCE_Correlation  = corrcoef(RMS_Data,VARINCE_Data)

suptitle('Correlation of Data');
%% B. Preparing Data
% What is Data Preparation for Machine Learning? 
% Data preparation (also referred to as “data preprocessing”) is the process of transforming raw data so that data scientists and analysts can run it through machine learning algorithms to uncover insights or make predictions. Improperly formatted / structured data.
% 1 = signal_RMS; 2 = signal_Var;
features_normal = feature_extraction_data_normal(:,1:2); 
features_myopathy =  feature_extraction_data_myopathy(:,1:2);  
features_neuropathy = feature_extraction_data_neuropathy(:,1:2); 

signal_features = [features_normal; features_myopathy; features_neuropathy];
signal_labels = zeros(308,1); % 308 is the total number of windows

signal_labels(1:50,1) = ones(50,1); % 1 for Normal Signal
signal_labels(51:160,1) = 2*ones(110,1); % 2 for Mypathy Signal
signal_labels(161:end,1) = 3*ones(148,1); % 3 for Neuropath Signal

preparing_data_table = [signal_labels signal_features];
xlswrite('preparing_data_table.xlsx',preparing_data_table);


%% C. Build Model
% Mdl = fitcecoc(Tbl,ResponseVarName) returns a full, trained, multiclass, 
% error-correcting output codes (ECOC) model using the predictors in table Tbl
% and the class labels in Tbl.ResponseVarName. fitcecoc uses K(K – 1)/2 binary support vector machine (SVM)
% models using the one-versus-one coding design, where K is the number of unique class labels (levels).
% Mdl is a ClassificationECOC model.

X_train = [signal_features(30:200,1); signal_features(200:250,1)]; %  70% of total data
Y_train = [signal_labels(30:200,1); signal_labels(200:250,1)]; %    70% of total data

X_test = [signal_features(1:30,1); signal_features(250:end,1)]; %  30% of total data
Y_test = [signal_labels(1:30,1); signal_labels(250:end,1)]; %  30% of total data

rng(1); % For reproducibility

modelTemplate = templateSVM('Standardize',true, 'KernelFunction','gaussian');
modelSupportVector = fitcecoc(X_train,Y_train,'Learners',modelTemplate,...
    'ClassNames',{'1','2','3'});


%% D. Test and Assessment Mode
% Feature selection, feature engineering, model selection, hyperparameter optimization, cross-validation, predictive performance evaluation, and classification accuracy comparison tests
% When you build a high-quality, predictive classification model, it is important to select the right features (or predictors) and tune hyperparameters (model parameters that are not estimated).
% Feature selection and hyperparameter tuning can yield multiple models. You can compare the k-fold misclassification rates, receiver operating characteristic (ROC) curves, or confusion matrices among the models.
% Or, conduct a statistical test to detect whether a classification model significantly outperforms another.
Y_predicted  = predict(modelSupportVector,X_test);

confusion_mat = confusion_matrix(Y_test,Y_predicted,[1,2,3])
model_accuracy = ((confusion_mat(1,1) + confusion_mat(2,2) + confusion_mat(3,3)) / length(Y_test)) * 100