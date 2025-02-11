%% MATLAB Script for Processing the EEG Dataset
% This script processes raw CSV files as described in the paper:
%   1. Down-sample from original sampling rate (assumed 200Hz) to 128Hz.
%   2. Apply a 1-40Hz band-pass Butterworth filter.
%   3. Extract the time window [0, 1.3] seconds.
%   4. Z-score normalize each channel.
%   5. Save the processed data for each subject in a .mat file.
%
% In this modified version, subject numbering is made 0-indexed.

% Define the directories:
inputDir  = 'C:\Users\farah\Desktop\adversarial_filter-master\EEG_data\ERN\train\';      % Raw CSV files
outputDir = 'C:\Users\farah\Desktop\adversarial_filter-master\EEG_data\ERN\processed\';  % To save .mat files

% Create the output directory if it does not exist:
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get a list of all CSV files in the train folder.
% (Assuming filenames like Data_S01_Sess01.csv, Data_S02_Sess01.csv, etc.)
csvFiles = dir(fullfile(inputDir, 'Data_S*_Sess*.csv'));

% Pre-allocate cell arrays to collect data for each subject.
% (The paper used 16 subjects for training.)
numSubjects = 16;  % Subjects will be numbered 0 to 15.
subjectData   = cell(numSubjects, 1);  % will hold EEG data
subjectLabels = cell(numSubjects, 1);  % will hold corresponding labels

% Processing parameters:
originalFs = 200;    % Assumed original sampling frequency (Hz); adjust if needed.
targetFs   = 128;    % Down-sampled frequency.
tStart     = 0;
tEnd       = 1.3;
numSamples = round((tEnd - tStart) * targetFs) + 1;  % Number of samples in [0, 1.3] sec

% Design a 4th order Butterworth band-pass filter (1-40Hz) for the target Fs.
[b, a] = butter(4, [1 40] / (targetFs/2), 'bandpass');

% Loop over each CSV file.
for k = 1:length(csvFiles)
    csvName     = csvFiles(k).name;
    fullCSVPath = fullfile(inputDir, csvName);
    
    % Extract the subject number from the filename (e.g., 'Data_S01_Sess01.csv')
    subjToken = regexp(csvName, 'Data_S(\d+)_Sess', 'tokens');
    if isempty(subjToken)
        warning('Filename %s does not match expected pattern.', csvName);
        continue;
    end
    % Subtract 1 to convert to 0-indexing.
    subjNum = str2double(subjToken{1}{1}) - 1;
    
    % Read the CSV file.
    % Create import options to skip the header row:
    opts = detectImportOptions(fullCSVPath);
    opts.DataLines = [2 Inf];  % Start reading from row 2 to skip the header
    data = readmatrix(fullCSVPath, opts);
    
    % Remove any rows with NaN values (in case there are blank lines)
    data = data(~any(isnan(data), 2), :);
    
    disp(['Total numeric rows: ', num2str(size(data, 1))]);
    disp('First few rows:');
    disp(data(1:5,:));
    
    % --- Segment the data into trials ---
    rawTrialSamples = 261;  % Each trial is 1.3 sec at 200 Hz (0.005 sec increments: 1.3/0.005 + 1 = 261)
    totalRows = size(data, 1);
    nTrials   = floor(totalRows / rawTrialSamples);
    
    if totalRows ~= nTrials * rawTrialSamples
        warning('The total number of rows in %s (%d rows) is not an integer multiple of rawTrialSamples. Only processing %d complete trials.', ...
                csvName, totalRows, nTrials);
        % Truncate data to only complete trials:
        data = data(1:nTrials * rawTrialSamples, :);
    end
    
    % Pre-allocate arrays for this session:
    % xSession: [numSamples x 56 x nTrials]
    xSession = zeros(numSamples, 56, nTrials);
    % ySession: [nTrials x 1] (assumes label is in column 57)
    ySession = zeros(nTrials, 1);
    
    % Process each trial:
    for t = 1:nTrials
        % Extract rows corresponding to trial t.
        trialRaw = data((t-1)*rawTrialSamples+1 : t*rawTrialSamples, :);
        
        % Separate EEG channels and label:
        if size(trialRaw, 2) < 57
            error('File %s does not have at least 57 columns (56 EEG channels + 1 label).', csvName);
        end
        eegTrial = trialRaw(:, 1:56);  % raw EEG data
        label    = trialRaw(1, 57);    % assume the label is constant within a trial
        
        % 1. Down-sample the trial from originalFs to targetFs.
        eegDown = resample(eegTrial, targetFs, originalFs);
        
        % 2. Apply the 1-40Hz band-pass filter using zero-phase filtering.
        eegFilt = filtfilt(b, a, eegDown);
        
        % 3. Extract the time window [0, 1.3] seconds.
        if size(eegFilt,1) >= numSamples
            eegEpoch = eegFilt(1:numSamples, :);
        else
            % If there aren’t enough samples, pad with zeros.
            eegEpoch = [eegFilt; zeros(numSamples - size(eegFilt,1), size(eegFilt,2))];
        end
        
        % 4. Z-score normalization (normalize each channel independently).
        eegNorm = (eegEpoch - mean(eegEpoch, 1)) ./ std(eegEpoch, 0, 1);
        
        % Store the processed trial and its label.
        xSession(:,:,t) = eegNorm;
        ySession(t)   = label;
    end
    
    % If a subject has multiple sessions, concatenate the trials.
    % (Note: Since subjNum is 0-indexed, store in cell index subjNum+1.)
    if isempty(subjectData{subjNum+1})
        subjectData{subjNum+1}   = xSession;
        subjectLabels{subjNum+1} = ySession;
    else
        subjectData{subjNum+1}   = cat(3, subjectData{subjNum+1}, xSession);
        subjectLabels{subjNum+1} = [subjectLabels{subjNum+1}; ySession];
    end
end

% --- Save each subject's processed data ---
% Save files as s0.mat, s1.mat, ..., s15.mat.
for subjIdx = 0:(numSubjects-1)
    % Access cell using index subjIdx+1.
    if ~isempty(subjectData{subjIdx+1})
        x = subjectData{subjIdx+1};    % EEG data: [numSamples x 56 x totalTrials]
        y = subjectLabels{subjIdx+1};  % Labels: [totalTrials x 1]
        outFile = fullfile(outputDir, sprintf('s%d.mat', subjIdx));
        save(outFile, 'x', 'y');
    else
        warning('No data found for subject %d.', subjIdx);
    end
end

disp('Processing complete. Processed files are saved in:');
disp(outputDir);
