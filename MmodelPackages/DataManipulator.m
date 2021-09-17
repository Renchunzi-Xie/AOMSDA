% Marcus Vinicius Sousa Leite de Carvalho
% marcus.decarvalho@ntu.edu.sg
%
% NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License Agreement
% Non-Commercial Use Only 
% This NTUITIVE License Agreement, including all exhibits ("NTUITIVE-LA") is a legal agreement between you and NTUITIVE (or “we”) located at 71 Nanyang Drive, NTU Innovation Centre, #01-109, Singapore 637722, a wholly owned subsidiary of Nanyang Technological University (“NTU”) for the software or data identified above, which may include source code, and any associated materials, text or speech files, associated media and "online" or electronic documentation and any updates we provide in our discretion (together, the "Software"). 
% 
% By installing, copying, or otherwise using this Software, found at https://github.com/Ivsucram/ATL_Matlab, you agree to be bound by the terms of this NTUITIVE-LA.  If you do not agree, do not install copy or use the Software. The Software is protected by copyright and other intellectual property laws and is licensed, not sold.   If you wish to obtain a commercial royalty bearing license to this software please contact us at marcus.decarvalho@ntu.edu.sg.
% 
% SCOPE OF RIGHTS:
% You may use, copy, reproduce, and distribute this Software for any non-commercial purpose, subject to the restrictions in this NTUITIVE-LA. Some purposes which can be non-commercial are teaching, academic research, public demonstrations and personal experimentation. You may also distribute this Software with books or other teaching materials, or publish the Software on websites, that are intended to teach the use of the Software for academic or other non-commercial purposes.
% You may not use or distribute this Software or any derivative works in any form for commercial purposes. Examples of commercial purposes would be running business operations, licensing, leasing, or selling the Software, distributing the Software for use with commercial products, using the Software in the creation or use of commercial products or any other activity which purpose is to procure a commercial gain to you or others.
% If the Software includes source code or data, you may create derivative works of such portions of the Software and distribute the modified Software for non-commercial purposes, as provided herein.  
% If you distribute the Software or any derivative works of the Software, you will distribute them under the same terms and conditions as in this license, and you will not grant other rights to the Software or derivative works that are different from those provided by this NTUITIVE-LA. 
% If you have created derivative works of the Software, and distribute such derivative works, you will cause the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.
% 
% You may not distribute this Software or any derivative works. 
% In return, we simply require that you agree: 
% 1.	That you will not remove any copyright or other notices from the Software.
% 2.	That if any of the Software is in binary format, you will not attempt to modify such portions of the Software, or to reverse engineer or decompile them, except and only to the extent authorized by applicable law. 
% 3.	That NTUITIVE is granted back, without any restrictions or limitations, a non-exclusive, perpetual, irrevocable, royalty-free, assignable and sub-licensable license, to reproduce, publicly perform or display, install, use, modify, post, distribute, make and have made, sell and transfer your modifications to and/or derivative works of the Software source code or data, for any purpose.  
% 4.	That any feedback about the Software provided by you to us is voluntarily given, and NTUITIVE shall be free to use the feedback as it sees fit without obligation or restriction of any kind, even if the feedback is designated by you as confidential. 
% 5.	THAT THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
% 6.	THAT NEITHER NTUITIVE NOR NTU NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS NTUITIVE-LA, INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
% 7.	That we have no duty of reasonable care or lack of negligence, and we are not obligated to (and will not) provide technical support for the Software.
% 8.	That if you breach this NTUITIVE-LA or if you sue anyone over patents that you think may apply to or read on the Software or anyone's use of the Software, this NTUITIVE-LA (and your license and rights obtained herein) terminate automatically.  Upon any such termination, you shall destroy all of your copies of the Software immediately.  Sections 3, 4, 5, 6, 7, 8, 11 and 12 of this NTUITIVE-LA shall survive any termination of this NTUITIVE-LA.
% 9.	That the patent rights, if any, granted to you in this NTUITIVE-LA only apply to the Software, not to any derivative works you make.
% 10.	That the Software may be subject to U.S. export jurisdiction at the time it is licensed to you, and it may be subject to additional export or import laws in other places.  You agree to comply with all such laws and regulations that may apply to the Software after delivery of the software to you.
% 11.	That all rights not expressly granted to you in this NTUITIVE-LA are reserved.
% 12.	That this NTUITIVE-LA shall be construed and controlled by the laws of the Republic of Singapore without regard to conflicts of law.  If any provision of this NTUITIVE-LA shall be deemed unenforceable or contrary to law, the rest of this NTUITIVE-LA shall remain in full effect and interpreted in an enforceable manner that most nearly captures the intent of the original language. 
% 
% Copyright (c) NTUITIVE. All rights reserved.

classdef DataManipulator < handle
    %DataManipulator It manipulates and prepare data
    %   It manipulates and prepare data used to train and test our research
    %   models.
    %   It is already prepared to load and interact with mostly of the data
    %   used in our lab.
    properties (Access = public)
        data = []; % Whole dataset
        nFeatures = 0; % Number of features from the dataset
        nClasses = 0; % Number of classes from the dataset
        
        nFoldElements = 0; % Number of elements per fold
        nMinibatches = 0; % Number of minibatches
        samplingRatio =0;
        labelledSourceRowIndex;
        source = {}; % Souce data
        target = {}; % Target data
        nSource = 3;
        
       
    end
    properties (Access = private)
        X  = {}; % Input data
        y  = {}; % Class data
        Xs = {}; % Source input data
        ys = {}; % Source class data
        Xt = {}; % Target input data
        yt = {}; % Target class data
        
        permutedX = {}; % Permutted Input data
        permutedy = {}; % Permutted Class data
        
        indexPermutation = {}; % Permuttation index (in order to know if it source or target)
        
        dataFolderPath = '';
    end
    
    methods (Access = public)
        function self = DataManipulator(dataFolderPath)
            self.dataFolderPath = dataFolderPath;
        end

        
        function loadWeatherDataset(self)
            %loadWeatherDatasetDataset
            %   Load the Weather Pricing Dataset
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/weather.mat'));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 8;
            self.nClasses = 2;
            self.X = self.data(:,1:end-self.nClasses);
            self.y = self.data(:,self.nFeatures+1:end);
        end
        
         function loadSeaDataset(self)
            %loadElectricityPricingDataset
            %   Load the SEA Dataset
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/sea.mat'));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 3;
            self.nClasses = 2;
            self.X = self.data(:,1:end-self.nClasses);
            self.y = self.data(:,self.nFeatures+1:end);
        end
        
        function loadHyperplaneDataset(self)
            %loadHyperplaneDataset
            %   Load the HYPERPLANE Dataset
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/hyperplane.mat'));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 4;
            self.nClasses = 2;
            self.X = self.data(:,1:end-self.nClasses);
            self.y = self.data(:,self.nFeatures+1:end);
        end
        
        function loadInjordexpDataset(self)
            %loadInjordexpDataset
            %   Load the Injorde Dataset 
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/injordexp.mat'));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 48;
            self.nClasses = 3;
            self.X = self.data(:,1:self.nFeatures);
            self.y = self.data(:,self.nFeatures+1:end);
            
            self.y(self.y == 0) = 3;
            y_one_hot = dummyvar(self.y);
            
            self.y = y_one_hot;
            self.data = [self.X self.y];
        end
       
        function loadHepmassDataset(self)
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/hepmass.mat'));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 28;
            self.nClasses = 2;
            self.X = self.data(:,1:self.nFeatures);
            self.y = self.data(:,self.nFeatures+1:end);
            self.data = [self.X self.y];
        end
        
        function loadOccupancyDataset(self)
            %loadOccupancyDataset
            %   Load the Occupancy Dataset 
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/occupancy_ori.mat'));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 5;
            self.nClasses = 2;
            self.X = self.data(:,1:self.nFeatures);
            self.y = self.data(:,self.nFeatures+1:end);
            self.nClasses = max(self.y);
            
            y_one_hot = dummyvar(self.y);
            
            self.y = y_one_hot;
            self.data = [self.X self.y];
        end
        
        function loadSUSYDataset(self, ratio)
            %loadHepmassDataset
            %   Load the Hepmass Dataset 
            
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/SUSY.csv'));
            %Temp
            self.data = [self.data(:, 1:end-1), dummyvar(self.data(:,end))];
            %Temp
            self.data = self.data(1:size(self.data, 1) * ratio, :);
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 18;
            self.nClasses = 2;
            self.X = self.data(:,1:end-self.nClasses);
            self.y = self.data(:,self.nFeatures+1:end);
        end
        function loadKittiDataset(self, ratio)
            %loadSUSYDataset
            %   Load the SUSY Dataset 
            if nargin == 1
                ratio = 1;
            end
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/kitti_ori.csv'));
            self.nFeatures = size(self.data, 2) - 1;
            self.nClasses = max(self.data(:, end));
            %Temp
            self.data = [self.data(:, 1:end-1), dummyvar(self.data(:,end))];
            %Temp
            self.data = self.data(1:size(self.data, 1) * ratio, :);
            self.checkDatasetEven();
            self.data = double(self.data);
            
            self.X = self.data(:,1:end-self.nClasses);
            self.y = self.data(:,self.nFeatures+1:end);
        end
        function loadKDDDataset(self)
            %loadKDDDataset
            %   Load the KDD Dataset
            self.data = [];
            self.data = importdata(strcat(self.dataFolderPath, '/KDDCUP.mat'));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 41;
            self.nClasses = 2;
            self.X = self.data(:,1:end-self.nClasses);
            self.y = self.data(:,self.nFeatures+1:end);
        end
        
        function loadForestCoverTypeDataset(self)
            %loadForestCoverTypeDataset
            %   Load the Occupancy Dataset
            %   
            self.data = [];
            self.data = csvread(strcat(self.dataFolderPath, '/forest.csv'));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = 54;
            self.X = self.data(:,1:self.nFeatures);
            self.y = self.data(:,self.nFeatures+1:end);
            self.nClasses = max(self.y);
            self.y = dummyvar(self.y);
            self.data = [self.X, self.y];
        end
        
        function normalize2(self, data)
            %normalize
            %   Normalize every feature between 0 and 1
            fprintf('Normalizing data\n');
            for i = 1 : self.nFeatures
                self.data(:, i) = (data(:, i) - min(data(:, i), [], 'all'))/max(data(:, i), [], 'all');
            end
            nDelete = size(data(:, ~any(~isnan(data), 1)), 2);
            data(:, ~any(~isnan(data), 1))=[];
%             if sum(nDelete) ~= 0
%                 self.nFeatures = self.nFeatures - nDelete;
%             end
%             self.X = self.data(:, 1 : self.nFeatures);
%             self.y = self.data(:, self.nFeatures + 1 : end);
        end
        
        function normalize(self)
            %normalize
            %   Normalize every feature between 0 and 1
            fprintf('Normalizing data\n');
            for i = 1 : self.nFeatures
                self.data(:, i) = (self.data(:, i) - min(self.data(:, i), [], 'all'))/max(self.data(:, i), [], 'all');
            end
            nDelete = size(self.data(:, ~any(~isnan(self.data), 1)), 2);
            self.data(:, ~any(~isnan(self.data), 1))=[];
            if sum(nDelete) ~= 0
                self.nFeatures = self.nFeatures - nDelete;
            end
            self.X = self.data(:, 1 : self.nFeatures);
            self.y = self.data(:, self.nFeatures + 1 : end);
        end
        
         function splitAsSourceTargetStreams2(self, nFoldElements, samplingRatio)
            %This function is for multi-source domain problem.
            self.nFoldElements = nFoldElements;
            self.samplingRatio = samplingRatio;
            self.multiSourceSplit();
            self.generateChunks();
        end
        function generateArtificialsets(self, nSamples, drift)
            self.nFoldElements = 500;
            self.nFeatures = 2;
            self.nClasses = 2;
            
            if drift == 0
                self.nSource = 2;
                nTargets = 1;
                u_source = [];
                sigma_source = [];
                u_source(:, 1, 1) = [-3, 6];
                u_source(:, 2, 1) = [7, 8];
                u_source(:, 1, 2) = [2, 1];
                u_source(:, 2, 2) = [7, 8];
                sigma_source(:, :, 1, 1) = [3, 0; 0, 2];
                sigma_source(:, :, 1, 2) = [3, 0; 0, 2];
                sigma_source(:, :, 2, 1) = [1, 0; 0 ,2];
                sigma_source(:, :, 2, 2) = [1, 0; 0, 2];
                
                u_target = [];
                sigma_target = [];
                u_target(:, 1, 1) = [2, 3];
                u_target(:, 2, 1) = [7, 8];
                sigma_target(:, :, 1, 1) = [2, 0; 0, 2];
                sigma_target(:, :, 2, 1) = [2, 0; 0, 2];
                for i= 1: self.nSource
                    self.source{i} = mvnrnd(u_source(:, 1, i),sigma_source(:, :, 1, i),nSamples);
                    self.source{i} = [self.source{i}, ones(nSamples, 1), zeros(nSamples, 1)];
                    temp = mvnrnd(u_source(:, 2, i),sigma_source(:, :, 2, i),nSamples);
                    temp = [temp, zeros(nSamples, 1), ones(nSamples, 1)];
                    self.source{i} = [self.source{i}; temp];
                    self.source{i} = self.source{i}(randperm(size(self.source{i},1)), :);
                end
                for i = 1: nTargets
                    self.target{i} = mvnrnd(u_target(:, 1, i),sigma_target(:, :, 1, i),nSamples);
                    self.target{i} = [self.target{i}, ones(nSamples, 1), zeros(nSamples, 1)];
                    temp = mvnrnd(u_target(:, 2, i),sigma_target(:, :, 2, i),nSamples);
                    temp = [temp, zeros(nSamples, 1), ones(nSamples, 1)];
                    self.target{i} = [self.target{i}; temp];
                    self.target{i} = self.target{i}(randperm(size(self.target{i},1)), :);
                end
            elseif drift == 1
                self.nSource = 1;
                nTargets = 1;
                u_source = [];
                sigma_source = [];
                u_source(:, 1, 1) = [2, 9];
                u_source(:, 2, 1) = [5, 4];
                sigma_source(:, :, 1, 1) = [1, 0; 0, 2];
                sigma_source(:, :, 2, 1) = [1, 0; 0, 2];
                
                u_target = [];
                sigma_target = [];
                u_target(:, 1, 1, 1) = [2, 3];
                u_target(:, 1, 2, 1) = [7, 8];
                sigma_target(:, :, 1, 1, 1) = [1, 0; 0, 2];
                sigma_target(:, :, 2, 1, 1) = [1, 0; 0, 2];
                u_target(:, 1, 1, 2) = [2, 9];
                u_target(:, 1, 2, 2) = [5, 4];
                sigma_target(:, :, 1, 1, 2) = [1, 0; 0, 2];
                sigma_target(:, :, 2, 1, 2) = [1, 0; 0, 2];
                
                for i = 1: self.nSource
                    self.source{i} = mvnrnd(u_source(:, 1, i),sigma_source(:, :, 1, i),nSamples);
                    self.source{i} = [self.source{i}, ones(nSamples, 1), zeros(nSamples, 1)];
                    temp = mvnrnd(u_source(:, 2, i),sigma_source(:, :, 2, i),nSamples);
                    temp = [temp, zeros(nSamples, 1), ones(nSamples, 1)];
                    self.source{i} = [self.source{i}; temp];
                    self.source{i} = self.source{i}(randperm(size(self.source{i},1)), :);
                end
                for i = 1;nTargets
                    self.target{i} = mvnrnd(u_target(:, i, 1, 1),sigma_target(:, :, 1, i, 1),nSamples/2);
                    self.target{i} = [self.target{i}, ones(nSamples/2, 1), zeros(nSamples/2, 1)];
                    temp = mvnrnd(u_target(:, i, 2, i),sigma_target(:, :, 2, i, 1),nSamples/2);
                    temp = [temp, zeros(nSamples/2, 1), ones(nSamples/2, 1)];
                    self.target{i} = [self.target{i}; temp];
                    temp = mvnrnd(u_target(:, i, 1, 2),sigma_target(:, :, 1, i, 2),nSamples/2);
                    temp = [temp, ones(nSamples/2, 1), zeros(nSamples/2, 1)];
                    self.target{i} = [self.target{i}; temp];
                    temp = mvnrnd(u_target(:, i, 2, 2),sigma_target(:, :, 2, i, 2),nSamples/2);
                    temp = [temp, zeros(nSamples/2, 1), ones(nSamples/2, 1)];
                    self.target{i} = [self.target{i}; temp];
                    self.target{i} = self.target{i}(randperm(size(self.target{i},1)), :);
                end
            elseif drift == 2
                self.nSource = 6;
                nTargets = 1;
                u = [2,3;7,8];
                sigma = [1,0;0,2];
                for i = 1:self.nSource
                    self.source{i} = mvnrnd(u(1, :)+i-1,sigma, nSamples);
                    self.source{i} = [self.source{i}, ones(nSamples, 1), zeros(nSamples, 1)];
                    temp = mvnrnd(u(2, :)-i+1, sigma, nSamples);
                    temp = [temp, zeros(nSamples, 1), ones(nSamples, 1)];
                    self.source{i} = [self.source{i}; temp];
                    self.source{i} = self.source{i}(randperm(size(self.source{i},1)), :);
                end
                for i = 1:nTargets
                    target_nSamples = round(nSamples/6);
                    self.target{i} = [];
                    for j = 1 :6                        
                        temp1 = mvnrnd(u(1, :)+i-1, sigma, target_nSamples);
                        temp1 = [temp1, ones(target_nSamples, 1), zeros(target_nSamples, 1)];
                        self.target{i} = [self.target{i}; temp1];
                        temp2 = mvnrnd(u(2, :)-i+1, sigma, target_nSamples);
                        temp2 = [temp2, zeros(target_nSamples, 1), ones(target_nSamples, 1)];
                        self.target{i} = [self.target{i}; temp2];
                    end
                    self.target{i} = self.target{i}(randperm(size(self.target{i},1)), :);
                end
            end
            self.generateChunks();
        end
        function multiSourceSplit(self)
            %This function is for multi-source problem.
            mean_x = mean(self.X, 1);
            std_x = std(self.X);
            probability = vecnorm((self.X - mean_x).^2./(2*std_x.^2), 2, 2);
            [out,idx] = sort(probability, "ascend");
            source_total = self.data(idx(1:round((1-self.samplingRatio)*length(idx))),:);
            self.target{1} = self.data(idx(round((1-self.samplingRatio)*length(idx))+1:end),:);
            mean_source_x = mean(source_total(:, 1:self.nFeatures), 1);
            std_source_x = std(source_total(:, 1:self.nFeatures));
            probability = vecnorm((source_total(:, 1:self.nFeatures)-mean_source_x).^2./(2*std_source_x.^2), 2, 2);
            [out, idx] = sort(probability, "ascend");
            ps = 1/self.nSource;
            for i = 1:self.nSource
                self.source{i} = source_total(idx(round((i-1)*length(idx)*ps)+1:round(i*length(idx)*ps)),:);
            end
        end
        
        function generateChunks(self)
            
            numOfchunks = ceil(length(self.target{1})/self.nFoldElements);
            self.nMinibatches = numOfchunks;
            final_target = {};
            final_source = {};
            
            for i = 1:numOfchunks
                if i == numOfchunks
                    final_target{1,i} = self.target{1}((i-1)*self.nFoldElements+1:end,:);
                    for j = 1:self.nSource
                      final_source{j,i} = self.source{j}((i-1)*self.nFoldElements+1:end,:);
                    end
                else
                    final_target{1,i} = self.target{1}((i-1)*self.nFoldElements+1:i*self.nFoldElements,:);
                    for j = 1:self.nSource
                        final_source{j,i} = self.source{j}((i-1)*self.nFoldElements+1:i*self.nFoldElements,:);
                    end
                end
            end
            self.source = final_source;
            self.target = final_target;
        end
        
        function splitAsSourceTargetStreams(self, nFoldElements, method, samplingRatio)
            %splitAsSourceTargetStreams
            %   Split the function to simulate a Multistream classification
            %   input domains.
            %   In a Multistream classification problem, we consider that
            %   two different but related processes generate data
            %   continuously from a domain D (in this case, self.data). The
            %   first process operates in a supervised environment, i.e.,
            %   all the data instances that are generated from the first
            %   process are labeled. On the contraty, the second process
            %   generates unlabeled data from the same domain. The stream
            %   of data generated form the above processes are called the
            %   source stream and the target stream.
            %   This functions will return label for the target stream,
            %   which the user should only use for ensemble evaluation
            %   purposes
            %   nFoldElements (integer)
            %       Both source and target data will be splited in chunks
            %       of data containing n elements per chunk/fold.
            %       If you only want one chunk, pass zero or size(data,1)
            %       as argument.
            %   method (string)
            %       What kind of method will be used to generated
            %       distribute the data into source and target. Usually,
            %       Multistream Classification problems distribute the data
            %       using some bias probability.
            %       Options:
            %           'none': Source and Target streams will be splited on
            %           half
            %           'dallas_1: Source and Target streams will be splited
            %           on half using the bias described by paper "An
            %           adaptive framework for multistream classification"
            %           from the CS deparment of the university of Texas at
            %           Dallas
            %           'dallas_2:' Source and Target streams will be
            %           splited on half using the bias described by paper
            %           "FUSION - An online method for multistream
            %           classification" from the university of Texas at
            %           Dallas.
            %   samplingRatio (double)
            %       Value in the interval [0.0,1.0] which describes the
            %       percentage of sampling that would go to Source Stream.
            %       Target will have 1 - n percentagem of data.
            if nFoldElements == 0
                self.nFoldElements = length(self.data);
            else
                self.nFoldElements = nFoldElements;
            end
            
            switch method
                case 'none'
                    self.splitAsSourceTargetStreams_none(self.nFoldElements, samplingRatio)
                case 'dallas_1'
                    self.splitAsSourceTargetStreams_dallas1(self.nFoldElements, samplingRatio)
                case 'dallas_2'
                    self.splitAsSourceTargetStreams_dallas2(self.nFoldElements, samplingRatio)
            end
            self.finalSource(0.25, false, false);
            self.source = self.finalSourceData;
            self.createXsYsXtYt()
        end
        
         function finalSource(self, sampleRatio, isArgmented, isAlwaysArgmented)
            %Build the final source data adding argmented, unlabelled and
            %labelled samples.
            %   sampleRatio (double)
            %       The proportion of labelled samples of source.
            %   isArgmented (logic)
            %       The augment tragger.
            %   isAlwaysArgmented (logic)
            %       'true' : Every labelled samples should be argmented.
            %       'false': Argment those labelled samples that are
            %       checked in terms of error distribution. 
            rng(10);
            self.generateLabelledSource(sampleRatio);
            for i = 1:length(self.source)
                if isArgmented
                    if isAlwaysArgmented
%                       argSamples = awgn(self.labelledSource{i}(:,1:self.nFeatures),20);
                        argSamples = self.addNoise(self.labelledSource{i}(:,1:self.nFeatures), 'method_1');
                        argSamples = [argSamples,self.labelledSource{i}(:,self.nFeatures+1:end)];
                        self.finalSourceData{i} = [argSamples;self.labelledSource{i};self.unlabelledSource{i}];
                       
                    else
                        self.finalSourceData{i} =[self.labelledSource{i};self.unlabelledSource{i}];
                        
                    end
                else
                    self.finalSourceData{i} =[self.labelledSource{i};self.unlabelledSource{i}];                 
                end
                order = randperm(size(self.finalSourceData{i},1));
                self.finalSourceData{i} = self.finalSourceData{i}(order,:);
            end
            rng shuffle
         end
        
         function generateLabelledSource(self, sampleRatio)
            %generateLabelledSource
            %   Decide which source data are labelled and which are
            %   unlabelled.
            %   sampleRatio (double)
            %       the proportion of labelled samples of source.
            nSource = (length(self.source)-1)*self.nFoldElements/2+size(self.source{end},1);
            nLabelledSource = ceil(nSource*sampleRatio);
            self.labelledSourceRowIndex = datasample([1:nSource], nLabelledSource, ...
                'replace', false);
            self.labelledSourceRowIndex = sort(self.labelledSourceRowIndex);
            totalIndex = [1:nSource];
            unlabelledSourceRowIndex = totalIndex;
            unlabelledSourceRowIndex(self.labelledSourceRowIndex) = [];
            tem = cell2mat(self.source');
            for i = 1:length(self.source) 
                self.labelledSource{i} = tem(self.labelledSourceRowIndex>=(i-1)*size(self.source{i},1)+1&...
                    self.labelledSourceRowIndex<=i*size(self.source{i},1),:);
                self.unlabelledSource{i} = zeros(size(tem(unlabelledSourceRowIndex>=(i-1)*size(self.source{i},1)+1&...
                    unlabelledSourceRowIndex<=i*size(self.source{i},1),:)));
%                 self.unlabelledSource{i} = tem(unlabelledSourceRowIndex>=(i-1)*size(self.source{i},1)+1&...
%                     unlabelledSourceRowIndex<=i*size(self.source{i},1),:);
%                 self.unlabelledSource{i}(:, end-1) = -1; 
            end
        end
         
        function Xs = getXs(self, nMinibatch)
            %getXs
            %   Get the input matrix from a specific source data stream.
            %   The source stream will be only created when we are dealing 
            %   with a dataset that was splitted into source and target 
            %   data streams.
            %   nMinibatch (integer)
            %       The minibatch iteration
            Xs = self.Xs{nMinibatch};
        end
        function Xs = getXs2(self, iSource, iMinibatch)
            Xs = self.source{iSource, iMinibatch}(:, 1:self.nFeatures);
        end
        function ys = getYs(self, nMinibatch)
            %getXs
            %   Get the target matrix from a specific source data stream.
            %   The source stream will be only created when we are dealing 
            %   with a dataset that was splitted into source and target 
            %   data streams.
            %   nMinibatch (integer)
            %       The minibatch iteration
            ys = self.ys{nMinibatch};
        end
        function ys = getYs2(self, iSource, iMinibatch)
            ys = self.source{iSource, iMinibatch}(:, self.nFeatures + 1:end);
        end
        function Xt = getXt(self, nMinibatch)
            %getXt
            %   Get the input matrix from a specific target data stream.
            %   The target stream will be only created when we are dealing 
            %   with a dataset that was splitted into source and target 
            %   data streams.
            %   nMinibatch (integer)
            %       The minibatch iteration
            Xt = self.Xt{nMinibatch};
        end
        function Xt = getXt2(self, iTarget, iMinibatch)
            Xt = self.target{iTarget, iMinibatch}(:, 1:self.nFeatures);
        end
        function yt = getYt(self, nMinibatch)
            %getXs
            %   Get the target matrix from a specific target data stream.
            %   The target stream will be only created when we are dealing 
            %   with a dataset that was splitted into source and target 
            %   data streams.
            %   nMinibatch (integer)
            %       The minibatch iteration
            yt = self.yt{nMinibatch};
        end
        function yt = getYt2(self, iTarget, iMinibatch)
            yt = self.target{iTarget, iMinibatch}(:, self.nFeatures+1:end);
        end
    end
    methods (Access = private)
        function splitAsSourceTargetStreams_none(self, elementsPerFold, samplingRatio)
            %splitAsSourceTargetStreams_none
            %   Split the function to simulate a Multistream classification
            %   input domains.
            %
            %   Source and Target streams will be splited on half
            %
            %   nFoldElements (integer)
            %       Both source and target data will be splited in chunks
            %       of data containing n elements per chunk/fold.
            %       If you only want one chunk, pass zero or size(data,1)
            %       as argument.
            %   samplingRatio (double)
            %       Value in the interval [0.0,1.0] which describes the
            %       percentage of sampling that would go to Source Stream.
            %       Target will have 1 - n percentagem of data.
            [rowsNumber, ~] = size(self.data);
    
            numberOfFolds = round(length(self.data)/elementsPerFold);
            chunkSize = round(rowsNumber/numberOfFolds);
            numberOfFoldsRounded = round(rowsNumber/chunkSize);
            self.nFoldElements = min(elementsPerFold, length(self.data)/numberOfFoldsRounded);
            
            if length(self.data)/numberOfFoldsRounded > elementsPerFold
                numberOfFolds = numberOfFolds + 1;
            end
            self.nMinibatches = numberOfFolds;
            ck = self.nFoldElements;
            
            for i = 1:numberOfFolds
                data = [];
                if i > numberOfFoldsRounded
                    data = self.data(ck * (i-1) + 1:end,1:end);
                else
                    data = self.data(ck * (i-1) + 1:ck * i,1:end);
                end
                
                m = size(data,1);
                source = data(1:ceil(m*samplingRatio),1:end);
                target = data(ceil(m*samplingRatio)+1:m,1:end);
                
                self.source{i} = source;
                self.target{i} = target;
            end
        end
        function splitAsSourceTargetStreams_dallas1(self, elementsPerFold, samplingRatio)
            %splitAsSourceTargetStreams_dallas1
            %   Split the function to simulate a Multistream classification
            %   input domains.
            %
            %   Source and Target streams will be splited on half using the
            %   bias described by paper "An adaptive framework for 
            %   multistream classification" from the CS deparment of the 
            %   university of Texas at Dallas
            %
            %   nFoldElements (integer)
            %       Both source and target data will be splited in chunks
            %       of data containing n elements per chunk/fold.
            %       If you only want one chunk, pass zero or size(data,1)
            %       as argument.
            %   samplingRatio (double)
            %       Value in the interval [0.0,1.0] which describes the
            %       percentage of sampling that would go to Source Stream.
            %       Target will have 1 - n percentagem of data.
            [rowsNumber, ~] = size(self.data);
    
            numberOfFolds = round(length(self.data)/elementsPerFold);
            chunkSize = round(rowsNumber/numberOfFolds);
            numberOfFoldsRounded = round(rowsNumber/chunkSize);
            self.nFoldElements = min(elementsPerFold, length(self.data)/numberOfFoldsRounded);
            
            if length(self.data)/numberOfFoldsRounded > elementsPerFold
                numberOfFolds = numberOfFolds + 1;
            end
            self.nMinibatches = numberOfFolds;
            ck = self.nFoldElements;
            
            for i = 1:numberOfFolds
                x = [];
                data = [];
                if i > numberOfFoldsRounded
                    x = self.data(ck * (i-1) + 1:end,1:end-self.nClasses);
                    data = self.data(ck * (i-1) + 1:end,1:end);
                else
                    x = self.data(ck * (i-1) + 1:ck * i,1:end-self.nClasses);
                    data = self.data(ck * (i-1) + 1:ck * i,1:end);
                end
                
                x_mean = mean(x);
                probability = exp(-abs(x - x_mean).^2);
                [~,idx] = sort(probability);
                
                m = size(data,1);
                source = data(idx(1:ceil(m*samplingRatio)),1:end);
                target = data(idx(ceil(m*samplingRatio)+1:length(data)),1:end);
                
                self.source{i} = source;
                self.target{i} = target;
            end
        end
        function splitAsSourceTargetStreams_dallas2(self, elementsPerFold, samplingRatio)
            %splitAsSourceTargetStreams_dallas2
            %   Split the function to simulate a Multistream classification
            %   input domains.
            %
            %   Source and Target streams will be splited on half using the
            %   bias described by paper "FUSION - An online method for 
            %   multistream classification" from the university of Texas at
            %   Dallas.
            %
            %   nFoldElements (integer)
            %       Both source and target data will be splited in chunks
            %       of data containing n elements per chunk/fold.
            %       If you only want one chunk, pass zero or size(data,1)
            %       as argument.
            %   samplingRatio (double)
            %       Value in the interval [0.0,1.0] which describes the
            %       percentage of sampling that would go to Source Stream.
            %       Target will have 1 - n percentagem of data.

            [rowsNumber, ~] = size(self.data);
    
            numberOfFolds = round(length(self.data)/elementsPerFold);
            chunkSize = round(rowsNumber/numberOfFolds);
            numberOfFoldsRounded = round(rowsNumber/chunkSize);
            if mod(floor(size(self.data, 1)/numberOfFoldsRounded), 2) == 0
                self.nFoldElements = min(elementsPerFold, floor(size(self.data, 1)/numberOfFoldsRounded));
            else
                self.nFoldElements = min(elementsPerFold, floor(size(self.data, 1)/numberOfFoldsRounded) - 1);
            end
            
            
            if length(self.data)/numberOfFoldsRounded > elementsPerFold
                numberOfFolds = numberOfFolds + 1;
            end
            self.nMinibatches = numberOfFolds;
            ck = self.nFoldElements;
            
            for i = 1 : numberOfFolds
                x = [];
                data = [];
                if i > numberOfFoldsRounded
                    x = self.data(ck * (i-1) + 1:end,1:end-self.nClasses);
                    data = self.data(ck * (i-1) + 1:end,1:end);
                else
                    x = self.data(ck * (i-1) + 1:ck * i,1:end-self.nClasses);
                    data = self.data(ck * (i-1) + 1:ck * i,1:end);
                end
                
                x_mean = mean(x);
                norm_1 = vecnorm((x - x_mean)',1)';
                norm_2 = vecnorm((x - x_mean)',2)';
                numerator   = norm_2;
                denominator = 2 * std(norm_1) ^ 2;
                probability = exp(-numerator/denominator);
                [~,idx] = sort(probability);
                
                m = size(data,1);
                source = data(idx(1 : ceil(m * samplingRatio)), 1 : end);
                target = data(idx(ceil(m * samplingRatio) + 1: size(data, 1)), 1 : end);
                
                self.source{i} = source;
                self.target{i} = target;
            end
        end
        
        function createXsYsXtYt(self)
            %createXsYsXtYt
            %   Split the datastream data into sets of input, output, input
            %   from source, output from source, input from target, output
            %   from target
            %   It also creates a permutted version of this data, in 
            self.X  = {};
            self.y  = {};
            self.Xs = {};
            self.ys = {};
            self.Xt = {};
            self.yt = {};
            self.permutedX = {};
            self.permutedy = {};
            for i = 1 : self.nMinibatches
                self.Xs{i} = self.source{i}(:,1:end-self.nClasses);
                self.ys{i} = self.source{i}(:,self.nFeatures+1:end);
                self.Xt{i} = self.target{i}(:,1:end-self.nClasses);
                self.yt{i} = self.target{i}(:,self.nFeatures+1:end);
                self.X{i}  = [self.Xs{i};self.Xt{i}];
                self.y{i}  = [self.ys{i};self.yt{i}];
                
                x = self.X{i};
                Y = self.y{i};
                
                p  = randperm(size(x, 1));
                self.permutedX{i} = x(p,:);
                self.permutedy{i} = Y(p,:);
                self.indexPermutation{i} = p;
            end
        end
        
        function checkDatasetEven(self)
            %checkDatasetEven
            %   Check if the number of rows in the whole dataset is even,
            %   so we can split in a equal number of elements for source
            %   and stream (when splitting by 0.5 ratio)
            %   If the number is odd, randomly trow a row away.
            if mod(length(self.data),2) ~= 0
                p = ceil(rand() * length(self.data));
                self.data = [self.data(1:p-1,:);self.data(p+1:end,:)];
            end
        end
    end
end

