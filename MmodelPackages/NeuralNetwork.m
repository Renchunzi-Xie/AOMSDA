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

classdef NeuralNetwork < handle & ElasticNodes & NeuralNetworkConstants
    %NEURALNETWORK It encapsulate a common MLP (Multilayer Perceptron, aka
    %Feedforward network)
    %   This object has the main attributes a Neural Network needs to
    %   operate, along with its main functions/behaviors. Some extra
    %   behaviors were built in order to achieve research goals.
    %
    %   This class features elastic network width by ElasticNodes
    %   inheritance. Network edith adaptation supports automatic generation
    %   of new hidden nodes and prunning of inconsequential nodes. This
    %   mechanism is controlled by the NS (Network Significance) method
    %   which estimates the network generalization power in terms of bias
    %   and variance.
    
    %% Standard Neural Network public properties
    properties (Access = public) 
        layers %Layers of a standard neural network
        layerValue % Layer (Input and Hidden Layer) values
        outputLayerValue % Output layers values
        
        weight % Weights
        bias % Added bias
        momentum % Weight momentum
        biasMomentum
        
        outputWeight % Weights to output layer
        outputBias % Bias to output layer
        outputMomentum % Weight momentum from output layer
        outputBiasMomentum
        
        gradient % Gradients
        outputGradient % Gradients from output layers
        biasGradient;
        outputBiasGradient;
        
        activationFunction % Each real layer activation function
        outputActivationFunctionLossFunction % Each output activation function
        
        learningRate = 0.01; % Learning rate
        momentumRate = 0.95; % Momentum rate
        
        errorValue % Network error
        lossValue % Network Loss
        
        lambda = 0.001;
        
        outputWeightInitial
        outputBiasInitial
        weightInitial
        biasInitial
    end
    
    %% Standard Neural Network protected properties
    properties (Access = protected)
        nHiddenLayers % Number of hidden layers (i.e., not counting input and output layer       
        inputSize % Size of input layer
        outputSize % Size of output layer
    end
    %% TODO define section name
    properties (Access = public)
        agmm
    end
    properties (Access = protected)
        isAgmmAble = false;
    end
    
    %% Metrics and performance public properties
    properties (Access = public)
        %test metrics
        sigma              % Network's prediction
        misclassifications % Number of misclassifications after test
        classificationRate % Classification rate after test
        residualError      % Residual error after test
        outputedClasses    % Classed outputed during classes
        trueClasses        % True target classes
    end
    
    %% Helpers protected properties
    properties (Access = protected)
        util = Util; % Caller for several util computations
    end
    
    %% Standard Neural Network public methods
    methods (Access = public)
        function self = NeuralNetwork(layers)
            %NeuralNetwork
            %   layers (array)
            %       This array describes a FeedForward Network structure by
            %       the number of layers on it.
            %       An FFNN with an input layer of 8 nodes, a hidden layer
            %       of 10 nodes and an output layer of 3 nodes would be
            %       described by [8 10 3].
            %       An FFNN with an input layer of 784 nodes, a hidden
            %       layer 1 of 800 nodes, a hidden layer 2 of 400 nodes and
            %       an output layer of 10 nodes would be described as [784 800 400 10]
            self@ElasticNodes(numel(layers) - 1);
            
            self.inputSize  = layers(1);
            self.outputSize = layers(end);
            
            self.layers = layers;
            self.nHiddenLayers = length(layers) - 2;
            
            for i = 1 : self.nHiddenLayers
                self.weight{i}       = normrnd(0, sqrt(2 / self.layers(i) + 1), [self.layers(i + 1), self.layers(i)]);
                self.bias{i}         = normrnd(0, sqrt(2 / self.layers(i) + 1), [1,                  self.layers(i + 1)]);
                self.momentum{i}     = zeros(size(self.weight{i}));
                self.biasMomentum{i} = zeros(size(self.bias{i}));
                self.activationFunction(i) = self.ACTIVATION_FUNCTION_SIGMOID();
                self.weightInitial{i} = self.weight{i};
                self.biasInitial{i} = self.bias{i};
            end
            self.outputWeight          = normrnd(0, sqrt(2 / self.layers(end) + 1), [self.layers(end), self.layers(end - 1)]);
            self.outputBias            = normrnd(0, sqrt(2 / self.layers(end) + 1), [1,                self.layers(end)]);
            self.outputMomentum        = zeros(size(self.outputWeight));
            self.outputBiasMomentum = zeros(size(self.outputBias));
            self.outputWeightInitial = self.outputWeight;
            self.outputBiasInitial = self.outputBias;
            self.outputActivationFunctionLossFunction = self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY();
        end
                
        function feedforward(self, X, y)
%             % feedforward
            %   Perform the forwarding pass throughout the network and
            %   calculate the network error
            %   X (matrix)
            %       Input matrix
            %   y (matrix)
            %       Target matrix
            self.forwardpass(X)
            self.calculateError(y)
        end

        function forwardpass(self, X)
            % forwardpass
            %   Perform the forwarding pass throughout the network without
            %   calculate the network error, that's why it doesn't need the
            %   target class.
            %   Because of this, we can use this class just to populate the
            %   hidden layers from the source data
            %   X (matrix)
            %       Input matrix
            self.layerValue{1} = X;
            
            for i = 1 : self.nHiddenLayers
                previousLayerValueWithBias = [ones(size(self.layerValue{i}, 1), 1) self.layerValue{i}];
                switch self.activationFunction(i)
                    case self.ACTIVATION_FUNCTION_SIGMOID()
                        self.layerValue{i + 1} = sigmf(previousLayerValueWithBias * [self.bias{i}' self.weight{i}]', [1, 0]);
                    
                    case self.ACTIVATION_FUNCTION_TANH()
                        error('Not implemented');
                    
                    case self.ACTIVATION_FUNCTION_RELU()
                        error('Not implemented yet');
                    
                    case self.ACTIVATION_FUNCTION_LINEAR()
                        error('Not implemented yet');
                    
                    case self.ACTIVATION_FUNCTION_SOFTMAX()
                        error('Not implemented yet');
                end               
            end
            
            previousLayerValueWithBias = [ones(size(self.layerValue{end}, 1), 1) self.layerValue{end}];
            switch self.outputActivationFunctionLossFunction
                case self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE()
                    self.outputLayerValue = sigmf(previousLayerValueWithBias * [self.outputBias' self.outputWeight]', [1, 0]);
                
                case self.ACTIVATION_LOSS_FUNCTION_TANH()
                    error('Not implemented yet');
                
                case self.ACTIVATION_LOSS_FUNCTION_RELU()
                    error('Not implemented yet');
                
                case self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY()
                    self.outputLayerValue = previousLayerValueWithBias * [self.outputBias' self.outputWeight]';
                    self.outputLayerValue = exp(self.outputLayerValue - max(self.outputLayerValue, [], 2));
                    self.outputLayerValue = self.outputLayerValue./sum(self.outputLayerValue, 2);
                
                case self.ACTIVATION_LOSS_FUNCTION_LINEAR_CROSS_ENTROPY()
                    error('Not implemented yet');
            end
        end
        
        function backpropagate(self)
            %backpropagate
            % Perform back-propagation thoughout the network.
            % We assume that you already populate the hidden layers and the
            % network error by calling the feedforward method.

            dW = {zeros(1, self.nHiddenLayers + 1)};
            db = {zeros(1, self.nHiddenLayers + 1)};
            for i = self.nHiddenLayers : - 1 : 1
                if i == self.nHiddenLayers
                    % THIS IS THE GRADIENT OF THE LOSS FUNCTION
                    switch self.outputActivationFunctionLossFunction
                        case self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE()
                            dW{i + 1} = - self.errorValue .* self.outputLayerValue .* (1 - self.outputLayerValue);
                            db{i + 1} = - sum(self.errorValue, 1)/size(self.errorValue, 1);
                        
                        case self.ACTIVATION_LOSS_FUNCTION_TANH()
                            error('Not implemented');
                        
                        case self.ACTIVATION_LOSS_FUNCTION_RELU()
                            error('Not implemented');
                        
                        case self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY()
                            dW{i + 1} = - self.errorValue;
                            db{i + 1} = - sum(self.errorValue, 1)/size(self.errorValue, 1);
                        
                        case self.ACTIVATION_LOSS_FUNCTION_LINEAR_CROSS_ENTROPY()
                            dW{i + 1} = - self.errorValue;
                            db{i + 1} = - sum(self.errorValue, 1)/size(self.errorValue, 1);
                    end
                    
                end
                
                switch char(self.activationFunction(i))
                    case self.ACTIVATION_FUNCTION_SIGMOID()
                        dActivationFunction = self.layerValue{i + 1} .* (1 - self.layerValue{i + 1});
                    case self.ACTIVATION_FUNCTION_TANH()
                        error('Not implemented');
                    
                    case self.ACTIVATION_FUNCTION_RELU()
                        error('Not implemented');
                    
                    case self.ACTIVATION_FUNCTION_LINEAR()
                        dActivationFunction = 1;
                    
                    case self.ACTIVATION_FUNCTION_SOFTMAX()
                        error('Not implemented');
                end
                
                if i == self.nHiddenLayers
                        z     = dW{i + 1} * self.outputWeight;
                        dW{i} = z .* dActivationFunction;
                        db{i} = sum(dW{i})/size(dW{i}, 1);
                else
                        z     = dW{i + 1} * self.weight{i + 1};
                        dW{i} = z .* dActivationFunction;
                        db{i} = sum(dW{i})/size(dW{i}, 1);
                end
                
            end
            
            self.outputGradient     = dW{end}' * self.layerValue{end};
            self.outputBiasGradient = db{end};
            for i = 1 : self.nHiddenLayers
                self.gradient{i}     = dW{i}' * self.layerValue{i};
                self.biasGradient{i} = db{i};
            end
        end
        
        function backpropagate2(self, outputValue1, outputValue2, s)
            %backpropagate2
            % Perform back-propagation thoughout the network.
            % We assume that you already populate the hidden layers and the
            % network error by calling the feedforward method.

            dW = {zeros(1, self.nHiddenLayers + 1)};
            db = {zeros(1, self.nHiddenLayers + 1)};
            for i = self.nHiddenLayers : - 1 : 1
                if i == self.nHiddenLayers
                    % THIS IS THE GRADIENT OF THE LOSS FUNCTION
                    switch self.outputActivationFunctionLossFunction
                        case self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE()
                            error('Not implemented');
                        
                        case self.ACTIVATION_LOSS_FUNCTION_TANH()
                            error('Not implemented');
                        
                        case self.ACTIVATION_LOSS_FUNCTION_RELU()
                            error('Not implemented');
                        
                        case self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY()
                            dW{i + 1} =  - 2*(outputValue1 - outputValue2).*s.* self.outputLayerValue .* (1 - self.outputLayerValue);
                            db{i + 1} =  - sum(2*(outputValue1 - outputValue2).*s, 1)/size(outputValue1, 1);
                        
                        case self.ACTIVATION_LOSS_FUNCTION_LINEAR_CROSS_ENTROPY()
                            error('Not implemented');
                    end
                    
                end
                
                switch char(self.activationFunction(i))
                    case self.ACTIVATION_FUNCTION_SIGMOID()
                        dActivationFunction = self.layerValue{i + 1} .* (1 - self.layerValue{i + 1});
                    case self.ACTIVATION_FUNCTION_TANH()
                        error('Not implemented');
                    
                    case self.ACTIVATION_FUNCTION_RELU()
                        error('Not implemented');
                    
                    case self.ACTIVATION_FUNCTION_LINEAR()
                        dActivationFunction = 1;
                    
                    case self.ACTIVATION_FUNCTION_SOFTMAX()
                        error('Not implemented');
                end
                
                if i == self.nHiddenLayers
                        z     = dW{i + 1} * self.outputWeight;
                        dW{i} = z .* dActivationFunction;
                        db{i} = sum(dW{i})/size(dW{i}, 1);
                else
                        z     = dW{i + 1} * self.weight{i + 1};
                        dW{i} = z .* dActivationFunction;
                        db{i} = sum(dW{i})/size(dW{i}, 1);
                end
                
            end
            
            self.outputGradient     = dW{end}' * self.layerValue{end};
            self.outputBiasGradient = db{end};
            for i = 1 : self.nHiddenLayers
                self.gradient{i}     = dW{i}' * self.layerValue{i};
                self.biasGradient{i} = db{i};
            end
        end
        
        function test(self, X, y)
            %test
            %   Test the neural network, getting its output by an ensemble
            %   composed of a selected numbers of outputLayers.
            %   It also has the ability to update the importance weight of
            %   each output layer, if necessary.
            %   X (matrix)
            %       Input matrix
            %   y (matrix)
            %       Target matrix
            
            self.feedforward(X, y);
            
            m = size(y, 1);
            [~, self.trueClasses] = max(y, [], 2);
            
            self.sigma = self.outputLayerValue;
            [rawOutput, outputtedClasses] = max(self.sigma, [], 2);
            self.misclassifications = find(outputtedClasses ~= self.trueClasses);
            self.classificationRate = 1 - numel(self.misclassifications) / m;
            self.residualError = 1 - rawOutput;
            self.outputedClasses = outputtedClasses;
        end
        
        function train(self, X, y, weightNo)
            %train
            %   Train the neural network performing 3 complete stages:
            %       - Feed-forward
            %       - Back-propagation
            %       - Weight updates
            %   X (matrix)
            %       Input matrix
            %   y (matrix)
            %       Target matrix
            %   weightNo (integer) [optional]
            %       You has the ability to define which weight and bias you
            %       want to update using backpropagation. This method will
            %       update only that weight and bias, even if there is
            %       weights and biases on layers before and after that.
            %       The number of the weight and bias you want to update.
            %       Remember that 1 indicates the weight and bias that get
            %       out of the input layer.
            self.feedforward(X,y);
            self.backpropagate();
            
            switch nargin
                case 4
                    self.trainWeight(weightNo);
                case 3
                    for i = self.nHiddenLayers + 1 : -1 : 1
                        self.trainWeight(i);
                    end
            end
        end
        
        function trainWeight(self,weightNo)
            %trainWeight
            %   This methods will only update a set of weights and biases.
            %   Normally you will not call this method directly, but will
            %   the method train as a middle man.
            %   weightNo (integer)
            %       The number of the weight and bias you want to update.
            %       Remember that 1 indicates the weight and bias that get
            %       out of the input layer.
            self.updateWeight(weightNo);
        end
        
        function L1train(self, weightNo)
            switch nargin
                case 4
                    self.L1Update(weightNo);
                case 3
                    for i = self.nHiddenLayers + 1 : -1 : 1
                        self.L1Update(i);
                    end
            end
        end
    end
    %% Similarity For Predictive labels
    methods (Access = public)
        function similarityValue = similarity(self, x1, x2)
            similarityValue = exp(-sum((x1 - x2).^2, 2)/2);
        end
        
        function SimilarityTrain(self, X)
            gradient1 = {};
            bias1 = {};
            gradient2 = {};
            bias2 = {};
            for ii = 1:size(X ,1)
                x1 = X(ii, :);
                x2 = X;
                s = self.similarity(x1, x2);
                self.forwardpass(x1);
                outputValue1 = self.outputLayerValue;
                self.forwardpass(x2);
                outputValue2 = self.outputLayerValue;
                self.backpropagate2(outputValue1, outputValue2, s);           
                outputGradient1 = self.outputGradient;
                outputBiasGradient1 = self.outputBiasGradient;
                for i = self.nHiddenLayers: -1 : 1
                    gradient1{i} = self.gradient{i};
                    bias1{i} = self.bias{i};
                end
                self.forwardpass(x2);
                self.backpropagate2(outputValue1, outputValue2, s);
                outputGradient2 = self.outputGradient;
                outputBiasGradient2 = self.outputBiasGradient;
                for i = self.nHiddenLayers: -1 : 1
                    gradient2{i} = self.gradient{i};
                    bias2{i} = self.bias{i};
                end
                for i = self.nHiddenLayers + 1 : -1 : 1
                    if i > self.nHiddenLayers
                        dW = self.learningRate .* (outputGradient1 - outputGradient2);
                        db = self.learningRate' .* (outputBiasGradient1 - outputBiasGradient2);
                        self.outputWeight = self.outputWeight - dW;
                        self.outputBias   = self.outputBias   - db;
                    else
                        dW = self.learningRate.* (gradient1{i} - gradient2{i});
                        db = self.learningRate' .* (bias1{i} - bias2{i});
                        self.weight{i} = self.weight{i} - dW;
                        self.bias{i}   = self.bias{i}   - db;
                    end
                end
            end
        end
    end
    %% Standard Neural Network private methods
    methods (Access = private)
        function updateWeight(self, weightNo)
            %updateWeights
            % Perform weight and bias update into a single weight and bias
            %   weightNo (integer)
            %       Number/Position of the weight/bias you want to update
            w = weightNo; %readability
            if w > self.nHiddenLayers
                dW = self.learningRate .* self.outputGradient;
                db = self.learningRate' .* self.outputBiasGradient;
                if self.momentumRate > 0
                    self.outputMomentum     = self.momentumRate * self.outputMomentum + dW;
                    self.outputBiasMomentum = self.momentumRate * self.outputBiasMomentum + db;
                    dW = self.outputMomentum;
                    db = self.outputBiasMomentum;
                end
                if true == false
                    self.outputWeight = (1 - self.learningRate * self.lambda) * self.outputWeight - dW;
                    self.outputBias   = (1 - self.learningRate * self.lambda) * self.outputBias   - db;
                    self.outputWeight = (1 - self.learningRate * self.lambda) * self.outputWeight - dW;
                    self.outputBias   = (1 - self.learningRate * self.lambda) * self.outputBias   - db;
                else
                    self.outputWeight = self.outputWeight - dW;
                    self.outputBias   = self.outputBias   - db;
                end
                
            else
                dW = self.learningRate .* self.gradient{w};
                db = self.learningRate' .* self.biasGradient{w};
                if self.momentumRate > 0
                    self.momentum{w}     = self.momentumRate * self.momentum{w} + dW;
                    self.biasMomentum{w} = self.momentumRate * self.biasMomentum{w} + db;
                    dW = self.momentum{w};
                    db = self.biasMomentum{w};
                end
                if true == false
                    self.weight{w} = (1 - self.learningRate * self.lambda) * self.weight{w} - dW;
                    self.bias{w}   = (1 - self.learningRate * self.lambda) * self.bias{w}   - db;
                else
                    self.weight{w} = self.weight{w} - dW;
                    self.bias{w}   = self.bias{w}   - db;
                end
                
            end
        end
        
        function calculateError(self, y)
            %calculateError
            %   Calculates the error.
            %   This method probably will be called by the feedforward
            %   method and seldom will be used standalone.
            m = size(y,1);
            
            %TODO: Add the possibility to input which error function we
            %want to use
            switch self.outputActivationFunctionLossFunction
                case self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE()
                    self.errorValue = y - self.outputLayerValue;
                    self.lossValue  = 1 / 2 * sum(sum(self.errorValue .^ 2)) / m;
                
                case self.ACTIVATION_LOSS_FUNCTION_TANH()
                    error('Not implemented');
                
                case self.ACTIVATION_LOSS_FUNCTION_RELU()
                    error('Not implemented');
                
                case self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY()
                    self.errorValue = y - self.outputLayerValue;
                    self.lossValue  = - sum(sum(y .* log(self.outputLayerValue))) / m;
                
                case self.ACTIVATION_LOSS_FUNCTION_LINEAR_CROSS_ENTROPY()
                    error('Not implemented yet');
            end
        end
    end
    
    %% Standard Neural Network statistical metrics public methods
    methods (Access = public)
        function bias2 = computeNetworkBiasSquare(self, y)
            %computeNetworkBias
            %   Compute the Network Squared Bias in relation to a target
            %
            %   y (vector)
            %       A single target
            %   agmm (object)
            %       AGMM object
            %
            %   Returns
            %       The squared bias of the network related to this target
            dataMean = self.dataMean;
            dataStd  = self.dataStd;
            dataVar  = self.dataVar;
            self.nSamplesFeed = self.nSamplesFeed + 1;
            [~, ~, Ez, ~] = self.computeExpectedValues(self.nHiddenLayers + 1);
            bias2 = self.computeBIAS2(Ez, y);
            self.nSamplesFeed = self.nSamplesFeed - 1;
            self.dataMean = dataMean;
            self.dataStd  = dataStd;
            self.dataVar  = dataVar;
        end
        
        function var = computeNetworkVariance(self)
            %computeNetworkVariance
            %   Compute the Network Variance in relation to a target
            %
            %   agmm (object)
            %       AGMM object
            %
            %   Returns
            %       The squared bias of the network related to this target
            dataMean = self.dataMean;
            dataStd  = self.dataStd;
            dataVar  = self.dataVar;
            self.nSamplesFeed = self.nSamplesFeed + 1;
            [~, ~, Ez, Ez2] = self.computeExpectedValues(self.nHiddenLayers + 1);
            var = self.computeVAR(Ez, Ez2);
            self.nSamplesFeed = self.nSamplesFeed - 1;
            self.dataMean = dataMean;
            self.dataStd  = dataStd;
            self.dataVar  = dataVar;
        end
    end
    %% Standard Neural Networks extra public methods
    methods (Access = public)
         function distance = L2Distance(self, X, Y)
            distance = sqrt(sum((X - Y).^ 2));
        end
        
        function value = empiricalExpectation(self, X)           
           value = sum(X, 1) ./ size(X, 1);
        end
         function loss = updateWeightsByCMD(self, Xs, Xt, k)
            %updateWeightsByKullbackLeibler
            %   This method is used on Transfer Learning procedures. The
            %   idea is to approximate the source and target distributions.
            %   Xs (matrix)
            %       Source input
            %   Xt (matrix)
            %       Target input
            %   k (integer)
            %       k_th order sample central moments 
            if size(Xs,1) ~= size(Xt, 1)
                cutSize = min(size(Xs, 1), size(Xt, 1));
                Xs = Xs(1:cutSize, :);
                Xt = Xt(1:cutSize, :);
            end
            nHL = self.nHiddenLayers + 1;
            self.forwardpass(Xs);
            sourceInput = self.layerValue{1};
            sourceHiddenValue = self.layerValue{2};
            self.forwardpass(Xt);
            targetInput = self.layerValue{1};
            targetHiddenValue = self.layerValue{2};
            
            % record the compact interval
            b = 1;
            a = 0;
            
                % backpropagation
                
                
                EX = (sourceHiddenValue .* (1 - sourceHiddenValue))' * sourceInput .* 1/size(sourceHiddenValue, 1);
                EY = (targetHiddenValue .* (1 - targetHiddenValue))' * targetInput .* 1/size(sourceHiddenValue, 1);
                dW = 1/(2*self.L2Distance(self.empiricalExpectation(sourceHiddenValue), self.empiricalExpectation(targetHiddenValue))) .*...
                    2 .* (self.empiricalExpectation(sourceHiddenValue) - self.empiricalExpectation(targetHiddenValue))' .* ...
                    (EX - EY);
                EX2 = sum(sourceHiddenValue .* (1 - sourceHiddenValue)) .* 1/size(sourceHiddenValue, 1);
                EY2 = sum(targetHiddenValue .* (1 - targetHiddenValue)) .* 1/size(targetHiddenValue, 1);
                db = 1/(2*self.L2Distance(self.empiricalExpectation(sourceHiddenValue), self.empiricalExpectation(targetHiddenValue))) .*...
                    2 .* (self.empiricalExpectation(sourceHiddenValue) - self.empiricalExpectation(targetHiddenValue)) .* ...
                    (EX2 - EY2);
                dW = dW / self.L2Distance(b, a);
                db = db / self.L2Distance(b, a);
                
                dW_temp = 0;
                db_temp = 0;
                S = 0;
                Sb = 0;       
                T = 0;
                Tb = 0;
                for i = 2:k
                    for j = 1 : size(Xs ,1)
                        termS = (i * (sourceHiddenValue - self.empiricalExpectation(sourceHiddenValue)).^(i-1));
                        term2S = (termS(j, :))' .* ((sourceHiddenValue(j, :) .* (1 - sourceHiddenValue(j, :)))' * sourceInput(j, :) - EX);
                        term3S = termS(j, :) .* sum(sourceHiddenValue(j, :) .* (1 - sourceHiddenValue(j, :)),1);
                        S = S + term2S;
                        Sb = Sb + term3S;
                        termT = (i * (targetHiddenValue - self.empiricalExpectation(targetHiddenValue)).^(i-1));
                        term2T = (termT(j, :))' .* ((targetHiddenValue(j, :) .* (1 - targetHiddenValue(j, :)))' * targetInput(j, :) - EY);
                        term3T = termT(j, :) .* sum(targetHiddenValue(j, :) .* (1 - targetHiddenValue(j, :)),1);
                        T = T + term2T;
                        Tb = Tb + term3T;
                    end
                    S = S / size(Xs, 1);
                    Sb = Sb/size(Xs, 1);
                    T = T / size(Xt, 1);
                    Tb = Tb / size(Xt, 1);
                    dW_temp = dW_temp + 1/self.L2Distance(a, b)^i .* 1/(2*self.L2Distance((self.empiricalExpectation(sourceHiddenValue - self.empiricalExpectation(sourceHiddenValue)).^ i)...
                        , self.empiricalExpectation((targetHiddenValue - self.empiricalExpectation(targetHiddenValue)).^i))) .* ...
                        2 * ((self.empiricalExpectation(sourceHiddenValue - self.empiricalExpectation(sourceHiddenValue)).^ i)...
                        - self.empiricalExpectation((targetHiddenValue - self.empiricalExpectation(targetHiddenValue)).^i))' .* ...
                        (S - T);
                    db_temp = db_temp + 1/self.L2Distance(a, b)^i .* 1/(2*self.L2Distance((self.empiricalExpectation(sourceHiddenValue - self.empiricalExpectation(sourceHiddenValue)).^ i)...
                        , self.empiricalExpectation((targetHiddenValue - self.empiricalExpectation(targetHiddenValue)).^i))) .* ...
                        2 * ((self.empiricalExpectation(sourceHiddenValue - self.empiricalExpectation(sourceHiddenValue)).^ i)...
                        - self.empiricalExpectation((targetHiddenValue - self.empiricalExpectation(targetHiddenValue)).^i)) .* ...
                        (Sb - Tb);
                end
                dW = dW + dW_temp;
                db = db + db_temp;
                self.weight{1} = self.weight{1} - self.learningRate * dW;
                self.bias{1} = self.bias{1} - self.learningRate * db;
                
                % compute loss
                self.forwardpass(Xs);
                sourceX = self.layerValue{nHL};
                self.forwardpass(Xt);
                targetX = self.layerValue{nHL};
                loss = 1/self.L2Distance(b, a) * self.L2Distance(self.empiricalExpectation(Xs),...
                    self.empiricalExpectation(Xt));
                for i = 2: k
                    loss = loss + 1/self.L2Distance(b, a)^i * self.L2Distance(self.empiricalExpectation((Xs - self.empiricalExpectation(Xs)) .^ i),...
                        self.empiricalExpectation((Xt -self.empiricalExpectation(Xt)) .^ i));
                end

        end
    end
    
    %% Elastic/Evolving Neural Network public methods
    methods (Access = public)        
        function widthAdaptationStepwise(self, y, sg)
            %widthAdaptationStepwise
            %   Performs network width adaptation in a specific layer,
            %   stepwise (it means that it execute one row at a time).
            %   Also, this method assume that you already passe the input
            %   data through the model via forwardpass procedure.           
            %   sg(cell)
            %       Estimated Gaussian distribution for this domain.
            %   y (double or vector)
            %       Double, if you are performing regression
            %       Vector if you are performing classification
            %       The targer data to be used as validation
            
            
            [Ex, ~, Ey, Ey2] = computeExpectedValues(self, sg);
            
            bias2 = self.computeBIAS2(Ey, y);
            var   = self.computeVAR(Ey, Ey2);
            
            [self.meanBIAS, self.varBIAS, self.stdBIAS] ...
                    = self.util.recursiveMeanStd(bias2, self.meanBIAS, self.varBIAS, sg.Supm);
                
            [self.meanVAR, self.varVAR, self.stdVAR] ...
                    = self.util.recursiveMeanStd(var, self.meanVAR, self.varVAR, sg.Supm);
                
            if sg.Supm <= 1 || self.growable(1) == true
                self.minMeanBIAS = self.meanBIAS;
                self.minStdBIAS  = self.stdBIAS;
            else
                self.minMeanBIAS = min(self.minMeanBIAS, self.meanBIAS);
                self.minStdBIAS  = min(self.minStdBIAS, self.stdBIAS);
            end
            
            if sg.Supm <= self.inputSize + 1 || self.prunable(1) ~= 0
                self.minMeanVAR = self.meanVAR;
                self.minStdVAR  = self.stdVAR;
            else
                self.minMeanVAR = min(self.minMeanVAR, self.meanVAR);
                self.minStdVAR  = min(self.minStdVAR, self.stdVAR);
            end
            
%             self.BIAS2{nhl} = [self.BIAS2{nhl} self.meanBIAS(nhl)];
%             self.VAR{nhl}   = [self.VAR{nhl} self.meanVAR(nhl)];
            self.growable = self.isGrowable(bias2, sg);
            self.prunable = self.isPrunable(var, Ex, self.PRUNE_SINGLE_LEAST_CONTRIBUTION_NODES(), sg);          
        end 
        
        function grow(self, layerNo)
            %grow
            %   Add 1 new node to a hidden layer. Because of this, it will
            %   add 1 extra weight and bias at the outbound row and 1 extra
            %   weight at the inbound row.
            %   layerNo (integer)
            %       Number of the layer you want to add a node.
            self.layers(layerNo) = self.layers(layerNo) + 1;
            if layerNo > 1
                self.growWeightRow(layerNo - 1)
                self.growBias(layerNo - 1);
            end
            if layerNo < numel(self.layers)
                self.growWeightColumn(layerNo)
            end
        end
        
        function prune(self, layerNo, nodeNo)
            %prune
            %   Remove 1 node from the hidden layer. Because of this, it
            %   will remove 1 weight and bias at the outbound row and 1
            %   weight from the inbound row.
            %   layerNo (integer)
            %       Number of the layer you want to add a node.
            %   nodeNo (integer)
            %       Position of the node to be removed
            self.layers(layerNo) = self.layers(layerNo) - 1;
            if layerNo > 1
                self.pruneWeightRow(layerNo - 1, nodeNo);
                self.pruneBias(layerNo - 1, nodeNo);
            end
            if layerNo < numel(self.layers)
                self.pruneWeightColumn(layerNo, nodeNo);
            end
        end
    end
    
    %% Elastic/Evolving Neural Network protected methods
    methods (Access = protected) 
        function isGrowable = isGrowable(self, BIAS2, sg)
            %isGrowable
            %   Evaluate if a specific layer need a node added to have its
            %   network significance parameters stable
            %   layerNo (integer)
            %       Layer which the evaluation will be performed. Usually
            %       it is a hidden layer.
            %   BIAS2 (double)
            %       The squished BIAS2 of that layer at that time
            %
            %   returns a boolean indicating if that layer is ready to
            %   receive a new node or not.
            isGrowable = false;
            ALPHA_1 = 1.25;
            ALPHA_2 = 0.75;
            
            current    = (self.meanBIAS + self.stdBIAS);
            biased_min = (self.minMeanBIAS...
                       + (ALPHA_1 * exp(-BIAS2) + ALPHA_2)...
                       * self.minStdBIAS);
            
            if sg.Supm > 1 && norm(current) >= norm(biased_min)
                isGrowable = true;
            end
        end
        
        function prunableNodes = isPrunable(self, VAR, expectedY, option, sg)
            %isPrunable
            %   Evaluate if a specific layer need a node pruned to have its
            %   network significance parameters stable
            %   layerNo (integer)
            %       Layer which the evaluation will be performed. Usually
            %       it is a hidden layer.
            %   VAR (double)
            %       The squished VAR of that layer at that time
            %   expectedY (vector)
            %       See self.getExpectedValues
            %       This value is used to determine the node with minimum
            %       contribution to the network.
            %   option (string)
            %       'least_contribution': In case the pruning rule get
            %       approved, it will return the position for the least
            %       contributing node.
            %       'below_contribution': In case the pruning rule get
            %       approved, it will return an array with the position for
            %       all nodes that have the contribution below a certain
            %       quantity
            %
            %   returns a integer indicating the position of which node
            %   should be removed from that layer. If no node should be
            %   removed, returns zero instead.
            prunableNodes = 0;
            ALPHA_1 = 2.5;
            ALPHA_2 = 1.5;
            
            current = (self.meanVAR + self.stdVAR);
            biased_min = (self.minMeanVAR...
                       + (ALPHA_1 * exp(-VAR) + ALPHA_2)...
                       * self.minStdVAR);
            
            if self.growable == false ...
                    && sg.Supm > self.inputSize + 1 ...
                    && norm(current) >= norm(biased_min)
                
                switch option
                    case self.PRUNE_SINGLE_LEAST_CONTRIBUTION_NODES()
                        [~, prunableNodes] = min(expectedY);
                    case self.PRUNE_MULTIPLE_NODES_WITH_CONTRIBUTION_BELOW_EXPECTED()
                        nodesToPrune = expectedY <= abs(mean(expectedY) - var(expectedY));
                        if sum(nodesToPrune)
                            prunableNodes = find(expectedY <= abs(mean(expectedY) - var(expectedY)));
                        else
                            [~, prunableNodes] = min(expectedY);
                        end
                end
            end
        end
        
        function growWeightRow(self, weightArrayNo)
            %growWeightRow
            %   Add 1 extra weight at the inbound row.
            %   weightArrayNo (integer)
            %       Weight position
            w = weightArrayNo; % readability
            if w > numel(self.weight)
                [n_in, n_out] = size(self.outputWeight);
                n_in = n_in + 1;
                self.outputWeight = [self.outputWeight; normrnd(0, sqrt(2 / (n_in)), [1, n_out])];
                self.outputMomentum = [self.outputMomentum; zeros(1, n_out)];
            else               
                [n_in, n_out] = size(self.weight{w});
                n_in = n_in + 1;
                self.weight{w} = [self.weight{w}; normrnd(0, sqrt(2 / (n_in)), [1, n_out])];
                self.momentum{w} = [self.momentum{w}; zeros(1, n_out)];
            end
        end
        
        function growWeightColumn(self, weightArrayNo)
            %growWeightColumn
            %   Add 1 extra weight at the outbound column.
            %   weightArrayNo (integer)
            %       Weight position
            w = weightArrayNo; % readability
            if w > numel(self.weight)
                [n_out, n_in] = size(self.outputWeight);
                n_in = n_in + 1;
                self.outputWeight = [self.outputWeight normrnd(0, sqrt(2 / (n_in)), [n_out, 1])];
                self.outputMomentum = [self.outputMomentum zeros(n_out, 1)];
            else              
                [n_out, n_in] = size(self.weight{w});
                n_in = n_in + 1;
                self.weight{w} = [self.weight{w} normrnd(0, sqrt(2 / (n_in)), [n_out, 1])];
                self.momentum{w} = [self.momentum{w} zeros(n_out, 1)];
            end
        end
        
        function pruneWeightRow(self, weightNo, nodeNo)
            %pruneWeightRow
            %   Remove 1 weight from the inbound row.
            %   weightNo (integer)
            %       Weight position
            %   nodeNo (integer)
            %       Position of the node to be removed
            w = weightNo; % readability
            n = nodeNo;   %readability
            if w > numel(self.weight)
                self.outputWeight(n, :)   = [];
                self.outputMomentum(n, :) = [];
            else
                self.weight{w}(n, :)   = [];
                self.momentum{w}(n, :) = [];
            end
        end
        
        
        function L1Update(self, weightNo)
            w = weightNo; %readability
            beta = 0.01;
            if w > self.nHiddenLayers
                dW = beta * self.learningRate .* sgn(self.outputWeight);
                db = beta * self.learningRate' .* sgn(self.outputBias);              
                if true == false
                    self.outputWeight = (1 - self.learningRate * self.lambda) * self.outputWeight - dW;
                    self.outputBias   = (1 - self.learningRate * self.lambda) * self.outputBias   - db;
                    self.outputWeight = (1 - self.learningRate * self.lambda) * self.outputWeight - dW;
                    self.outputBias   = (1 - self.learningRate * self.lambda) * self.outputBias   - db;
                else
                    self.outputWeight = self.outputWeight - dW;
                    self.outputBias   = self.outputBias   - db;
                end
                
            else
                dW = beta * self.learningRate .* sgn(self.weight{w});
                db = beta * self.learningRate' .* sgn(self.bias{w});
               
                if true == false
                    self.weight{w} = (1 - self.learningRate * self.lambda) * self.weight{w} - dW;
                    self.bias{w}   = (1 - self.learningRate * self.lambda) * self.bias{w}   - db;
                else
                    self.weight{w} = self.weight{w} - dW;
                    self.bias{w}   = self.bias{w}   - db;
                end               
            end
        end
         function L1Update2(self, weightNo)
            w = weightNo; %readability
            beta = 0.01;
            if w > self.nHiddenLayers
                dW = beta * self.learningRate .* sgn(self.outputWeight - self.outputWeightInitial);
                db = beta * self.learningRate' .* sgn(self.outputBias - self.outputBiasInitial);              
                if true == false
                    self.outputWeight = (1 - self.learningRate * self.lambda) * self.outputWeight - dW;
                    self.outputBias   = (1 - self.learningRate * self.lambda) * self.outputBias   - db;
                    self.outputWeight = (1 - self.learningRate * self.lambda) * self.outputWeight - dW;
                    self.outputBias   = (1 - self.learningRate * self.lambda) * self.outputBias   - db;
                else
                    self.outputWeight = self.outputWeight - dW;
                    self.outputBias   = self.outputBias   - db;
                end
                
            else
                dW = beta * self.learningRate .* sgn(self.weight{w} - self.weightInitial{w});
                db = beta * self.learningRate' .* sgn(self.bias{w} - self.biasInitial{w});
               
                if true == false
                    self.weight{w} = (1 - self.learningRate * self.lambda) * self.weight{w} - dW;
                    self.bias{w}   = (1 - self.learningRate * self.lambda) * self.bias{w}   - db;
                else
                    self.weight{w} = self.weight{w} - dW;
                    self.bias{w}   = self.bias{w}   - db;
                end               
            end
        end
        
        function pruneWeightColumn(self, weightNo, nodeNo)
            %pruneWeightColumn
            %   Remove 1 weight from the outbound column
            %   weightArrayNo (integer)
            %       Weight position
            %   nodeNo (integer)
            %       Position of the node to be removed
            w = weightNo; % readability
            n = nodeNo;   %readability
            if w > numel(self.weight)
                self.outputWeight(:, n)   = [];
                self.outputMomentum(:, n) = [];
            else
                self.weight{w}(:, n)   = [];
                self.momentum{w}(:, n) = [];
            end
        end
        
        function growBias(self, biasArrayNo)
            %growBias
            %   Add 1 extra bias at the inbound row.
            %   biasArrayNo (integer)
            %       Bias position
            b = biasArrayNo; %readability
            if b > numel(self.weight)
                self.outputBias         = [self.outputBias normrnd(0, sqrt(2 / (self.layers(end) + 1)))];
                self.outputBiasMomentum = [self.outputBiasMomentum 0];
            else
                self.bias{b} = [self.bias{b} normrnd(0, sqrt(2 / (self.layers(b) + 1)))];
                self.biasMomentum{b} = [self.biasMomentum{b} 0];
            end
        end
        
        function pruneBias(self, biasArrayNo, nodeNo)
            %pruneBias
            %   Remove 1 bias from the inbound row.
            %   biasArrayNo (integer)
            %       Bias position
            %   nodeNo (integer)
            %       Position of the node to be removed
            b = biasArrayNo; % readability
            n = nodeNo;   %readability
            if b > numel(self.weight)
                self.outputBias(n) = [];
                self.outputBiasMomentum(n) = [];
            else
                self.bias{b}(n) = [];
                self.biasMomentum{b}(n) = [];
            end
        end
        
        function [Ex, Ex2, Ez, Ez2] = computeExpectedValues(self, sg)
            %computeExpectedValues
            %   Compute statisticals expectations values for a specific
            %   hidden layer
            %
            %   Returns Ex  = Expected value of that layer
            %           Ex2 = Expected squared value of that layer
            %           Ez  = Expected outbound value of that layer
            %           Ez2 = Expected outbound squared value of that layer
            [Ex, Ex2] = computeInboundExpectedValues(self, sg);
            [Ez, Ez2] = computeOutboundExpectedValues(self, Ex, Ex2);
        end
        
        function [Ex, Ex2] = computeInboundExpectedValues(self, sg)
            %computeInboundExpectedValues
            %   Compute statisticals expectations values for a specific
            %   hidden layer
            %   nHiddenLayer (integer)
            %       layer to be evaluated
            %
            %   Returns Ex  = Expected value of that layer
            %           Ex2 = Expected squared value of that layer
            center = sg.mu;
            std = sqrt(sg.var);           
            py = self.util.probit(center, std);
            Ex = sigmf(self.weight{1} * py' + self.bias{1}', [1, 0]);
            Ex2 = Ex .^ 2;
        end
         
        
        function [Ez, Ez2] = computeOutboundExpectedValues(self, Ex, Ex2)
            %computeOutboundExpectedValues
            %   Compute statisticals expectations values for a specific
            %   hidden layer
            %   Ey (double, vector or matrix)
            %       Expected value
            %   Ey2 (double, vector or matrix)
            %       Expected squared value
            %
            %   Returns Ez  = Expected outbound value of that layer
            %           Ez2 = Expected outbound squared value of that layer
            Ez = self.outputWeight * Ex + self.outputBias';
            Ez = exp(Ez - max(Ez));
            Ez = Ez ./ sum(Ez);

            Ez2 = self.outputWeight * Ex2 + self.outputBias';
            Ez2 = exp(Ez2 - max(Ez2));
            Ez2 = Ez2 ./ sum(Ez2);
        end
        
        function NS = computeNetworkSignificance(self, Ez, Ez2, y)
            %computeNetworkSignificance
            %   Compute the current Network Significance of the model in
            %   respect to a target
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   Ez2 (double, vector or matrix)
            %       Expected outbound squared value of that layer
            %   y (double, vector or matrix)
            %       A target class
            %
            %   return NS = The network significance
            NS = self.computeBIAS2(Ez, z) + self.computeVAR(Ez, Ez2);
        end
        
        function BIAS2 = computeBIAS2(~, Ez, y)
            %computeBIAS2
            %   Compute the current BIAS2 of the model wrt a target
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   y (double, vector or matrix)
            %       A target class
            %
            %   return BIAS2 = The network squared BIAS
            BIAS2 = norm((Ez - y') .^2 , 'fro');
        end
        
        function VAR = computeVAR(~, Ez, Ez2)
            %computeVAR
            %   Compute the current VAR of the model
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   Ez2 (double, vector or matrix)
            %       Expected outbound squared value of that layer
            %
            %   return VAR = The network VAR (variance)
            VAR = norm(Ez2 - Ez .^ 2, 'fro');
        end        
    end
    %% GIVE A NAME TO THIS SECTION
    methods (Access = public)
        function agmm = runAgmm(self, x, y)
            
            bias2 = self.computeNetworkBiasSquare(y);
            
            self.agmm.run(x, bias2);
            
            agmm = self.agmm;
        end
    end
    
    %% Getters and Setters
    methods (Access = public)
        function setAgmm(self, agmm)
            %setAgmm
            %   You can use this method to set your own AGMM to this
            %   network
            %   agmm (AGMM)
            %       The AGMM you want to set to this network.
            self.isAgmmAble = true;
            self.agmm = agmm;
        end
        
        function agmm = getAgmm(self)
            %getAgmm
            %   Gets the agmm that the network is using. If the network has
            %   an empty agmm or is not using a agmm, it will enable AGMM
            %   and return to you a new AGMM
            if isempty(self.agmm) || self.isAgmmAble == false
                self.enableAgmm();
            end
            agmm = self.agmm;
        end
        
        function enableAgmm(self)
            %enableAgmm
            %   Tell the network that it will use AGMM from now on
            %   It also creates a random AGMM. If you want to use your own
            %   AGMM, make sure to use setAgmm method afterwards
            self.isAgmmAble = true;
            self.agmm = AGMM();
        end
        
        function disableAgmm(self)
            %disableAgmm
            %   Tell the network that it will NOT use AGMM frmo now on.
            %   It deletes the agmm that was attached to this model. If you
            %   want to keep track of that agmm, make sure to load it into
            %   some variable using the getAgmm method.
            self.isAgmmAble = false;
            self.agmm = [];
        end
        
        function nHiddenLayers = getNumberHiddenLayers(self)
            %getNumberHiddenLayers
            %   Return the number of hidden layers in the network
            %
            %   Returns
            %       nHiddenLayers (integer): Number of hidden layers
            nHiddenLayers = self.nHiddenLayers;
        end
    end
end

