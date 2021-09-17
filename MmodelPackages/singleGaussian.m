% MIT License
%
% Copyright (c) 2019
% Renchunzi Xie
% renchunzi.xie@ntu.edu.sg
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
classdef singleGaussian < handle
    %singleGaussian
    %   It estimates input features as a single Gaussian distribution model.
    properties(Access = public)
        mu;   %mean of the distribution
        var;  %variance of the distribution
        nFeatures;
        Supm;
    end
    
    properties(Access = protected)
        newOmiga = 0;   %Present parameter of mu
        oldOmiga = 0;   %Sum of omiga until the last one
        sumOmiga = 0;   %Sum of omiga until present
        oldMu           %The last mean of the distribution
       

    end
    
    methods (Access = public)
        function runGM(self, x)
            if length(self.mu) == 0
                self.mu = x;
                self.nFeatures = size(x,2);
                self.var = 0.01 * ones(1, self.nFeatures);
                self.Supm = 1;
            end           
            self.oldMu = self.mu;
            self.Supm = self.Supm + 1;
            self.meanEvolvement(x, self.Supm);
            self.varEvolvement(x, self.Supm);
        end
    end
    
    methods(Access = private)
        function varEvolvement(self, x, Supm)
            self.var = self.var + (x-self.oldMu).*(x-self.mu)/Supm;
        end
            
        function meanEvolvement(self, x, Supm)
            self.oldMu = self.mu;
            self.mu = self.mu + 1/Supm*(x-self.mu);
        end
        
        
    end
end