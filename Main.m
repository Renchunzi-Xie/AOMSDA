clc
clear
addpath MmodelPackages
dm = DataManipulator('./data');
%%--------dataset---------%%        
dm.loadWeatherDataset();   
% dm.loadSeaDataset();    
% dm.loadHyperplaneDataset();
% dm.loadKDDDataset();
% dm.loadSUSYDataset();      
% dm.loadKittiDataset();
% dm.loadHepmassDataset();
% dm.loadInjordexpDataset();
%%--------dataset---------%%
dm.normalize();
epochs = 1;
nRun = 1;
AOMSDA

