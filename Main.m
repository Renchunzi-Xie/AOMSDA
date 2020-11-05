clc
clear
addpath modelPackages
dm = DataManipulator('./data');
%%--------dataset---------%%
dm.loadWeatherDataset();
% dm.loadSeaDataset();
% dm.loadHyperplaneDataset();
% dm.loadKDDDataset();
% dm.loadSUSYDataset();
% dm.loadKittiDataset();
% dm.loadHepmassDataset();
%%--------dataset---------%%
dm.normalize();
epochs = 1;
nRun = 5;
AOMSDA
