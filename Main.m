clc
clear
addpath modelPackages
dm = DataManipulator('./data');
%%--------dataset---------%%
% dm.loadWeatherDataset();
% dm.loadSeaDataset();
% dm.loadHyperplaneDataset();
% dm.loadOccupancyDataset();
% dm.loadKDDDataset();
% dm.loadForestCoverTypeDataset();
% dm.loadSUSYDataset();
dm.loadKittiDataset();
% dm.loadHepmassDataset();
%%--------dataset---------%%
dm.normalize();
epochs = 15;
nRun = 1;
AOMDA
