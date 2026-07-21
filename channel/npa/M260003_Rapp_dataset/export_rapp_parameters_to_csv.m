% export_rapp_parameters_to_csv.m
clear; clc;

load("AMAM_Rapp_model_parameters.mat");
load("AMPM_Rapp_model_parameters.mat");

whos

% 如果变量名就是 AMAM_Rapp_model_parameters / AMPM_Rapp_model_parameters
writetable(AMAM_Rapp_model_parameters, "AMAM_Rapp_model_parameters.csv");
writetable(AMPM_Rapp_model_parameters, "AMPM_Rapp_model_parameters.csv");

disp("Export finished.");