close all
clear all
clc

default_path = '/Users/akira/Documents/Github/python-code';
save_path = '/Volumes/DATA2/SDN_Data';

Fs = 1000;
force_levels_vec = 10:10:100;

pxx_all = zeros(10,501);
meanForce_all = zeros(length(force_levels_vec),10);
std_all = zeros(length(force_levels_vec),10);
CoV_all = zeros(length(force_levels_vec),10);
meanForce_all_2 = zeros(length(force_levels_vec),10);
std_all_2 = zeros(length(force_levels_vec),10);
CoV_all_2 = zeros(length(force_levels_vec),10);
for j = 1:length(force_levels_vec)
    force_level = force_levels_vec(j);
    
    cd (save_path)
    load(['Force_noTendon_0_1_0_' num2str(force_level) '.mat'])
    Force_1 = Force;
%     load(['Force_' num2str(force_level) '.mat'])
%     Force_2 = Force;
    cd (default_path)
    
    for i = 1:10
        Force_1_temp = Force_1(i,:);
        [pxx,f] = pwelch(Force_1_temp-mean(Force_1_temp),gausswin(2024*Fs/1000),1024*Fs/1000,0:0.1:50,Fs,'power');
        pxx_all(i,:) = pxx;
        
        meanForce(i) = mean(Force_1_temp);
        stdForce(i) = std(Force_1_temp);
        CoVForce(i) = stdForce(i)/meanForce(i);
        
%         Force_2_temp = Force_2(i,:);
%         meanForce_2(i) = mean(Force_2_temp);
%         stdForce_2(i) = std(Force_2_temp);
%         CoVForce_2(i) = stdForce_2(i)/meanForce_2(i);
    end
    
    meanForce_all(j,:) = meanForce; 
    std_all(j,:) = stdForce; 
    CoV_all(j,:) = CoVForce; 
    
%     meanForce_all_2(j,:) = meanForce_2; 
%     std_all_2(j,:) = stdForce_2; 
%     CoV_all_2(j,:) = CoVForce_2; 
    
    figure(1)
    plot(f,mean(pxx_all))
    hold on
end

figure(2)
plot(mean(meanForce_all,2))
% hold on 
% plot(mean(meanForce_all_2,2))
title('Mean Force (N)')


figure(3)
boxplot(std_all')
ylabel('SD (N)')
%title('Constant Noise')
% figure(4)
% boxplot(std_all_2')
% ylabel('SD')
% title('Signal Dependent Noise')
% hold on 
% boxplot(std_all_2')

figure(5)
boxplot(CoV_all'*100)
ylabel('CoV (%)')
%title('Constant Noise')
% figure(6)
% boxplot(CoV_all_2')
% ylabel('CoV')
% title('Signal Dependent Noise')

