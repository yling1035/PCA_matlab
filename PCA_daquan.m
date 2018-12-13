clc;
clear;
close all;
load class_data.mat;
%%%取出测试集和训练集的X数据
train_data=xx(:,1:10000);
train_label=xx(:,10001);
test_data=yy(:,1:10000);
test_label=yy(:,10001);
%%%%%%主成分提取
[COEFF,SCORE,latent]=princomp(train_data);
explained=100*latent/sum(latent);
[msc_m,msc_n]=size(train_data);
result1=cell(msc_n+1,4);
result1(1,:)={'特征值','差值','贡献率','累计贡献率'};
result1(2:end,1)=num2cell(latent);
result1(2:end-1,2)=num2cell(-diff(latent));
result1(2:end,3:4)=num2cell([explained,cumsum(explained)]);
%提取累计贡献率到达90%以上的特性
a=find(cumsum(explained)>90);
b=a(1,:);
pc=COEFF(:,1:b);
% 将原始数据进行降维处理
train_date_pca=train_data*pc;
test_date_pca=test_data*pc;
