clc;
clear;
close all;
load class_data.mat;
%%%ȡ�����Լ���ѵ������X����
train_data=xx(:,1:10000);
train_label=xx(:,10001);
test_data=yy(:,1:10000);
test_label=yy(:,10001);
%%%%%%���ɷ���ȡ
[COEFF,SCORE,latent]=princomp(train_data);
explained=100*latent/sum(latent);
[msc_m,msc_n]=size(train_data);
result1=cell(msc_n+1,4);
result1(1,:)={'����ֵ','��ֵ','������','�ۼƹ�����'};
result1(2:end,1)=num2cell(latent);
result1(2:end-1,2)=num2cell(-diff(latent));
result1(2:end,3:4)=num2cell([explained,cumsum(explained)]);
%��ȡ�ۼƹ����ʵ���90%���ϵ�����
a=find(cumsum(explained)>90);
b=a(1,:);
pc=COEFF(:,1:b);
% ��ԭʼ���ݽ��н�ά����
train_date_pca=train_data*pc;
test_date_pca=test_data*pc;
