DataPath = '.\DataBase\';
DataName = 'Yale_32x32.mat'; d = 32; m =1.08; lambda = 1; p = [1 , 1]; gamma_ratio= 0; h = 0.05; k_ne = 60;
load([DataPath,DataName]);
[Final,Result] = MFSC(fea,gnd,d,m,lambda,gamma_ratio,p,h, k_ne);
