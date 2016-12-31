% This file demonstrates how to call the MNR_ADL_SR2 algorithm to remove
% multiplicative noise in real SAR images

% Parameters:
% n: image patch size
% d: signal dimension
% lambda1: multiplier of the analysis dictioary based regularizer
% lambda2: multiplier of the smoothness regularizer

% References:
% J. Dong, Z. Han, Y. Zhao, W. Wang, A. Prochazka, J. Chambers,
% "Sparse Analysis Model Based Multiplicative Noise Removal with Enhanced Regularization,"
% submitted to Signal Processing, October 2016.
%
% The codes & data have been deposited to https://github.com/jd0710/MNR-ADL-SR
%
% Written by Jing Dong, moderated by Wenwu Wang, version 1.0
%
% If you have any questions or comments regarding this package, or if you want to
% report any bugs or unexpected error messages, please send an e-mail to
% w.wang@surrey.ac.uk
%
% Copyright 2016 J. Dong, Z. Han, Y. Zhao, W. Wang, A. Prochazka, J. Chambers,
%
% This software is a free software distributed under the terms of the GNU
% Public License version 3 (http://www.gnu.org/licenses/gpl.txt). You can
% redistribute it and/or modify it under the terms of this licence, for
% personal and non-commercial use and research purpose.


close all
clear

% patch dimensions
n=8;
d=n^2; 

% load dictionary data
OmegaFileName = 'Omega_LogDom_cleanImgs_DIFInit_cosparsity_100.mat';
load(OmegaFileName, 'Omega');

% imread speckled SAR image
In_3d = double(imread('speckle1.tif'));
In = In_3d(:,:,1);
In = max(In, eps);

% reshape to pachtes 
Xn=im2col(In,[n,n],'sliding');
Xdn=zeros(size(Xn));

Z = log(Xn);
exp_Z = Xn;

[MM,NN]=size(In);
cnt=countcover([MM NN],[n n],[1 1]);

% set lambda1, lambda2
% same parameters as used for Cameraman, L=4
lambda1 = 0.6;
lambda2 = 0.1;

gamma1 = lambda1;
gamma2 = lambda2;

AL_iters = 2000; % iteration number for ADMM
Xdn_prev = Xn;

%% ADMM
Y = Z;
T = Omega*Y;
M = Y;
B1 = zeros(size(T));
B2 = zeros(size(Y));

ReFluct = [];
Idn = In;

px = zeros(size(In));
py = zeros(size(In));

for ii = 1:AL_iters
    %% update Y - gradient descent - 1 iteration
    alpha = 1e-2; % slow convergence
    for k = 1:1
        gY = (1-exp_Z.*exp(-Y)) + gamma1*Omega'*(B1+Omega*Y-T) + ...
            gamma2*(B2+Y-M);
        Y = Y - alpha*gY;
    end
    
    %% update T - softthresholding
    M_softTH = Omega*Y+B1;
    epsilon = lambda1/gamma1;
    T = sign(M_softTH).*max(abs(M_softTH)-epsilon, 0);
    
    %% update M - TV denoising
    % reshpe M as square image and TV denoising
    M_square=col2imstep(M,[MM  NN],[n n]);
    B2_square=col2imstep(B2,[MM NN],[n n]);
    Y_square=col2imstep(Y,[MM NN],[n n]);
    
    M_square = M_square ./ cnt;
    B2_square = B2_square ./ cnt;
    Y_square = Y_square ./ cnt;   
    
    max_iters = 30;
    dt = 0.02;
    [M_square_qv] = qv_denoising(B2_square+Y_square, max_iters, dt,lambda2, gamma2);
    M = im2col(M_square_qv,[n,n],'sliding');
    
    %% update B1
    B1 = B1+(Omega*Y-T);
    
    %% update B2
    B2 = B2+(Y-M);
    
    % reshape exp(Y) to denoised image 
    Xdn = exp(Y);
    Idn=col2imstep(Xdn,[MM NN],[n n]);
    Idn=Idn./cnt;
    Idn=max(0,min(255,Idn));
        
    ReFluct(ii) = norm(Xdn-Xdn_prev, 'fro')/norm(Xdn_prev, 'fro');
    if (ii>2) && (abs(ReFluct(ii)) <=1e-4)
        break;
    end
    Xdn_prev = Xdn;
end

% save results
filename = 'RealSAR_Speckle1_MNR_ADL_SR2_Cameraman_L4_Params';
save(filename, 'Idn','lambda1','lambda2','In');




