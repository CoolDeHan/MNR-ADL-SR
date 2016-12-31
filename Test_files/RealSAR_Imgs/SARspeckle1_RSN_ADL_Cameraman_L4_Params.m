% This file demonstrates how to call the RSN_ADL algorithm to remove
% multiplicative noise in real SAR images

% Parameters:
% n: image patch size
% d: signal dimension
% lambda: multiplier of the analysis dictioary based regularizer

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

rng(1);

% patch dimension
n=8;
d=n^2; 

% load dictionary data
OmegaFileName = 'Omega_LogDom_cleanImgs_DIFInit_cosparsity_100.mat';
load(OmegaFileName, 'Omega');

%% dictionary - DIF initial, learned with ASimCO, different cosparsities
load(OmegaFileName, 'Omega');

% imread speckled SAR image
In_3d = double(imread('speckle1.tif'));
In = In_3d(:,:,1);
In = max(In, eps);

% reshape to patches 
Xn=im2col(In,[n,n],'sliding');
Xdn=zeros(size(Xn));

Z = log(Xn);
exp_Z = Xn;

% set lambda
lambda = 0.7;
gamma = lambda;

AL_iters = 2000; % iteration number for ADMM
Xdn_prev = Xn;

%% ADMM
Y = Z;
T = Omega*Y;
B1 = zeros(size(T));

ReFluct = [];

for ii = 1:AL_iters
    %% update Y - gradient descent - 1 iteration
    alpha = 1e-2; % slow convergence
    for k = 1:1
        gY = (1-exp_Z.*exp(-Y)) + gamma*Omega'*(Omega*Y-T+B1);
        Y = Y - alpha*gY;        
    end
    
    %% update T - soft-thresholding
    M_softTH = Omega*Y+B1;
    epsilon = lambda/gamma;
    T = sign(M_softTH).*max(abs(M_softTH)-epsilon, 0);
    
    %% update B1
    B1 = B1+(Omega*Y-T);
    
    % reshape exp(Y) to denoised image -- check PSNRdn
    Xdn = exp(Y);
    [MM,NN]=size(In);
    Idn=col2imstep(Xdn,[MM NN],[n n]);
    cnt=countcover([MM NN],[n n],[1 1]);
    Idn=Idn./cnt;
    Idn=max(0,min(255,Idn));
    
    ReFluct(ii) = norm(Xdn-Xdn_prev, 'fro')/norm(Xdn_prev, 'fro');
    if (ii>2) && (abs(ReFluct(ii)) <=1e-4)
        break;
    end
    Xdn_prev = Xdn;
end

% save results
filename = 'RealSAR_Speckle1_RSN_ADL_Cameraman_L4_Params';
save(filename, 'In','Idn','lambda');



