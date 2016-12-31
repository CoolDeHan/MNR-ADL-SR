% This file demonstrates how to call the RNS_ADL algorithm to remove
% multiplicative noise in synthetic noisy images

% Parameters:
% L: noise level
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
noiseLevel_vec = [10 4 1];
for num_noiseLevel = 1:length(noiseLevel_vec)
    L = noiseLevel_vec(num_noiseLevel);
    
    % patch dimension
    n=8;
    d=n^2; 
    
    % read original image
    I = double(imread('cameraman.png'));
    
    % load dictionary data
    OmegaFileName = 'Omega_LogDom_cleanImgs_DIFInit_cosparsity_100.mat';
    load(OmegaFileName, 'Omega');
    
    for num_test =1:30
        %% Generate noisy image
        % multiply noise
        r = gamrnd(L,1/L,size(I));
        In = I.*r;
        
        PSNRnoisy=10*log10(255.^2/mean((In(:)-I(:)).^2));
        
        % reshape to patches 
        Xn=im2col(In,[n,n],'sliding');
        Xdn=zeros(size(Xn));
        
        X = im2col(I, [n, n], 'sliding');
        YI = log(X);
        
        Z = log(Xn);
        exp_Z = Xn;
        
        % set lambda
        switch(L)
            case 10
                lambda = 0.4;
            case 4
                lambda = 0.7;
            case 1
                lambda = 1.6;
        end
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
            alpha = 1e-2;
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
            
            % reshape exp(Y) to the denoised image
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
        
        PSNRdn = 10*log10((size(I, 1)^2*(max(I(:))-min(I(:)))^2)/sum((Idn(:)-I(:)).^2));
        MAE = sum(abs(Idn(:)-I(:))) / (size(I, 1)^2);
        MSSIM = ssim_index(I, Idn);
        
        PSNRnoisy_mtx(num_noiseLevel, num_test) = PSNRnoisy;        
        
        PSNRdn_final_mtx(num_noiseLevel, num_test) = PSNRdn
        MAE_final_mtx(num_noiseLevel, num_test) = MAE;
        MSSIM_final_mtx(num_noiseLevel, num_test) = MSSIM;
    end
    % average result
    PSNRdn_mean = mean(PSNRdn_final_mtx, 2);
    MAE_mean = mean(MAE_final_mtx, 2);
    MSSIM_mean = mean(MSSIM_final_mtx, 2);
    
    % standard variation
    PSNRdn_std = std(PSNRdn_final_mtx, 0, 2);
    MAE_std = std(MAE_final_mtx, 0, 2);
    MSSIM_std = std(MSSIM_final_mtx, 0, 2);
    
    fileName = ['Cameraman_RSN_ADL_diffL_30tests'];
    save(fileName, 'PSNRdn_final_mtx','MAE_final_mtx','MSSIM_final_mtx','PSNRnoisy_mtx',...
        'PSNRdn_mean', 'MAE_mean', 'MSSIM_mean', ...
        'PSNRdn_std', 'MAE_std', 'MSSIM_std');
end

