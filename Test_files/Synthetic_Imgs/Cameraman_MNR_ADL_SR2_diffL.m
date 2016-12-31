% This file demonstrates how to call the MNR_ADL_SR2 algorithm to remove
% multiplicative noise in synthetic noisy images

% Parameters:
% L: noise level
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

rng(1);

noiseLevel_vec = [10 4 1];
for num_noiseLevel = 1:length(noiseLevel_vec)
    L = noiseLevel_vec(num_noiseLevel);
    
    % patch dimension
    n=8;
    d=n^2; 
    
    % load dictionary data
    OmegaFileName = 'Omega_LogDom_cleanImgs_DIFInit_cosparsity_100.mat';
    load(OmegaFileName, 'Omega');
    
    % read original image
    I = double(imread('cameraman.png'));
    
    for num_test = 1:30
        
        % multiply noise
        r = gamrnd(L,1/L,size(I));
        In = I.*r;               
        In = max(In, eps);

        
        PSNRnoisy=10*log10(255.^2/mean((In(:)-I(:)).^2));
        
        % reshape to pachtes 
        Xn=im2col(In,[n,n],'sliding');
        Xdn=zeros(size(Xn));
        
        X = im2col(I, [n, n], 'sliding');
        YI = log(X);
        
        Z = log(Xn);
        exp_Z = Xn;
        
        [MM,NN]=size(In);
        cnt=countcover([MM NN],[n n],[1 1]);
        
        % set lambda1, lambda2
        switch(L)
            case 10
                lambda1 = 0.3;
                lambda2 = 0.1;
            case 4
                lambda1 = 0.6;
                lambda2 = 0.1;
            case 1
                lambda1 = 1.3;
                lambda2 = 0.2;
        end
        
        gamma1 = lambda1;
        gamma2 = lambda2;
        
        AL_iters = 2000; % maximum iteration number for ADMM
        Xdn_prev = Xn;
        
        %% ADMM
        Y = Z;
        T = Omega*Y;
        M = Y;
        B1 = zeros(size(T));
        B2 = zeros(size(Y));
        
        ReFluct = [];
        Idn = In;
                
        for ii = 1:AL_iters
            %% update Y - gradient descent - 1 iteration
            alpha = 1e-2; 
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
        
        PSNRdn = 10*log10((size(I, 1)^2*(max(I(:))-min(I(:)))^2)/sum((Idn(:)-I(:)).^2));
        MAE = sum(abs(Idn(:)-I(:))) / (size(I, 1)^2);
        MSSIM = ssim_index(I, Idn);
        
        PSNRnoisy_mtx(num_noiseLevel, num_test) = PSNRnoisy;
        
        PSNRdn_final_mtx(num_noiseLevel, num_test) = PSNRdn;
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
    
    % save data
    fileName = ['Cameraman_MNR_ADL_SR2_diffL_30tests'];
    save(fileName, 'PSNRdn_final_mtx','MAE_final_mtx','MSSIM_final_mtx','PSNRnoisy_mtx','iter_num_mtx', ...
        'PSNRdn_mean', 'MAE_mean', 'MSSIM_mean', ...
        'PSNRdn_std', 'MAE_std', 'MSSIM_std');
end

