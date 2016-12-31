function [Omega, f_itn] = analysis_SimCO_normalizedRowOmega(Y, param)
% This function implements the analysis SimCO dictionary learning algorithm 
% @ September 2013, University of Surrey
% 
% Details about the algorithm can be found in the paper:
% J. Dong, W. Wang, W. Dai, M. D. Plumbley, Z. Han, J. Chambers
% "Analysis SimCO Algorithms for Sparse Analysis Model Based Dictionary Learning,"
% submitted to IEEE Transactions on Signal Processing, Feburary 2015.
%
% The codes & data have been deposited to http://dx.doi.org/10.15126/surreydata.00808101.
%
% Y: Training signals
% param: parameters defined 
% Omega: learned analysis dictionary
% f_itn: cost function values at each iteration
%
% Written by Jing Dong, moderated by Wenwu Wang, version 1.0                                    
%
% If you have any questions or comments regarding this package, or if you want to 
% report any bugs or unexpected error messages, please send an e-mail to
% w.wang@surrey.ac.uk
%     
% Copyright 2015 J. Dong, W. Wang, W. Dai, M. D. Plumbley, Z. Han, and J. Chambers
% 
% This software is a free software distributed under the terms of the GNU 
% Public License version 3 (http://www.gnu.org/licenses/gpl.txt). You can 
% redistribute it and/or modify it under the terms of this licence, for 
% personal and non-commercial use and research purpose.

Omega = param.initialDictionary;
N = size(Y, 2);
l = param.cosparsity;

IPara.gmin = 1e-5; % the minimum value of gradient
IPara.Lmin = 1e-6; %t4-t1 should be larger than Lmin
IPara.t4 = 1e-2; %the initial value of t4
IPara.rNmax = 3; %the number of iterative refinement in Part B in DictLineSearch03.m
itN =param.itN;
num_Omega_update = param.numOmegaUpdate;

%% main iteration
for it=1:itN
    %% analysis sparse coding
    X_sparse=Omega*Y;
    %hard thresholding
    [~, index]=sort(abs(X_sparse), 1, 'ascend');
    for i=1:1:N
        X_sparse(index(1:l ,i), i)=0;
    end
    
    %% update Omega
    for i_Omega_update = 1:num_Omega_update
        [Omega]=DictLineSearch03_analysis_rowOperation(X_sparse, Omega, Y, IPara);
    end  
   
end


