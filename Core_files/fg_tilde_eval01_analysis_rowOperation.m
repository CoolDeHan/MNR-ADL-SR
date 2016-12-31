function [f, g] = fg_tilde_eval01_analysis_rowOperation(X_sparse, Omega, Y)
% This function computes the gradient descent direction for LineSearch
% @ September 2013, University of Surrey 
%
% Details about the algorithm can be found in the paper:
% J. Dong, W. Wang, W. Dai, M. D. Plumbley, Z. Han, J. Chambers
% "Analysis SimCO Algorithms for Sparse Analysis Model Based Dictionary Learning,"
% submitted to IEEE Transactions on Signal Processing, Feburary 2015.
%
% The codes & data have been deposited to http://dx.doi.org/10.15126/surreydata.00808101.
%
% Input parameters:-
% X_Sparse: sparse coefficients
% Omega: initial dictionary
% Y: training data/signals
%
% Output parameters-
% f: cost function value
% g: gradient of the cost function with respect to Omega on the manifold
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

Yr = X_sparse - Omega*Y;
f = sum(sum(Yr.*Yr));

if nargout >= 2
    % jing
    g = -2*(X_sparse-Omega*Y)*Y';     
    
    % make the ith row of g_bar is orthogonal to the ith row of Omega
    d = size(Omega, 2);
    DGcorr = sum(Omega.*g,2);
    g = g - Omega.*repmat(DGcorr,1,d);
end