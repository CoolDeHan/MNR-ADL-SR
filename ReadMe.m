% The folder "MNR_ADL_SR" contains the codes for the following four sub-folders:
%
% Image_data: contains the training images for analysis dictionary learning
% and the test images for multiplicative noise removal

% Core_files: contains the implementation of the Analysis SimCO algorithm
% and subfunctions for removing multiplicative noise.

% Learn_Omega: contains the file to learn an analysis dictionary using
% Analysis SimCO and the data file of the learned dictionary Omega

% Sample_test: contains two examples on how to use the Analysis SimCO
% algorithms - one for image denoising, and the other for dictionary
% recovery for synthetically generated data. 
%
% Test_files: containts the comprehensive tests performed for generating
% the results in the paper referenced below.
%
%
% References:
% J. Dong, Z. Han, Y. Zhao, W. Wang, A. Prochazka, J. Chambers,
% "Sparse Analysis Model Based Multiplicative Noise Removal with Enhanced Regularization,"
% submitted to Signal Processing, October 2016.
%
% J. Dong, W. Wang, W. Dai, M. D. Plumbley, Z. F. Han and J. Chambers, 
% "Analysis SimCO Algorithms for Sparse Analysis Model Based Dictionary Learning," 
% in IEEE Transactions on Signal Processing, vol. 64, no. 2, pp. 417-431, Jan.15, 2016.
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
