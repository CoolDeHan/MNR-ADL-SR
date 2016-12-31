% Jing Dong Junly 27, 2015
% Learn an analysis dictionary Omega with Analysis SimCO
% training samples: log transform of some other clean images
% preprocess: (1) remove the image patches with zero pixel values; 
% (2) apply log transform

% initial Omega:  DIF dictionary.

% get training patches from the log transform of some clean natural images
ImNames={'biker.png', 'haus-sonnenstrahl-10.png', 'Skyline.png', ...
    'woman1.bmp', 'barbara.png'};
X = [];
patch_sz = 8;
m = patch_sz^2;
for ii = 1:length(ImNames);
    I = double(imread(ImNames{ii}));
    X = [X im2col(I,[patch_sz,patch_sz],'distinct')];
end

% remove the columns with zero elements 
X_col_min = min(X);
index_col_nonzero = find(X_col_min); 
X = X(:,index_col_nonzero); 

% apply log-transform to the image patches
X = log(X);

randomSeed = 1;
rng(randomSeed); 
N = 20000;
IdxN=randperm(size(X, 2));
pos=IdxN(IdxN(1:min(N,size(X, 2))));
Xntrain=X(:,pos);
disp(['Number of examples: ',num2str(size(Xntrain,2))]);

cosparsity = 100;
p = 2*m; % number of atoms

% initialize Omega
rng(randomSeed+1);
OmegaInit = GenerateOmegaDIF(patch_sz);

%%  Learn Omega using Analysis SimCO
param.initialDictionary = OmegaInit;
param.itN = 2000;
param.cosparsity = cosparsity;
param.numOmegaUpdate = 1;
[Omega] = analysis_SimCO_normalizedRowOmega(Xntrain, param);

% save data
OmegaFileName = ['Omega_LogDom_cleanImgs_DIFInit_cosparsity_', num2str(cosparsity)];
save(OmegaFileName, 'Omega');

