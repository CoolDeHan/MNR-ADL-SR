function [Q,inds] = ComputeOrthoSet(Q0)
%ComputeOrthoSet computes the orthogonal set for the rows of a matrix
%  [Q,inds] = ComputeOrthoSet(Q0)
%  ========================================================================
%  Input:
%  Q0 - input matrix.
%  ========================================================================
%  Output:
%  Q    - orthogonal set of the rows of the input matrix. 
%  inds - index set for the linearly independent rows in the input matrix.
%  ========================================================================
%  Tomer Peleg
%  Department of Electrical Engineering
%  Technion, Haifa 32000 Israel
%  tomerfa@tx.technion.ac.il
%
%  October 2012
%  ========================================================================
Q=Q0;
cnt=1;
inds=zeros(1,size(Q,1));
inds(cnt)=1;
for j=2:size(Q,1)
    Q(j,:)=Q(j,:)-(Q(j,:)*Q(1:cnt,:)')*Q(1:cnt,:);
    v=sqrt(Q(j,:)*Q(j,:)');
    if v>1e-6
        cnt=cnt+1;
        Q(cnt,:)=Q(j,:)/v;
        inds(cnt)=j;
    end
end
Q=Q(1:cnt,:);
inds=inds(1:cnt);