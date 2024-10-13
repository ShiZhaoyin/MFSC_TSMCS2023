function[Final,Result] = MFSC(fea,gnd,d,m,lambda,gamma_ratio,p,h, k_ne)
k = max(gnd);
% Nomalize each vector to unit
% ===========================================
[nSmp,~] = size(fea);
for i = 1:nSmp
     fea(i,:) = fea(i,:) ./ max(1e-12,norm(fea(i,:)));
end
% ===========================================
% Scale the features (pixel values) to [0,1]
% ===========================================
maxValue = max(max(fea));
fea = fea/maxValue;
%% construction of Graph
X = full(fea');
d1 = d; d2 = d;
XoT = reshape(X,d1,d2,[]); 
[WEI] =  make_Adjacency(X,k_ne,h);
L = diag(sum(WEI,2)) - WEI;
Lp2 = kron(L,sparse(eye(p(2))));
Lp1 = kron(L,sparse(eye(p(1))));
%%
gamma = gamma_ratio*lambda;
XoT = gpuArray(XoT);
tol = 5e-5;
%%
iter = 30;
Result  = zeros(iter,4);
for q = 1:iter
    tStart = tic;
    try
        [~, U, ~] = two_dFuzzyPca_graph(XoT,k,m,lambda,p(1),p(2),L,Lp2,Lp1,gamma,tol);
        [~, label] = max(U,[],2);
        Result(q,4) = toc(tStart);
        Result(q,1:3) = ClusteringMeasure(label, gnd);
        fprintf('ACC is:%f\n ',Result(q,1));
        fprintf('NMI is:%f\n ',Result(q,2));
        fprintf('Purity is:%f\n ',Result(q,3));
    catch
        continue
    end
    fprintf('Processing is:%.2f%%\n ',(q/iter)*100);
end
Final(1,:) = mean(Result);
Final(2,:) = std(Result);
end
%%
function [model, R, llh] = two_dFuzzyPca_graph(XoT,k,m,lambda,p1,p2,L,Lp2,Lp1,gamma,tol) 
XoT1 = XoT;
[d1, d2, n] = size(XoT);
d = d1*d2;
X = reshape(XoT,[d,n]);
maxiter = 500;
llh = inf(1,maxiter);
label = ceil(k*rand(1,n)); 
R = full(sparse(1:n,label,1,n,k,n));
G1 = zeros(size(R));
G2 = zeros(1,k);
U = zeros(d1,p1,k);
V = zeros(d2,p2,k);
for iter = 2:maxiter
    nk = sum(R.^m,1);
    mu = bsxfun(@times, X*R.^m, 1./nk);
    Sigma1 = zeros(d1,d1,k);
    Sigma2 = zeros(d2,d2,k);
    r = R.^(m/2);
    for i = 1:k
        Xo = bsxfun(@minus,X,mu(:,i));
        Yo = bsxfun(@times,Xo,r(:,i)');
        XoT = reshape(Xo,[d1, d2, n]);
        YoT = reshape(Yo,[d1, d2, n]);
        if iter == 2
            Sigma1(:,:,i) = lambda*sum(pagemtimes(YoT,permute(YoT,[2 1 3])),3);
        else
            Sigma1(:,:,i) = lambda*sum(pagemtimes(pagemtimes(YoT,repmat(V(:,:,i)*V(:,:,i)',[1 1 n])),permute(YoT,[2 1 3])),3);
            Xv = ten2mat(pagemtimes(XoT1,V(:,:,i)),1);
            Sigma1(:,:,i) = Sigma1(:,:,i) - gamma*Xv*Lp2*Xv';
        end
        [U(:,:,i),~,~] = svds(Sigma1(:,:,i),p1);
        Sigma2(:,:,i) = lambda*sum(pagemtimes(pagemtimes(permute(YoT,[2 1 3]),repmat(U(:,:,i)*U(:,:,i)',[1 1 n])),YoT),3);
        Xu = ten2mat(pagemtimes(permute(XoT1,[2 1 3]),U(:,:,i)),1);
        Sigma2(:,:,i) = Sigma2(:,:,i) - gamma*Xu*Lp1*Xu';
        [V(:,:,i),~,~] = svds(Sigma2(:,:,i),p2);
        G1(:,i) = reshape(sum(sum((XoT - lambda*pagemtimes(pagemtimes(repmat(U(:,:,i)*U(:,:,i)',[1 1 n]),XoT),repmat(V(:,:,i)*V(:,:,i)',[1 1 n]))).^2,1),2),[n,1]);
        XvecT = pagemtimes(pagemtimes(permute(U(:,:,i),[2 1 3]),XoT),V(:,:,i));
        Xvec = reshape(XvecT,(p1*p2),n);
        G2(:,i) = trace(Xvec*L*Xvec');
    end
    G = G1 + gamma*repmat(G2,[n,1]);
    R = bsxfun(@rdivide,G.^(-1/(m-1)),sum(G.^(-1/(m-1)),2));
    llh (iter) = sum(sum(R.^m.*G));
    %%
    % fprintf('llh is:%f\n ',llh(iter));
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter)); break; end
    if llh(iter - 1) < llh(iter); break; end
end
model.mu = mu;
model.Sigma1 = Sigma1;
model.Sigma2 = Sigma2;
model.U = U;
model.V = V;
llh = llh(2:iter);
end
%%
function [W] =  make_Adjacency(X,k_ne,h)
[~,n] = size(X);
[idx,D] = knnsearch(X',X','k',k_ne + 1);
idx(:,1) = [];
D(:,1) =[];
W = zeros(n,n);
for i = 1:n
    W(i,idx(i,:)) = exp(-D(i,:).^2/(2*h.^2));
end
e = W - W';
e(e>0) = 0;
W = W + abs(e);
end

function mat = ten2mat( ten,k)
dim = size(ten);
mat = reshape(permute(ten,[k,1:k-1,k+1:length(dim)]),dim(k),[]);
end