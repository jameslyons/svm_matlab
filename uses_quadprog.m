% basic svm that uses quadprog for optimisation. Improvements: compute RBF kernel, 
% use SMO for optimisation.

clear all;
close all;
% This first part just sets up a train and test set
N = 100;
D = 2;
X = zeros(2*N,D);
c1ind = 1:N;
c2ind = N+1:2*N;
X(c1ind,:) = rand(N,D)+repmat([0.3,0],N,1);
X(c2ind,:) = rand(N,D)+repmat([-0.3,0],N,1);
y = -ones(2*N,1);
y(c2ind) = 1;
plot(X(c1ind,1),X(c1ind,2),'b.'); hold on;
plot(X(c2ind,1),X(c2ind,2),'r.');

% This is the meaty part, everything is handled by quadprog
H = (y*y').*(X*X');
C = 1;
options = optimset('Display','off','LargeScale','off');
alphas = quadprog(H,-ones(2*N,1),-eye(2*N),zeros(2*N,1),y',0,zeros(2*N,1),C*ones(2*N,1),[],options);

% get the subset of points that have alpha > 0
ind = alphas>1e-8; % 1e-8 is the tolerance in X for interior-point algo
sv = zeros(sum(ind),1);
sv = alphas(ind); % keep track of the support vectors
svy = y(ind);
svx = X(ind,:);

% compute the w and b vectors, we only need worry about the ones with
% alpha>0
w = sum(repmat(sv,1,D).*repmat(svy,1,D).*svx);
b = median(w*svx' - svy');

% plot the discriminant line on the plot
lx = [-0.5;2];
ly = (b - w(1)*lx)./w(2);
line(lx,ly);
xlim([-0.5,2]);
ylim([-0.5,2]);
fprintf('complete\n');
