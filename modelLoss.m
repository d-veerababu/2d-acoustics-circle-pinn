function [loss,gradients] = modelLoss(parameters,X,Y,U0,freq,c,R)
% Make predictions with the initial conditions.
k = 2*pi*freq/c;
U = model(parameters,X,Y);

% Calculate approximate distance function 
phi = (R^2-(X.^2+Y.^2))/(2*R);

% Construct trial solution
G = (1-phi).*U0+phi.*U;

% Calculate derivatives with respect to X and Y.
gradientsG = dlgradient(sum(G,"all"),{X,Y},EnableHigherDerivatives=true);
Gx = gradientsG{1};
Gy = gradientsG{2};


% Calculate second-order derivatives with respect to X.
Gxx = dlgradient(sum(Gx,'all'),X,'EnableHigherDerivatives',true);

% Calculate second-order derivatives with respect to X.
Gyy = dlgradient(sum(Gy,'all'),Y,'EnableHigherDerivatives',true);

% Calculate functional loss.
f = Gxx+Gyy+k^2*G;
zeroTarget = zeros(size(f),"like",f);
loss = l2loss(f, zeroTarget);


% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
