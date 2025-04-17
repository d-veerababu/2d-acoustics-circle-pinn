% Program to solve the Helmholtz equation using L-BFGS algorithm on a circular domain
clc;
clear all;

%% Generate training data
freq = 500;                     % Frequency
c = 340;                        % Speed of sound in air

R = 1;                          % Radius
T = 2*pi;                       % Azimuthal angle

nDiv = 100;                      % No.of divisions along radial and azimuthal directions

U0 = 1;                         % Dirichlet boundary value

pointSet = sobolset(2,"Skip",0);            % Base-2 digital sequence that fills space in a highly uniform manner
coordinates = net(pointSet,nDiv);           % Generates quasirandom point set

r = R*coordinates(:,1);                     % Creates random r-data points along the radius
t = T*coordinates(:,2);                     % Creates random y-data points in azimuthal direction

[dataR, dataT] = meshgrid(r,t);             % Generate meshgrid

dataX = dataR.*cos(dataT);                  % Calculate x-data in rectangular coordinate system
dataY = dataR.*sin(dataT);                  % Calculate y-data in rectangualr coordinate system

%% Define deep learning model
numLayers = 5;
numNeurons = 90;
maxFuncEvaluations = 20000;
maxIterations = 20000;

parameters = buildNet(numLayers,numNeurons);

%% Specify optimization options
options = optimoptions("fmincon", ...
    HessianApproximation="lbfgs", ...
    MaxIterations=maxIterations, ...
    MaxFunctionEvaluations=maxFuncEvaluations, ...
    OptimalityTolerance=1e-5, ...
    SpecifyObjectiveGradient=true, ...
    Display='iter');

%% Train network
start = tic;

[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = extractdata(parametersV);

% Convert the variables into deep-learning variables
X = dlarray(dataX(:)',"CB");
Y = dlarray(dataY(:)',"CB");
U0 = dlarray(U0,"CB");
R = dlarray(R,"CB");

objFun = @(parameters) objectiveFunction(parameters,X,Y,U0,parameterNames,parameterSizes,freq,c,R);

parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);

parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

toc(start)

%% Saving the network parameters
archFileName = sprintf('params_l%d_n%d_iter%d_sin_%dHz_trial_soln.mat',numLayers,numNeurons,maxFuncEvaluations,freq);
save(archFileName,"-struct","parameters")


%% Evaluate model accuracy
nDivTest = 50;
rTest = linspace(0,extractdata(R),nDivTest);
tTest = linspace(0,T,nDivTest);
[RTest,TTest] = meshgrid(rTest,tTest);

XTest = RTest.*cos(TTest);
YTest = RTest.*sin(TTest);

dlXTest = dlarray(XTest(:)',"CB");
dlYTest = dlarray(YTest(:)',"CB");
dlUPred = model(parameters,dlXTest,dlYTest);

phi_test = (R^2-(dlXTest.^2+dlYTest.^2))/(2*R);

dlUPred = (1-phi_test).*U0+phi_test.*dlUPred;

f1 = figure;

UPred = reshape(extractdata(dlUPred),[nDivTest,nDivTest]);
surf(XTest,YTest,UPred)
view(2)
colormap jet
colorbar
clim([-7 2])
title("Frequency = " + freq + " Hz")

figFileName = sprintf('UPred_%dHz_trial_soln.jpg',freq);
saveas(f1,figFileName)
