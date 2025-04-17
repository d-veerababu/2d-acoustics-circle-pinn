function [loss,gradientsV] = objectiveFunction(parametersV,X,Y,U0,parameterNames,parameterSizes,freq,c,R)

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

% Evaluate model loss and gradients.
[loss,gradients] = dlfeval(@modelLoss,parameters,X,Y,U0,freq,c,R);

% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);

end
