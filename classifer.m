load fisheriris
x = meas
y = species
yhat = classify(x,x,y);
[cm,order] = confusionmat(y,yhat);