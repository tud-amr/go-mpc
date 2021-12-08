% Plot sigmoid

distance = 0:0.1:2;

obs_lambda = 10.0;
obs_buffer = 1.1;
   
field1 =  1 ./ (1+exp(obs_lambda.*(distance - obs_buffer))); 

plot(distance,field1.^2);