function [cost] =  objective_scenario( z, p,i)
%% Cost for ego-vehicle    
% states and inputs for ego vehicle
%            inputs               |               states
%                v   w  sv     x      y       theta      dummy
x_R = z(4: 8);
u_R = z(1: 2);
a = u_R(1);
alpha = u_R(2);
sv = z(3);
x = x_R(1);
y = x_R(2);
theta = x_R(3);
v = x_R(4);
%% Online parameters

x_goal = p(1); y_goal = p(2); Wrepulsive = p(3); 

% Weightsvref
Wx = p(4);
Wy = p(5);
Ww = p(6);
Wtheta = p(7);
Wv = p(8);
Ws = p(9);
% References
vref = p(10);
wref = p(11);

c1 = p(59);
c2 = p(60);
c3 = p(61);
c4 = p(62);
c5 = p(63);
c6 = p(64);
d = p(65);
w_cost = p(66);

%% Total cost (0.9^i)*(
x_error = x - x_goal;
y_error = y - y_goal;

%% Parameters
    r_disc = p(27); disc_pos_0 = p(28);
    obst1_x = p(29); obst1_y = p(30); obst1_theta = p(31); obst1_major = p(32); obst1_minor= p(33);
    obst2_x = p(34); obst2_y = p(35); obst2_theta = p(36); obst2_major = p(37); obst2_minor= p(38);
    obst3_x = p(39); obst3_y = p(40); obst3_theta = p(41); obst3_major = p(42); obst3_minor= p(43);
    obst4_x = p(44); obst4_y = p(45); obst4_theta = p(46); obst4_major = p(47); obst4_minor= p(48);
    obst5_x = p(49); obst5_y = p(50); obst5_theta = p(51); obst5_major = p(52); obst5_minor= p(53);
    obst6_x = p(54); obst6_y = p(55); obst6_theta = p(56); obst6_major = p(57); obst6_minor= p(58);
    
    %% Collision Avoidance Constraints
    
    %% Obstacles
    % Obstacle 1
	deltaPos_disc_0_obstacle_1 =  sqrt((obst1_x-x)^2+(obst1_y-y)^2);

    % Obstacle 2
	deltaPos_disc_0_obstacle_2 =   sqrt((obst2_x-x)^2+(obst2_y-y)^2);
    
    % Obstacle 3
	deltaPos_disc_0_obstacle_3 =   sqrt((obst3_x-x)^2+(obst3_y-y)^2);

    % Obstacle 4
	deltaPos_disc_0_obstacle_4 =   sqrt((obst4_x-x)^2+(obst4_y-y)^2);
    
    % Obstacle 5
	deltaPos_disc_0_obstacle_5 =  sqrt((obst5_x-x)^2+(obst5_y-y)^2);

    % Obstacle 6
	deltaPos_disc_0_obstacle_6 =  sqrt((obst6_x-x)^2+(obst6_y-y)^2);
    
    obs_lambda = 10.0;
    
    field1 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_1 - (obst1_major+r_disc+0.05))),10^3));
    field2 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_2 - (obst2_major+r_disc+0.05))),10^3));
    field3 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_3 - (obst3_major+r_disc+0.05))),10^3));
    field4 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_4 - (obst4_major+r_disc+0.05))),10^3));
    field5 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_5 - (obst5_major+r_disc+0.05))),10^3));
    field6 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_6 - (obst6_major+r_disc+0.05))),10^3));

%cost = Wx*x_error*x_error + Wy*y_error*y_error + Wv*v*v +Ww*w*w + Ws*sv*sv; % Wv*v*v +Ww*w*w  + Ws*sv*sv
%if i == 20
%    cost = Wx*x_error*x_error + Wy*y_error*y_error + w_cost*(c1 + c2*x + c3*y+ c4*x*x + c5*x*y + c6*y*y + d)+ Wv*v*v +Ww*w*w + Ws*sv*sv+Wrepulsive*(field1 + field2 +field3 + field4+field5 + field6);
%else
%    cost = Wx*x_error*x_error + Wy*y_error*y_error + Wv*v*v +Ww*w*w + Ws*sv*sv+Wrepulsive*(field1 + field2 +field3 + field4+field5 + field6);
%end

disToGoal = sqrt(x_error^2+y_error^2);
disToGoal   =   max(disToGoal, 1.0);      % in case arriving at goal posistion
max_v_range = 10.0;
max_w_range=12.0;

if i == 20
    cost = 8.0*(Wx*x_error^2/disToGoal + Wy*y_error^2/disToGoal)+ w_cost*(c1 + c2*x + c3*y+ c4*x*x + c5*x*y + c6*y*y + d) + Ws*sv*sv;%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
else
    cost = Wv*a*a/max_v_range +Ww*alpha*alpha/max_w_range + Ws*sv*sv ;%+ Wv*(v-vref)*(v-vref);%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
end

end
