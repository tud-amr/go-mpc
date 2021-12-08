function ineq_constr =  inequality_constr_scenario( z, p,i)

% states and inputs for ego vehicle
    x_R = z(4: 6);
    u_R = z(1: 3);
    x = x_R(1);
    y = x_R(2);
    theta = x_R(3);
    slack = u_R(3);
    
    %% Parameters
    r_disc = p(27); disc_pos_0 = p(28);
    obst1_x = p(29); obst1_y = p(30); obst1_theta = p(31); obst1_major = p(32); obst1_minor= p(33);
    obst2_x = p(34); obst2_y = p(35); obst2_theta = p(36); obst2_major = p(37); obst2_minor= p(38);
    obst3_x = p(39); obst3_y = p(40); obst3_theta = p(41); obst3_major = p(42); obst3_minor= p(43);
    obst4_x = p(44); obst4_y = p(45); obst4_theta = p(46); obst4_major = p(47); obst4_minor= p(48);
    obst5_x = p(49); obst5_y = p(50); obst5_theta = p(51); obst5_major = p(52); obst5_minor= p(53);
    obst6_x = p(54); obst6_y = p(55); obst6_theta = p(56); obst6_major = p(57); obst6_minor= p(58);
    
    %% Collision Avoidance Constraints
	R_car= [cos(theta), -sin(theta); sin(theta), cos(theta)];
	CoG = [x;y];

	shift_0 = [disc_pos_0; 0];

    % Car disc positions
	position_disc_0 = CoG+R_car*shift_0;
    
    %% Obstacles
    % Obstacle 1
	CoG_obst1 = [obst1_x;obst1_y];
	deltaPos_disc_0_obstacle_1 =  position_disc_0-CoG_obst1;

    % Obstacle 2
	CoG_obst2 = [obst2_x;obst2_y];
	deltaPos_disc_0_obstacle_2 =  position_disc_0-CoG_obst2;

    
    % Obstacle 3
	CoG_obst3 = [obst3_x;obst3_y];
	deltaPos_disc_0_obstacle_3 =  position_disc_0-CoG_obst3;

    % Obstacle 4
	CoG_obst4 = [obst4_x;obst4_y];
	deltaPos_disc_0_obstacle_4 =  position_disc_0-CoG_obst4;

    
    % Obstacle 5
	CoG_obst5 = [obst5_x;obst5_y];
	deltaPos_disc_0_obstacle_5 =  position_disc_0-CoG_obst5;


    % Obstacle 6
	CoG_obst6 = [obst6_x;obst6_y];
	deltaPos_disc_0_obstacle_6 =  position_disc_0-CoG_obst6;

    
    %% Relative Rotation Matrix
	ab_1 = [1/((obst1_major + r_disc)*(obst1_major + r_disc)),0;0,1/((obst1_minor + r_disc)*(obst1_minor + r_disc))];
	ab_2 = [1/((obst2_major + r_disc)*(obst2_major + r_disc)),0;0,1/((obst2_minor + r_disc)*(obst2_minor + r_disc))];
    ab_3 = [1/((obst3_major + r_disc)*(obst3_major + r_disc)),0;0,1/((obst3_minor + r_disc)*(obst3_minor + r_disc))];
	ab_4 = [1/((obst4_major + r_disc)*(obst4_major + r_disc)),0;0,1/((obst4_minor + r_disc)*(obst4_minor + r_disc))];
    ab_5 = [1/((obst5_major + r_disc)*(obst5_major + r_disc)),0;0,1/((obst5_minor + r_disc)*(obst5_minor + r_disc))];
	ab_6 = [1/((obst6_major + r_disc)*(obst6_major + r_disc)),0;0,1/((obst6_minor + r_disc)*(obst6_minor + r_disc))];

	R_obst_1 = [cos(obst1_theta), -sin(obst1_theta);sin(obst1_theta),cos(obst1_theta)];
	R_obst_2 = [cos(obst2_theta), -sin(obst2_theta);sin(obst2_theta),cos(obst2_theta)];
    R_obst_3 = [cos(obst3_theta), -sin(obst3_theta);sin(obst3_theta),cos(obst3_theta)];
	R_obst_4 = [cos(obst4_theta), -sin(obst4_theta);sin(obst4_theta),cos(obst4_theta)];
    R_obst_5 = [cos(obst5_theta), -sin(obst5_theta);sin(obst5_theta),cos(obst5_theta)];
	R_obst_6 = [cos(obst6_theta), -sin(obst6_theta);sin(obst6_theta),cos(obst6_theta)];

    %% Constraints
    c_disc_0_obst_1 = deltaPos_disc_0_obstacle_1' * R_obst_1' * ab_1 * R_obst_1 * deltaPos_disc_0_obstacle_1;
    c_disc_0_obst_2 = deltaPos_disc_0_obstacle_2' * R_obst_2' * ab_2 * R_obst_2 * deltaPos_disc_0_obstacle_2;
    c_disc_0_obst_3 = deltaPos_disc_0_obstacle_3' * R_obst_3' * ab_3 * R_obst_3 * deltaPos_disc_0_obstacle_3;
    c_disc_0_obst_4 = deltaPos_disc_0_obstacle_4' * R_obst_4' * ab_4 * R_obst_4 * deltaPos_disc_0_obstacle_4;
    c_disc_0_obst_5 = deltaPos_disc_0_obstacle_5' * R_obst_5' * ab_5 * R_obst_5 * deltaPos_disc_0_obstacle_5;
    c_disc_0_obst_6 = deltaPos_disc_0_obstacle_6' * R_obst_6' * ab_6 * R_obst_6 * deltaPos_disc_0_obstacle_6;

        ineq_constr = [c_disc_0_obst_1 + slack;
            c_disc_0_obst_2 + slack;
            c_disc_0_obst_3 + slack;
            c_disc_0_obst_4 + slack;
            c_disc_0_obst_5 + slack;
            c_disc_0_obst_6 + slack];

end
    
