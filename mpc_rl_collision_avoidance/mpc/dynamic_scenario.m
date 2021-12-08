function x_next =  dynamic_scenario( z, p )
%            inputs               |               states
%                acc   delta  sv     x      y       psi   v    s    dummy
    stepSize = 0.1;
    % ego-vehicle
    x = z(4: 8);
    u = z(1: 2);
    % integrator Runge-Kutta integrator of order 4
    x_R_next = RK4(x, u, @continuous_dynamics_R, stepSize);
    x_next = x_R_next;
end

function xdot = continuous_dynamics_R ( x, u )
    a = u(1);
    alpha = u(2);
    theta =  x(3);
    v = x(4);
    w=x(5);
    xdot = [v * cos(theta);
            v * sin(theta);
            w;
            a;
            alpha];
end