pkg load all;
clear all; close all; clc;

tol = 10^(-4);
xspan = [-4 4];

A = 0;
ic = [0.1 0.1];
epsilon = 0.01;

[t, y] = ode45(@homework1_1_1_rhs, xspan, ic, [], epsilon);

plot(t, y(:, 1));

#dbeta = 1;
#
#hold off;
#
#for jj = 0:4
#  dbeta = 1;
#  for j = 1:1000
#    [t, y] = ode45(@all_581_rhs, xspan, ic, [], beta);
#
#    if (abs(y(end, 1)) < tol)
#      beta;
#      beta = beta - 5;
#      #figure(jj + 1)
#      hold on;
#      plot(t, y(:, 1));
#      break;
#    end
#
#    if (y(end, 1) * (-1)^jj > 0)
#      beta = beta - dbeta;
#    else
#      beta = beta + dbeta / 2;
#      dbeta = dbeta / 2;
#    end
#  end
#end
