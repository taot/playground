pkg load all;
clear all; close all; clc;

tol = 10^(-4);
xspan = [-1 1];

A = 1;
ic = [0 A];
beta = 99;

dbeta = 1;

hold off;

for jj = 0:4
  dbeta = 1;
  for j = 1:1000
    [t, y] = ode45(@all_581_rhs, xspan, ic, [], beta);

    if (abs(y(end, 1)) < tol)
      beta;
      beta = beta - 5;
      #figure(jj + 1)
      hold on;
      plot(t, y(:, 1));
      break;
    end

    if (y(end, 1) * (-1)^jj > 0)
      beta = beta - dbeta;
    else
      beta = beta + dbeta / 2;
      dbeta = dbeta / 2;
    end
  end
end
