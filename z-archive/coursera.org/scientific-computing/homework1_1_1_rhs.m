function rhs = homework1_1_1_rhs(x, ic, dummy, epsilon)

  y1 = ic(1);
  y2 = ic(2);

  # take K = 1
  rhs = [y2
         (epsilon - x^2) * y1];
end
