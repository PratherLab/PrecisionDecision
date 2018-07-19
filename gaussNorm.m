function g = gaussNorm(range_x, mu, sigma)

g = exp(-0.5 * (range_x-mu).^2 / sigma^2);
if any(g)
  g = g / sum(g);
end

