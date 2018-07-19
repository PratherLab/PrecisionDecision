function g = gauss(range_x, mu, sigma)

g = exp(-0.5 * (range_x-mu).^2 / sigma^2);

