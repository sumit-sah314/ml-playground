/**
 * Linear regression from first principles. No numerical or ML libraries.
 * A point is { x: number, y: number }; the prediction is w * x + b.
 */
export function meanSquaredError(points, w, b) {
  if (!points.length) return NaN;
  return points.reduce((sum, point) => {
    const residual = w * point.x + b - point.y;
    return sum + residual * residual;
  }, 0) / points.length;
}

/**
 * Both derivatives are evaluated at the OLD parameters and updated together:
 * dL/dw = (2/n) Σ x_i(wx_i + b - y_i)
 * dL/db = (2/n) Σ (wx_i + b - y_i)
 * No normalization, parameter clipping, or automatic learning-rate changes.
 */
export function gradientStep(points, w, b, learningRate) {
  if (!points.length) {
    return { w, b, loss: NaN, gradientNorm: NaN };
  }
  let dw = 0;
  let db = 0;
  for (const point of points) {
    const residual = w * point.x + b - point.y;
    dw += point.x * residual;
    db += residual;
  }
  dw *= 2 / points.length;
  db *= 2 / points.length;
  const nextW = w - learningRate * dw;
  const nextB = b - learningRate * db;
  return {
    w: nextW,
    b: nextB,
    loss: meanSquaredError(points, nextW, nextB),
    gradientNorm: Math.hypot(dw, db),
  };
}

/**
 * Closed form: w = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²), b = ȳ - wx̄.
 * Centering first avoids subtracting two large, nearly equal sums.
 * When all x are identical, many lines minimize the loss. We return the
 * horizontal representative y = mean(y), explicitly marked non-unique.
 */
export function leastSquares(points) {
  if (!points.length) return null;
  const n = points.length;
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / n;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / n;
  let covariance = 0;
  let variance = 0;
  for (const point of points) {
    covariance += (point.x - meanX) * (point.y - meanY);
    variance += (point.x - meanX) ** 2;
  }
  const unique = variance > Number.EPSILON * Math.max(1, n * meanX * meanX);
  const w = unique ? covariance / variance : 0;
  const b = meanY - w * meanX;
  return { w, b, loss: meanSquaredError(points, w, b), unique };
}

/** Small practice datasets in the graph's [0,10] × [0,10] domain. */
export function createDataset(kind = 'linear', count = 15) {
  const size = Math.max(0, Math.min(200, Math.floor(count)));
  return Array.from({ length: size }, (_, index) => {
    const x = 0.7 + (index / Math.max(1, size - 1)) * 8.6;
    const noise = (Math.random() - 0.5) * (kind === 'noisy' ? 4 : 1.8);
    const y = kind === 'negative' ? 8.8 - 0.72 * x + noise : 1.2 + 0.72 * x + noise;
    return { x, y: Math.max(0.2, Math.min(9.8, y)) };
  });
}
