/** Logistic regression from scratch. Class A = 0; class B = 1. */
export function sigmoid(z) {
  // Branching avoids overflow in exp(-z) for very negative scores.
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

export function logisticScore(point, params) {
  return params.wx * point.x + params.wy * point.y + params.b;
}

export function logisticProbability(point, params) {
  return sigmoid(logisticScore(point, params));
}

export function logisticLoss(points, params) {
  if (!points.length) return NaN;
  // softplus(z) - label*z, evaluated without log(0), overflow,
  // or probability clipping that would hide confidently wrong predictions.
  return points.reduce((sum, point) => {
    const z = logisticScore(point, params);
    const correctSideLoss = point.label === 1 ? Math.max(-z, 0) : Math.max(z, 0);
    return sum + correctSideLoss + Math.log1p(Math.exp(-Math.abs(z)));
  }, 0) / points.length;
}

export function logisticGradient(points, params) {
  if (!points.length) return { wx: NaN, wy: NaN, b: NaN, norm: NaN };
  let wx = 0, wy = 0, b = 0;
  for (const point of points) {
    const error = logisticProbability(point, params) - point.label;
    wx += error * point.x;
    wy += error * point.y;
    b += error;
  }
  wx /= points.length;
  wy /= points.length;
  b /= points.length;
  return { wx, wy, b, norm: Math.hypot(wx, wy, b) };
}

export function logisticStep(points, params, rate) {
  const gradient = logisticGradient(points, params);
  const next = {
    wx: params.wx - rate * gradient.wx,
    wy: params.wy - rate * gradient.wy,
    b: params.b - rate * gradient.b,
  };
  return {
    ...next,
    loss: logisticLoss(points, next),
    gradientNorm: gradient.norm,
  };
}

export function classificationAccuracy(points, params) {
  if (!points.length) return NaN;
  return points.filter((point) => Number(logisticScore(point, params) >= 0) === point.label).length / points.length;
}

/** Fixed datasets make resets reproducible and illustrate linear limitations. */
export function makeClassificationDataset(kind = 'separable') {
  const a = [
    [-3.3, -1.6], [-2.9, 0.5], [-2.6, -2.8], [-2, -0.7],
    [-1.6, 1.2], [-1.3, -2.1], [-0.7, -0.8], [-0.3, -2.7],
  ];
  const b = [
    [0.4, 1.5], [0.9, 2.9], [1.3, 0.3], [1.7, 1.9],
    [2.2, -0.6], [2.7, 3.1], [3, 0.8], [3.4, 2],
  ];
  const points = [
    ...a.map(([x, y]) => ({ x, y, label: 0 })),
    ...b.map(([x, y]) => ({ x, y, label: 1 })),
  ];
  if (kind === 'overlap') {
    points.push(
      { x: 1.1, y: 1.7, label: 0 }, { x: -1.1, y: -1.7, label: 1 },
      { x: 0.3, y: -0.2, label: 0 }, { x: -0.3, y: 0.2, label: 1 },
    );
  }
  return points;
}
