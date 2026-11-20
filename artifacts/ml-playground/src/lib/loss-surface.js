import { leastSquares } from './regression.js';

// A fixed, non-centered dataset makes the two parameter directions correlated:
// the narrow valley reveals overshooting and zig-zagging without fake motion.
export const SURFACE_POINTS = [
  { x: 0, y: 1.1 }, { x: 0.5, y: 1.2 }, { x: 1, y: 1.9 },
  { x: 1.5, y: 2.0 }, { x: 2, y: 2.7 }, { x: 2.5, y: 2.8 },
  { x: 3, y: 3.5 }, { x: 3.5, y: 3.6 }, { x: 4, y: 4.3 },
];

/**
 * For L = mean((wx+b-y)^2), the Hessian is
 * H = 2 [[mean(x²), mean(x)], [mean(x), 1]].
 * Its eigenvectors give contour directions and its eigenvalues their widths.
 * Constant-rate GD is stable for 0 < rate < 2/largestEigenvalue.
 */
export function surfaceGeometry(points = SURFACE_POINTS) {
  const minimum = leastSquares(points);
  if (!minimum?.unique) throw new Error('A loss surface needs at least two distinct x values.');
  const n = points.length;
  const a = 2 * points.reduce((sum, p) => sum + p.x * p.x, 0) / n;
  const c = 2 * points.reduce((sum, p) => sum + p.x, 0) / n;
  const d = 2;
  const delta = Math.hypot(a - d, 2 * c);
  const large = (a + d + delta) / 2;
  const small = (a * d - c * c) / large;
  const angle = 0.5 * Math.atan2(2 * c, a - d);
  const majorCurvature = [Math.cos(angle), Math.sin(angle)];
  const minorCurvature = [-Math.sin(angle), Math.cos(angle)];
  return {
    minimum,
    eigenvalues: [large, small],
    eigenvectors: [majorCurvature, minorCurvature],
    stableRate: 2 / large,
    presets: { converge: 0.7 / large, oscillate: 1.8 / large, diverge: 2.15 / large },
  };
}

/**
 * L(theta) - L(theta*) = (1/2)(theta-theta*)ᵀH(theta-theta*).
 * Generate exact constant-MSE ellipses, not decorative approximation rings.
 */
export function contourRing(level, geometry, samples = 180) {
  if (!Number.isFinite(level) || level < 0) throw new Error('Contour height must be nonnegative and finite.');
  const count = Math.max(4, Math.floor(samples));
  const { minimum, eigenvalues, eigenvectors } = geometry;
  const r0 = Math.sqrt(2 * level / eigenvalues[0]);
  const r1 = Math.sqrt(2 * level / eigenvalues[1]);
  return Array.from({ length: count + 1 }, (_, index) => {
    const angle = 2 * Math.PI * index / count;
    const p = r0 * Math.cos(angle);
    const q = r1 * Math.sin(angle);
    return {
      w: minimum.w + p * eigenvectors[0][0] + q * eigenvectors[1][0],
      b: minimum.b + p * eigenvectors[0][1] + q * eigenvectors[1][1],
    };
  });
}
