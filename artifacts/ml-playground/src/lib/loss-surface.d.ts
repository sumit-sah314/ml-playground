import type { Point, FitResult } from './regression.js';
export const SURFACE_POINTS: Point[];
export interface SurfaceGeometry {
  minimum: FitResult;
  eigenvalues: [number, number];
  eigenvectors: [[number, number], [number, number]];
  stableRate: number;
  presets: { converge: number; oscillate: number; diverge: number };
}
export function surfaceGeometry(points?: Point[]): SurfaceGeometry;
export function contourRing(level: number, geometry: SurfaceGeometry, samples?: number): { w: number; b: number }[];
