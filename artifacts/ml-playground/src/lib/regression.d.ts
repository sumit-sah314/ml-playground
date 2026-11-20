export interface Point { x: number; y: number }
export interface FitResult { w: number; b: number; loss: number; unique: boolean }
export function meanSquaredError(points: Point[], w: number, b: number): number;
export function gradientStep(points: Point[], w: number, b: number, learningRate: number): {
  w: number; b: number; loss: number; gradientNorm: number;
};
export function leastSquares(points: Point[]): FitResult | null;
export function createDataset(kind?: 'linear' | 'noisy' | 'negative', count?: number): Point[];
