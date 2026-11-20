export interface ClassificationPoint { x: number; y: number; label: 0 | 1 }
export interface LogisticParams { wx: number; wy: number; b: number }
export function sigmoid(z: number): number;
export function logisticScore(point: { x: number; y: number }, params: LogisticParams): number;
export function logisticProbability(point: { x: number; y: number }, params: LogisticParams): number;
export function logisticLoss(points: ClassificationPoint[], params: LogisticParams): number;
export function logisticGradient(points: ClassificationPoint[], params: LogisticParams): LogisticParams & { norm: number };
export function logisticStep(points: ClassificationPoint[], params: LogisticParams, rate: number): LogisticParams & {
  loss: number; gradientNorm: number;
};
export function classificationAccuracy(points: ClassificationPoint[], params: LogisticParams): number;
export function makeClassificationDataset(kind?: 'separable' | 'overlap'): ClassificationPoint[];
