import type { ClassificationPoint } from './logistic.js';
export interface NetworkLayer { weights: number[][]; biases: number[] }
export interface NeuralNetwork {
  sizes: number[];
  activation: 'relu' | 'tanh';
  layers: NetworkLayer[];
}
export function createNetwork(hidden?: number[], activation?: 'relu' | 'tanh', seed?: number): NeuralNetwork;
export function forwardNetwork(network: NeuralNetwork, x: number, y: number): {
  probability: number; score: number; activations: number[][]; preActivations: number[][];
};
export function predictNetwork(network: NeuralNetwork, x: number, y: number): number;
export function networkMetrics(network: NeuralNetwork, points: ClassificationPoint[]): { loss: number; accuracy: number };
export function networkGradients(network: NeuralNetwork, points: ClassificationPoint[]): {
  layers: NetworkLayer[]; norm: number; loss: number;
};
export function trainEpoch(network: NeuralNetwork, points: ClassificationPoint[], learningRate: number): {
  loss: number; accuracy: number; gradientNorm: number;
};
export function makeNeuralDataset(kind?: 'circles' | 'xor' | 'spirals', seed?: number): ClassificationPoint[];
