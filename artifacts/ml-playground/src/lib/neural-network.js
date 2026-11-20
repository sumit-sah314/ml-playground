import { sigmoid } from './logistic.js';

/** Deterministic random numbers make architecture comparisons reproducible. */
function randomSource(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function createNetwork(hidden = [8, 8], activation = 'tanh', seed = 42) {
  if (!['tanh', 'relu'].includes(activation)) throw new Error('Choose tanh or ReLU.');
  if (!Array.isArray(hidden) || hidden.length > 3 || hidden.some((n) => !Number.isInteger(n) || n < 1 || n > 16)) {
    throw new Error('Choose up to three hidden layers, with 1–16 neurons each.');
  }
  const sizes = [2, ...hidden, 1];
  const random = randomSource(seed);
  const layers = sizes.slice(1).map((outputs, index) => {
    const inputs = sizes[index];
    const isHidden = index < sizes.length - 2;
    // Uniform He initialization for ReLU hidden layers; Xavier otherwise.
    const limit = activation === 'relu' && isHidden
      ? Math.sqrt(6 / inputs)
      : Math.sqrt(6 / (inputs + outputs));
    return {
      weights: Array.from({ length: outputs }, () =>
        Array.from({ length: inputs }, () => (2 * random() - 1) * limit)),
      biases: Array(outputs).fill(activation === 'relu' && isHidden ? 0.01 : 0),
    };
  });
  return { sizes, activation, layers };
}

/** Cache each activation and preactivation for the chain rule in backprop. */
export function forwardNetwork(network, x, y) {
  const activations = [[x, y]];
  const preActivations = [];
  let input = activations[0];
  for (let l = 0; l < network.layers.length; l++) {
    const layer = network.layers[l];
    const outputLayer = l === network.layers.length - 1;
    const z = layer.biases.map((bias, j) => {
      let sum = bias;
      for (let k = 0; k < input.length; k++) sum += layer.weights[j][k] * input[k];
      return sum;
    });
    const a = z.map((value) => outputLayer
      ? sigmoid(value)
      : network.activation === 'relu' ? Math.max(0, value) : Math.tanh(value));
    preActivations.push(z);
    activations.push(a);
    input = a;
  }
  return {
    probability: input[0],
    score: preActivations[preActivations.length - 1][0],
    activations,
    preActivations,
  };
}

/** Prediction avoids backprop caches when painting thousands of map pixels. */
export function predictNetwork(network, x, y) {
  let input = [x, y];
  for (let l = 0; l < network.layers.length; l++) {
    const layer = network.layers[l];
    const outputLayer = l === network.layers.length - 1;
    const next = Array(layer.biases.length);
    for (let j = 0; j < next.length; j++) {
      let z = layer.biases[j];
      for (let k = 0; k < input.length; k++) z += layer.weights[j][k] * input[k];
      next[j] = outputLayer ? sigmoid(z)
        : network.activation === 'relu' ? Math.max(0, z) : Math.tanh(z);
    }
    input = next;
  }
  return input[0];
}

function binaryLoss(score, label) {
  return (label === 1 ? Math.max(-score, 0) : Math.max(score, 0))
    + Math.log1p(Math.exp(-Math.abs(score)));
}

export function networkMetrics(network, points) {
  if (!points.length) return { loss: NaN, accuracy: NaN };
  let loss = 0, correct = 0;
  for (const point of points) {
    const result = forwardNetwork(network, point.x, point.y);
    loss += binaryLoss(result.score, point.label);
    if (Number(result.score >= 0) === point.label) correct++;
  }
  return { loss: loss / points.length, accuracy: correct / points.length };
}

/** Average full-batch gradients, evaluated at a single unchanged parameter set.
 * Sigmoid + binary cross entropy simplifies the output delta to p - label.
 * Hidden delta = (W_nextᵀ delta_next) * activation'(preactivation).
 */
export function networkGradients(network, points) {
  if (!points.length) throw new Error('Training requires at least one point.');
  const layers = network.layers.map((layer) => ({
    weights: layer.weights.map((row) => Array(row.length).fill(0)),
    biases: Array(layer.biases.length).fill(0),
  }));
  let loss = 0;
  for (const point of points) {
    const result = forwardNetwork(network, point.x, point.y);
    loss += binaryLoss(result.score, point.label);
    let delta = [result.probability - point.label];
    for (let l = network.layers.length - 1; l >= 0; l--) {
      const input = result.activations[l];
      const gradient = layers[l];
      for (let j = 0; j < delta.length; j++) {
        gradient.biases[j] += delta[j];
        for (let k = 0; k < input.length; k++) gradient.weights[j][k] += delta[j] * input[k];
      }
      if (l > 0) {
        const previous = Array(input.length).fill(0);
        for (let k = 0; k < input.length; k++) {
          let upstream = 0;
          for (let j = 0; j < delta.length; j++) upstream += network.layers[l].weights[j][k] * delta[j];
          const derivative = network.activation === 'relu'
            ? Number(result.preActivations[l - 1][k] > 0)
            : 1 - input[k] * input[k];
          previous[k] = upstream * derivative;
        }
        delta = previous;
      }
    }
  }
  let normSquared = 0;
  for (const layer of layers) {
    for (let j = 0; j < layer.biases.length; j++) {
      layer.biases[j] /= points.length;
      normSquared += layer.biases[j] ** 2;
      for (let k = 0; k < layer.weights[j].length; k++) {
        layer.weights[j][k] /= points.length;
        normSquared += layer.weights[j][k] ** 2;
      }
    }
  }
  return { layers, norm: Math.sqrt(normSquared), loss: loss / points.length };
}

/** One epoch = one full-batch gradient update; all parameters update together.
 * No automatic differentiation, momentum, clipping, or hidden normalization.
 */
export function trainEpoch(network, points, learningRate) {
  if (!Number.isFinite(learningRate) || learningRate <= 0) throw new Error('Learning rate must be positive and finite.');
  const gradient = networkGradients(network, points);
  for (let l = 0; l < network.layers.length; l++) {
    const layer = network.layers[l];
    for (let j = 0; j < layer.biases.length; j++) {
      layer.biases[j] -= learningRate * gradient.layers[l].biases[j];
      for (let k = 0; k < layer.weights[j].length; k++) {
        layer.weights[j][k] -= learningRate * gradient.layers[l].weights[j][k];
      }
    }
  }
  return { ...networkMetrics(network, points), gradientNorm: gradient.norm };
}

export function makeNeuralDataset(kind = 'circles', seed = 42) {
  if (!['circles', 'xor', 'spirals'].includes(kind)) throw new Error('Choose circles, XOR, or spirals.');
  const random = randomSource(seed);
  const points = [];
  const count = 48;
  for (let label = 0; label <= 1; label++) {
    for (let i = 0; i < count; i++) {
      let x, y;
      if (kind === 'circles') {
        const angle = 2 * Math.PI * (i / count) + (random() - 0.5) * 0.06;
        const radius = (label === 0 ? 0.28 : 0.78) + (random() - 0.5) * 0.12;
        x = radius * Math.cos(angle);
        y = radius * Math.sin(angle);
      } else if (kind === 'xor') {
        const signX = i % 2 === 0 ? -1 : 1;
        x = signX * (0.22 + 0.65 * random());
        y = (label === 0 ? signX : -signX) * (0.22 + 0.65 * random());
      } else {
        const t = i / (count - 1);
        const radius = 0.12 + 0.76 * t;
        const angle = 2 * Math.PI * t + label * Math.PI;
        x = radius * Math.cos(angle) + (random() - 0.5) * 0.035;
        y = radius * Math.sin(angle) + (random() - 0.5) * 0.035;
      }
      points.push({ x, y, label });
    }
  }
  return points;
}
