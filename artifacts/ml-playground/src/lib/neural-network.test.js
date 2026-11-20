import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createNetwork, forwardNetwork, predictNetwork, networkMetrics,
  networkGradients, trainEpoch, makeNeuralDataset,
} from './neural-network.js';

test('initialization and all three datasets are reproducible', () => {
  assert.deepEqual(createNetwork([4, 3], 'tanh', 42), createNetwork([4, 3], 'tanh', 42));
  assert.notDeepEqual(createNetwork([4, 3], 'tanh', 42), createNetwork([4, 3], 'tanh', 43));
  for (const kind of ['circles', 'xor', 'spirals']) {
    const points = makeNeuralDataset(kind);
    assert.deepEqual(points, makeNeuralDataset(kind));
    assert.equal(points.length, 96);
    assert.equal(points.filter((point) => point.label === 0).length, 48);
    assert.ok(points.every((point) => Math.abs(point.x) <= 1 && Math.abs(point.y) <= 1));
  }
});

for (const activation of ['tanh', 'relu']) {
  test(`${activation}: backpropagation matches numerical derivatives of every weight and bias`, () => {
    const network = createNetwork([3, 2], activation, 17);
    const points = makeNeuralDataset('xor').filter((_, index) => index % 12 === 0);
    const before = JSON.stringify(network);
    const gradient = networkGradients(network, points);
    assert.equal(JSON.stringify(network), before, 'gradient calculation cannot change weights');
    const epsilon = 1e-5;
    for (let l = 0; l < network.layers.length; l++) {
      const layer = network.layers[l];
      for (let j = 0; j < layer.biases.length; j++) {
        for (let k = -1; k < layer.weights[j].length; k++) {
          const target = k === -1 ? layer.biases : layer.weights[j];
          const index = k === -1 ? j : k;
          const saved = target[index];
          target[index] = saved + epsilon;
          const plus = networkMetrics(network, points).loss;
          target[index] = saved - epsilon;
          const minus = networkMetrics(network, points).loss;
          target[index] = saved;
          const analytical = k === -1 ? gradient.layers[l].biases[j] : gradient.layers[l].weights[j][k];
          assert.ok(Math.abs((plus - minus) / (2 * epsilon) - analytical) < 1e-7,
            `incorrect derivative at layer ${l}, neuron ${j}, input ${k}`);
        }
      }
    }
  });

  test(`${activation}: one epoch applies the same pre-update gradient to every parameter`, () => {
    const network = createNetwork([3, 2], activation);
    const points = makeNeuralDataset('circles');
    const old = structuredClone(network);
    const gradient = networkGradients(network, points);
    const metrics = trainEpoch(network, points, 0.1);
    for (let l = 0; l < network.layers.length; l++) {
      for (let j = 0; j < network.layers[l].biases.length; j++) {
        assert.equal(network.layers[l].biases[j], old.layers[l].biases[j] - 0.1 * gradient.layers[l].biases[j]);
        for (let k = 0; k < network.layers[l].weights[j].length; k++) {
          assert.equal(network.layers[l].weights[j][k], old.layers[l].weights[j][k] - 0.1 * gradient.layers[l].weights[j][k]);
        }
      }
    }
    assert.equal(metrics.loss, networkMetrics(network, points).loss);
    assert.equal(predictNetwork(network, 0.2, -0.5), forwardNetwork(network, 0.2, -0.5).probability);
  });

  for (const kind of ['circles', 'xor', 'spirals']) {
    test(`${activation}: learns ${kind} from scratch with full-batch gradient descent`, () => {
      const network = createNetwork(kind === 'spirals' ? [12, 12] : [8, 8], activation, 42);
      const points = makeNeuralDataset(kind);
      const initial = networkMetrics(network, points).loss;
      let result;
      for (let epoch = 0; epoch < 3000; epoch++) result = trainEpoch(network, points, 0.1);
      assert.ok(Number.isFinite(result.loss));
      assert.ok(result.loss < initial * 0.5, 'loss should fall by more than half');
      assert.ok(result.accuracy >= 0.95, 'training accuracy should exceed 95%');
    });
  }
}

test('empty and invalid inputs fail explicitly; the output loss stays stable for extreme logits', () => {
  const network = createNetwork([]);
  network.layers[0].weights[0] = [1000, 0];
  network.layers[0].biases[0] = 0;
  assert.equal(networkMetrics(network, [{ x: 1, y: 0, label: 0 }]).loss, 1000);
  assert.equal(networkMetrics(network, [{ x: 1, y: 0, label: 1 }]).loss, 0);
  assert.ok(Number.isNaN(networkMetrics(network, []).loss));
  assert.throws(() => trainEpoch(network, [], 0.1));
  assert.throws(() => trainEpoch(network, makeNeuralDataset(), 0));
  assert.throws(() => createNetwork([0]));
  assert.throws(() => createNetwork([17]));
  assert.throws(() => createNetwork([4], 'sigmoid'));
});
