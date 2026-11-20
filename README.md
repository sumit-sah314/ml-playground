# ML Playground

**Learn machine learning by moving points, adjusting parameters, and watching models train.**

ML Playground is a beginner-friendly, interactive web app with four visual lessons. The machine-learning math is implemented from scratch in plain JavaScript—no ML libraries, pretrained models, or external AI services.

**[Try the live app](https://visual-ml-lab.replit.app)** · **[GitHub repository](https://github.com/sumit-sah314/ml-playground)**

## Lessons

### 1. Linear Regression

- Click to add data points and drag them to change the dataset.
- Adjust the slope **w** and intercept **b** of `y = wx + b`.
- See vertical residuals and mean squared error update live.
- Animate gradient descent and follow the loss-versus-step chart.
- Compare the learned line against the exact least-squares solution.

### 2. Gradient Descent

- Explore a contour map of the actual MSE loss surface over **w** and **b**.
- Choose a starting point and watch the optimization path.
- Change the learning rate to see smooth convergence, oscillation, or divergence.
- Pause, take a single step, or reset the path.

### 3. Logistic Regression

- **Left-click** to place class A points and **right-click** for class B.
- Use the class selector for touch and pen input.
- Train with gradient descent and watch the decision boundary move.
- See prediction scores move along the sigmoid curve.
- Track binary log loss and training accuracy on separable or overlapping data.

### 4. Neural Networks

- Train a network with **2 inputs**, **1–3 hidden layers**, and **2–12 neurons per hidden layer**.
- Choose **tanh** or **ReLU** hidden activations and a sigmoid output.
- Learn **circles**, **XOR**, and **spirals** using handwritten backpropagation.
- Watch the nonlinear decision boundary, loss, and training accuracy update.
- Inspect a network diagram where edge thickness represents weight magnitude and color represents its sign.
- Learn why a straight logistic-regression boundary on raw coordinates cannot separate concentric circles, while nonlinear hidden layers can.

Every lesson includes a plain-English explanation, key equations, and a “Try this” challenge.

## How to use it

1. Open the [live app](https://visual-ml-lab.replit.app).
2. Pick a lesson from the sidebar.
3. Change the data or controls, then train or step through the updates.
4. Watch how the visualization and loss respond.

All learning runs in your browser. Experiments are kept in memory, not saved to an account or database; refreshing the page resets them. Reported accuracy is **training accuracy**, not a measure of performance on unseen data.

## Built with

- **React + Vite** for the application
- **JavaScript** for the machine-learning implementations and interactive lessons
- **TypeScript** for the app shell and supporting types
- **HTML Canvas and SVG** for plots and the network diagram
- **Tailwind CSS** for styling
- **pnpm** for dependency management

The learning algorithms use ordinary JavaScript arithmetic, arrays, and loops: least squares, analytical gradients, stable sigmoid and cross-entropy calculations, forward propagation, and backpropagation.

## Run locally

### Requirements

- Node.js **24 or newer**
- pnpm

```bash
git clone https://github.com/sumit-sah314/ml-playground.git
cd ml-playground
pnpm install
```

Start the web app from the repository root:

```bash
PORT=5173 BASE_PATH=/ pnpm --filter @workspace/ml-playground run dev
```

Open **http://localhost:5173**.

These environment-variable commands use a macOS/Linux shell, Git Bash, or WSL. `PORT` and `BASE_PATH` are required by the Vite configuration. The four lessons do **not** need API keys, authentication, a backend server, or a database.

### Production build

```bash
PORT=5173 BASE_PATH=/ pnpm --filter @workspace/ml-playground run build
```

The static output is written to `artifacts/ml-playground/dist/public`.

To preview that build locally:

```bash
PORT=5173 BASE_PATH=/ pnpm --filter @workspace/ml-playground run serve
```

For static hosting, configure an SPA fallback to `index.html` so direct links to lesson pages work.

## Checks

Build shared type declarations, then check the web app:

```bash
pnpm run typecheck:libs
pnpm --filter @workspace/ml-playground run typecheck
```

Run the neural-network numerical tests:

```bash
node --test artifacts/ml-playground/src/lib/neural-network.test.js
```

The tests check reproducible initialization, finite-difference verification of backpropagation, simultaneous parameter updates, numerical stability, and learning all three datasets with both activation functions.

## Project structure

```text
artifacts/ml-playground/
  src/
    App.tsx                  # App shell, sidebar, and lesson routes
    lessons/                 # Interactive lesson pages and visualizations
    lib/
      regression.js          # MSE, gradient descent, and least squares
      loss-surface.js        # Exact MSE contours and curvature
      logistic.js            # Sigmoid classification and binary log loss
      neural-network.js      # Forward pass, backpropagation, and datasets
      neural-network.test.js # Numerical tests
```

The repository also contains shared workspace packages, an API-server scaffold, and a component-preview sandbox. The four learning lessons are client-side and do not require those servers to run.

## Contributing

Suggestions and bug reports are welcome through [GitHub Issues](https://github.com/sumit-sah314/ml-playground/issues). For code changes, keep the math from scratch, explain concepts in beginner-friendly language, and include checks for numerical changes.
