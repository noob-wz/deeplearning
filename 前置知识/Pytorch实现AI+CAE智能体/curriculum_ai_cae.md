# AI for Engineering Simulation 学习课程（完整大纲）

> **这是配套文档，不是 system prompt**。
> 
> 使用方式：当学生触发 Protocol 4（明确询问学习路径 / 下一步该学什么）时，
> 将本文件的相关章节作为参考载入对话。
> 
> 日常对话中，system prompt 本身不需要加载这份完整课程——过度加载会让模型
> 在回答简单问题时也进入"stage brain"状态。
>
> 课程原则：
> - Phase 1（Stage 0-6）路线中立，所有学生都学
> - Stage 7 正式决策路线
> - Phase 2（Stage 8-12）按选定路线深耕
> - Stage 13 作品集 + 面试

---

### Curriculum Reference (consult only when student requests a learning path)

The curriculum has two phases:

**Phase 1 — Route-Neutral Foundations (Stages 0 through 6)**
Every student learns the same material regardless of which technical route 
they eventually choose. These stages build the common skills that all AI 
for Engineering Simulation paths require.

**Phase 2 — Route-Specific Deepening (Stages 7 through 12)**
At Stage 7, the student makes an informed route choice. Stages 8 onward 
are tailored to that choice.

---

#### PHASE 1: ROUTE-NEUTRAL FOUNDATIONS

  Stage Pre — Foundations (typically pre-completed)
  Prerequisites: None.
  Goal: Convert "can watch tutorials" into "can write code independently."
  Topics: Python classes and modules, NumPy arrays and shape manipulation, 
  vectorization and broadcasting, Matplotlib for loss curves and field 
  visualization, geometric intuition of derivatives and gradients, basic 
  FEM file parsing as a Python engineering exercise, Git basics.
  Completion signal: Student can independently write a function that parses 
  a simple FEM-like text file, convert node coordinates into a NumPy array, 
  and produce a clean visualization. Can explain what broadcasting does 
  for a specific shape combination without running the code.

  Stage 0 — PyTorch Mechanics
  Prerequisites: Stage Pre.
  Goal: Understand what a Tensor is and how PyTorch tracks computation.
  Topics: Tensor vs NumPy array, tensor operations and shape manipulation, 
  the computational graph, autograd and .backward(), building a single 
  neuron and then a small MLP by hand using raw tensor operations only — 
  no nn.Module yet.
  Completion signal: Student can write a forward pass and manual weight 
  update for a small MLP, can predict output shapes before running code, 
  and can explain when the computational graph is built and when it is freed.

  Stage 1 — Deep Learning Fundamentals + Minimal Training Loop
  Prerequisites: Stage 0.
  Goal: Write a complete training loop from scratch and understand what every 
  line is doing and why.
  Topics: Loss functions, gradient descent mechanics, overfitting and 
  underfitting, regularization, backpropagation mechanics through an MLP, 
  nn.Module training loop, Optimizer (SGD then Adam), DataLoader with 
  TensorDataset.
  Project: Multi-feature regression on a synthetic dataset that resembles 
  a simple engineering mapping (e.g., predicting displacement from 
  force/geometry parameters). Student must plot train loss, val loss, 
  and prediction quality independently.
  Completion signal: Student can answer without notes — why does overfitting 
  happen, why is val loss typically higher than train loss, what does Adam 
  do differently from SGD, what gets saved when you checkpoint a model.

  Stage 2 — Training Discipline: Optimization, Tuning, Error Analysis
  Prerequisites: Stage 1.
  Goal: Be able to look at a loss curve and know what is wrong.
  Topics: Diagnosing loss curves (exploding, plateauing, oscillating, 
  diverging), Adam / learning rate schedules / normalization (Z-score for 
  regression), basic error analysis for regression problems (where does 
  the model fail?), controlled experiment design for ML.
  Project: Upgrade Stage 1 regressor — add feature normalization, compare 
  optimizers, compare LR schedules, write experiment report with controlled 
  variables.
  Deliverables: baseline, adam_vs_sgd, normalization_ablation, 
  error_analysis.md.
  Completion signal: "I have trained a model and I know how to diagnose it 
  when training goes wrong." The student should be able to look at an 
  unfamiliar loss curve and propose three specific hypotheses.

  Stage 3 — Autograd Mastery and Physics Loss
  Prerequisites: Stage 2.
  Goal: Understand autograd deeply enough to write a physics-informed loss 
  that correctly embeds a PDE residual as a penalty term.
  Why this stage comes before GNN: every downstream route (PINN, GNN-based 
  simulator with physics loss, neural operator with physics constraints) 
  depends on the ability to compute derivatives of network outputs with 
  respect to inputs. This is a shared infrastructure skill.
  Topics: `torch.autograd.grad` vs `.backward()`, higher-order derivatives 
  with `create_graph=True`, partial derivatives of multi-input multi-output 
  functions, translating a PDE into a residual loss term, boundary condition 
  losses, weighting strategies for multi-term physics losses.
  Project: 1D elastic bar PINN — given boundary conditions, predict 
  displacement field u(x) such that the equilibrium equation E·d²u/dx² + f = 0 
  is satisfied. Verify against analytical solution.
  Completion signal: Student can implement a physics loss for an unseen 
  simple PDE (e.g., 1D heat equation) without reference code, and can 
  explain why `create_graph=True` is needed for second-order derivatives.

  Stage 4 — Graph Neural Network Mechanics
  Prerequisites: Stage 2 (Stage 3 helpful but not required).
  Goal: Understand the message passing paradigm as a designed mechanism, 
  not a black-box API call.
  Why before mesh simulators: mesh-based neural simulators are built on GNN 
  foundations. The student needs to understand message passing at the 
  mechanical level before assembling it into a simulator architecture.
  Topics: Graph data representation (nodes, edges, features), the PyTorch 
  Geometric `Data` object, GCN derivation (normalized adjacency, 
  Laplacian), GraphSAGE (sample-and-aggregate, inductive vs transductive 
  learning), the MessagePassing base class, over-smoothing and why it 
  happens, residual connections in GNNs.
  Project: Build a small GNN from scratch using `MessagePassing`, train it 
  on a synthetic graph regression task (e.g., predict a node property from 
  neighborhood structure), compare GCN vs GraphSAGE on the same data.
  Completion signal: Student can implement a custom message and aggregate 
  function in PyG, explain why GraphSAGE supports inductive learning while 
  GCN does not, and predict output shapes through a multi-layer GNN.

  Stage 5 — FEM Data Processing for ML
  Prerequisites: Stage 4 (parallel study with Stage 3 acceptable).
  Goal: Transform real FEM simulation data into a form that can be fed 
  into any neural model (GNN, MLP per node, or pointwise for operators).
  Topics: Anatomy of FEM data (nodes, elements, connectivity, materials, 
  boundary conditions, results), parsing common file formats (conceptually 
  .inp / .k / .msh via meshio), handling mixed element types (triangle, 
  quad, tet, hex), mesh-to-graph conversion (node-as-vertex with 
  element-derived edges), per-node and per-element feature construction, 
  data normalization strategies for physics fields, batching graph data 
  with PyG's `Batch` and block-diagonal aggregation.
  Tools introduced: PyTorch Geometric in depth, meshio for format 
  conversion, Gmsh for generating simple test meshes.
  Project: Given a set of synthetic FEM-like data (simple 2D geometry 
  with node displacements and element stresses), build a reusable 
  `mesh_to_pyg` function and a `Dataset` class that produces batched 
  graph samples ready for any downstream model.
  Completion signal: Student can handle a mesh with mixed element types 
  and produce a correctly batched PyG dataset without reference code. Can 
  explain what goes wrong in batched training if element-to-node index 
  offsets are not handled.

  Stage 6 — First Real Neural Simulator: MeshGraphNets-Style Baseline
  Prerequisites: Stages 3, 4, 5.
  Goal: Build one complete mesh-based neural simulator end to end, using 
  the MeshGraphNets architecture as the most standard and well-documented 
  reference. This serves as the baseline against which all other routes 
  will be compared at Stage 7.
  Why MeshGraphNets specifically as the Stage 6 reference: it is the most 
  widely cited mesh-based neural simulator, has DeepMind's official JAX 
  implementation plus multiple community PyTorch ports, and its 
  encoder-processor-decoder architecture is the template that PhyFENet, 
  Transolver, and many other methods build on. Learning it makes every 
  other route's design decisions visible as variations on a theme.
  Topics: The encoder-processor-decoder architecture (paper track), 
  edge-update message passing (paper + code track), training on simulation 
  rollouts vs single-step prediction, evaluating a neural simulator 
  (per-step error vs trajectory error), generating synthetic training 
  data with FEniCS / FEniCSx.
  Style 4 (李沐) activates here: present MeshGraphNets paper intent, 
  code implementation, and ablation results in parallel.
  Project: Implement a minimal MeshGraphNets on a 2D elastic problem 
  using FEniCS-generated training data. Must include data generation 
  pipeline, training loop, evaluation on held-out meshes, and 
  visualization of predicted vs ground-truth fields.
  Completion signal: Student can explain each line of the 
  encoder-processor-decoder architecture, diagnose a typical failure mode 
  (e.g., error accumulation over rollout steps), and articulate where 
  MeshGraphNets' assumptions might fail.

---

#### PHASE 2: ROUTE-SPECIFIC DEEPENING

  Stage 7 — Route Survey and Informed Choice (CRITICAL DECISION POINT)
  Prerequisites: Stages 0 through 6.
  Goal: Make an informed, deliberate choice of technical route before 
  committing 4+ months to a specific deep path. This is not a "stage of 
  learning" in the same sense as the others — it is a structured 
  decision-making process.

  Stage 7 has two versions. The student chooses which to run:

  Stage 7-Structured (default for students without strong preferences):
    A 2-3 week mini process:
    - Days 1-3: Read one recent survey paper on neural PDE solvers / neural 
      simulators covering PINN, GNN simulators, and neural operators. Write 
      a one-page comparison table.
    - Days 4-10: Minimal reimplementation of two competing methods on the 
      same toy problem (e.g., 2D Poisson or 2D elasticity). Recommended 
      pairings: (MeshGraphNets + FNO), (PINN + MeshGraphNets), or 
      (PhyFENet + FNO).
    - Days 11-12: Job market signal gathering — pull 20-30 current job 
      postings, tally the technical keywords that appear.
    - Days 13-14: Consultation with 2-3 domain practitioners if possible 
      (academic + industry); formal route decision with written rationale.

  Stage 7-Direct (for students with clear preference):
    Student skips to the deep stage matching their preference. Prompt 
    still requires them to write a one-page rationale: what did you 
    consider, what did you reject, why this route.

  Route options emerging from Stage 7:
    Route A — PINN deep specialization: adaptive sampling, variational 
      PINN, conservative PINN, domain decomposition.
    Route B — GNN-based simulator specialization: MeshGraphNets extensions, 
      PhyFENet-style physics embedding, multi-scale GNN, temporal rollout 
      strategies.
    Route C — Neural Operator specialization: FNO variants, DeepONet, 
      Geo-FNO for complex geometries, physics-informed neural operators.
    Route D — Transformer-based simulator: Transolver, GNOT, attention 
      mechanisms on unstructured meshes.
    Route E — Hybrid specialization: the student identifies a combination 
      (e.g., GNN + physics loss, as PhyFENet does) and commits to that.

  Completion signal: Student can articulate in one paragraph why they 
  chose their route, what they gave up by not choosing the others, and 
  what specific job category or research direction their choice 
  targets.

  Stage 8 — Route-Specific Architecture Deep Dive
  Prerequisites: Stage 7 route chosen.
  Goal: Master the core architecture of the chosen route at implementation 
  level — able to write it from scratch, reproduce a paper's key result, 
  and identify its failure modes.
  Content: Tailored to the chosen route. For example:
    - If Route C (Neural Operator): deep dive into FNO's Fourier layer 
      mechanics, spectral convolution, resolution invariance, extensions 
      like Geo-FNO for irregular geometries.
    - If Route B (GNN Simulator): multi-level GNN, physics loss integration 
      (if going PhyFENet-style), temporal rollout stability techniques, 
      multi-scale mesh representations.
  Style 4 (李沐) active throughout: every named technique introduced 
  via paper + code + ablation.

  Stage 9 — Industrial Data and Realistic Constraints
  Prerequisites: Stage 8.
  Goal: Move from synthetic / academic datasets to the challenges of 
  real engineering data.
  Topics: Handling industrial mesh quality issues, dealing with limited 
  training data (small sample regimes), transfer learning across 
  materials / geometries / boundary conditions, conceptual familiarity 
  with Abaqus / LS-DYNA / HyperMesh file structures, benchmarking 
  neural surrogates against ground-truth solver runtime and accuracy, 
  understanding the gap between academic benchmarks and industrial 
  deployment requirements.
  Project: Take the route-specific model from Stage 8 and train it on 
  a realistic engineering problem (parameterized 2D or 3D setup) with 
  proper train/test splits, error analysis by region / boundary condition 
  / material parameter.

  Stage 10 — Engineering Project and Code Discipline
  Prerequisites: Stage 9.
  Goal: Take the route-specific model from Stage 8–9 and refactor it into 
  a real engineering project. This stage is less about new ML concepts and 
  more about learning to write code that can be maintained, extended, and 
  handed off. For a self-taught cross-trainer, this is the single largest 
  gap to close before interviewing.
  Style 6 (Senior Engineer Code Review Mode) activates here as the 
  default teaching mode.
  Topics — Project organization:
    - Canonical project layout: configs/ (Hydra or OmegaConf YAML), 
      src/ (reusable modules), scripts/ (entry points), tests/ (pytest), 
      data/ (versioned artifacts), notebooks/ (exploration only, 
      not production code).
    - Separating concerns: data loading module, model module, training 
      module, evaluation module — each testable in isolation.
    - Configuration over hardcoding: all hyperparameters, paths, 
      and experimental knobs live in config files. A new experiment 
      is a new config, not a modified script.
  Topics — Testing ML code:
    - pytest fundamentals for ML codebases.
    - What to test in a neural surrogate: data loader correctness 
      (shape, dtype, NaN absence), forward pass shape contract, 
      physics loss correctness on a known analytical case, training 
      step produces decreasing loss over 10 steps.
    - What not to test: exact numerical outputs across GPU/CPU, 
      full training convergence.
  Topics — Logging and experiment tracking:
    - Structured logging: every training run produces a timestamped 
      directory with config snapshot, git commit hash, environment 
      info, loss curves, sample predictions.
    - WandB or TensorBoard — introduced here for the first time as a 
      mandatory tool, not an optional one.
    - Naming experiments: you will run 50+ experiments by Stage 12; 
      if the 30th one has no interpretable name, it's lost.
  Project: Refactor the Stage 9 model into a proper project. Produce 
  a repository that another engineer could clone, run `pip install -e .`, 
  run `pytest`, run `python scripts/train.py --config configs/baseline.yaml`, 
  and reproduce your results. The repository must pass Style 6 review.
  Completion signal: Student's project repository is one that the mentor 
  would hire the author based on. No configuration is hardcoded. Every 
  module has tests. The README is usable by a stranger.

  Stage 11 — Scale and Real Data Engineering
  Prerequisites: Stage 10.
  Goal: Move from "works on synthetic data" to "works on real engineering 
  data at real scale." This stage is about the engineering concerns that 
  academic papers consistently ignore but industrial roles require.
  Topics — Real data quality:
    - Data quality audit framework for FEM datasets: detecting 
      non-converged solver runs, identifying degenerate meshes (inverted 
      elements, duplicate nodes, disconnected regions), finding outlier 
      samples where physical fields exceed reasonable bounds.
    - Systematic handling of missing data: interpolation vs exclusion 
      vs imputation, and the consequences of each choice for downstream 
      training.
    - Data versioning concepts (DVC or plain git-LFS): why you need to 
      version the data alongside the code, how reproducibility breaks 
      when the data silently changes.
  Topics — Memory and scale:
    - When the dataset exceeds RAM: memory-mapped NumPy arrays, 
      PyTorch's IterableDataset for streaming, lazy loading patterns 
      for mesh data.
    - When a single mesh exceeds GPU memory: chunked processing, 
      gradient checkpointing, sparse operations in PyG.
    - Multi-GPU training: DDP (DistributedDataParallel) from scratch, 
      understanding what gets synchronized and when, common DDP bugs 
      (unused parameters, uneven workloads across graphs of varying 
      size).
    - Mixed precision (AMP): where it helps, where it hurts (physics 
      loss involving higher-order derivatives needs careful handling 
      in FP16).
    - Gradient accumulation for small GPU budgets.
  Topics — Evaluation with engineering rigor:
    - Bucketed error analysis: error by geometry complexity, by material 
      parameter, by boundary condition type, by mesh density.
    - Out-of-distribution testing: hold out geometries, materials, or 
      loading conditions the model never saw.
    - Robustness probes: what happens with slightly perturbed inputs, 
      with partially noisy boundary conditions.
    - Fair benchmarking: when comparing to a traditional solver, is the 
      solver's mesh the same as the model's? Is the hardware matched? 
      Are you counting preprocessing time?
  Project: Take a publicly available realistic dataset (or one generated 
  from FEniCS with industrial-scale complexity) with at least 1000 
  samples. Produce a production-quality training and evaluation pipeline 
  that handles data quality issues, trains with DDP on multi-GPU if 
  available (or documents how it would), and produces a bucketed error 
  analysis report.
  Completion signal: Student can identify three distinct data quality 
  problems in an unfamiliar FEM dataset within 30 minutes, design the 
  bucketed evaluation strategy for a novel problem, and explain what 
  happens to their physics loss under AMP.

  Stage 12 — Deployment and Integration (Dual Track)
  Prerequisites: Stage 11.
  Goal: Take a trained neural surrogate model and make it usable in two 
  distinct settings — as a generic ML service (ONNX / TensorRT / API), 
  and as a component integrated with commercial CAE toolchains. The 
  target student needs fluency in both tracks: the CAE integration track 
  is critical for the specific target role (AI + CAE engineer), and the 
  general ML deployment track provides career insurance if the narrow 
  AI + CAE market is not accessible.

  Track A — General ML Deployment (~50% of Stage 12):
  Topics:
    - ONNX export in depth: what's supported, what's not (especially 
      relevant for PyG / custom autograd operations / higher-order 
      derivative models), common export failures and how to diagnose 
      them.
    - ONNX Runtime for CPU inference, TensorRT for GPU inference, 
      when each is appropriate.
    - Inference benchmarking: throughput, latency, memory, cold vs 
      warm start — producing a comparison table the business side 
      can actually read.
    - Model optimization: quantization (INT8 and its impact on physics 
      correctness), pruning (structured vs unstructured), distillation 
      concepts. Warning: quantization often breaks physics-informed 
      models. Demonstrate this experimentally, not theoretically.
    - Serving the model: FastAPI wrapper, batching strategies, 
      handling variable-size mesh inputs via padding or dynamic batching.
    - Docker containerization for reproducible deployment environments.
  Deliverable: A containerized inference service that takes a mesh as 
  input and returns predicted fields via HTTP API, with a benchmark 
  report comparing PyTorch / ONNX Runtime / TensorRT on at least three 
  input sizes.

  Track B — CAE Toolchain Integration (~50% of Stage 12):
  Topics:
    - Anatomy of commercial CAE Python APIs: Abaqus Python scripting 
      (abaqus CAE plugins), ANSYS ACT and the ANSYS Python APIs, 
      LS-DYNA keyword file programmatic generation.
    - File-level integration patterns: writing Python tools that 
      parse .inp / .k files, substitute neural surrogate predictions 
      for specific result fields, write back compliant output.
    - Co-simulation patterns: when the neural model replaces one 
      physics module in a multi-physics simulation, the interface 
      contract (which fields in, which fields out, on which mesh), 
      how time-stepping is coordinated.
    - User-defined subroutines at interface level: UMAT (user material 
      subroutine in Abaqus/LS-DYNA) — you probably won't write Fortran, 
      but you need to understand what the interface expects so your 
      neural surrogate can be wrapped as one. Conceptually, not 
      implementation-level.
    - Practical constraints: licensing (commercial CAE licenses are 
      floating and limited), version compatibility (Abaqus 2022 
      vs 2024 APIs differ), deployment to engineers who don't have 
      Python environments.
    - Honest discussion of the state of this space: most production 
      AI + CAE integrations today are prototypes, not deployed 
      systems. The student's goal is to understand the integration 
      patterns well enough to lead one of these prototypes, not to 
      claim fluency with a mature ecosystem.
  Deliverable: One integration demo. Options:
    - Option 1: A Python script that reads an Abaqus .inp file, runs 
      the neural surrogate prediction, and writes back an output file 
      compatible with Abaqus post-processing. Demonstrates the 
      file-level integration pattern end to end.
    - Option 2: A conceptual design document for a UMAT-style wrapper 
      around the neural surrogate, with pseudocode showing the 
      interface contract and a clear explanation of what real 
      implementation would require.
  The student should produce at least one of these. If time allows, 
  both.

  Stage 12 integration note: the two tracks are not independent. A 
  realistic AI + CAE deployment requires the generic ML deployment 
  stack (model is exported, optimized, served as an API) underneath 
  the CAE integration layer (CAE tool calls the API, gets predictions, 
  integrates them into the simulation). Treat them as layers, not as 
  alternatives.

  Completion signal: Student can walk through the full path from 
  trained PyTorch model to a value being consumed by Abaqus or 
  LS-DYNA, naming each intermediate format and the failure modes 
  at each transition. Can estimate the engineering effort required 
  for each step.

  Stage 13 — Portfolio and Interview Preparation
  Prerequisites: All target stages complete.
  Goal: Three representative projects, each answerable in an interview 
  across six dimensions: problem, data, model, metrics, hardest issue 
  encountered, improvement made over baseline.
  Portfolio structure:
    - Project 1: Route-specific deep implementation (from Stage 8 or 
      10). Demonstrates ML depth in the chosen technical direction.
    - Project 2: Route-neutral foundational project (MeshGraphNets from 
      Stage 6, refactored with Stage 10 engineering discipline). 
      Demonstrates breadth and foundational understanding.
    - Project 3: One of:
      (a) The industrial-scale engineering project from Stage 11 
          (with bucketed error analysis and real data quality work).
      (b) The Stage 12 deployment/integration demo (CAE integration 
          or ML service).
      Project 3 is the "engineering maturity" signal — pick whichever 
      reflects better on the student's specific target role.
  Interview topics:
    - Foundational mechanics: autograd with create_graph, 
      computational graph lifecycle, higher-order derivatives for 
      physics loss.
    - Architecture derivations: message passing, MeshGraphNets 
      encoder-processor-decoder, the chosen route's core architecture.
    - FEM data literacy: node-element relationships, mesh quality, 
      boundary conditions, what a .inp file contains.
    - Route comparison: why you chose your route, when you'd pick a 
      different one, honest limitations of your chosen route.
    - Engineering judgment: how you'd handle dataset quality issues, 
      reproducibility, deployment tradeoffs.
    - Industry literacy: honest discussion of simulation-to-reality 
      gap, state of AI + CAE deployment in industry (mostly 
      prototypes, not production), what a realistic 6-month scope 
      looks like at a new role.
  Mock interview protocol: conduct at least three mock interviews, 
  each covering a different angle (technical depth, project 
  presentation, system design / extension question). After each mock, 
  the mentor produces a written evaluation in the same format a real 
  interviewer would — not a cheerleading summary.
  Completion signal: Student can defend every line of their portfolio, 
  articulate the engineering tradeoffs in every major design decision, 
  and discuss their chosen route's limitations without sounding 
  defensive.

---

### Learning Path Flexibility Rules

  1. ASSESS FIRST: Ask what the student has already built or studied. Skip 
     stages where they demonstrate working engineering-level competence, not 
     just familiarity.

  2. ALLOW JUMPING: If the student wants to start at a specific stage, allow 
     it. Briefly check prerequisites: "This stage builds on X and Y. 
     Comfortable with both? If not, I'll cover them as we go."

  3. GOAL-DRIVEN REORDERING:
     - Student wants interview ready fast → Stages 0-1-2-4-6 (skip Stage 3 
       initially, return to PINN only if a targeted role requires it)
     - Student has strong PINN interest → Stage 3 expanded into multiple 
       weeks, Stage 4 lighter, jump to Stage 7-Direct with Route A chosen
     - Student has strong operator learning interest → Stage 6 remains 
       MeshGraphNets for comparison baseline, then Stage 7-Direct with 
       Route C chosen

  4. PRESENT AS A MENU: Always offer 2-3 options with rationale. 
     Let the student choose.

  5. HONOR ROUTE NEUTRALITY IN PHASE 1: Never collapse the curriculum into 
     "just learn PhyFENet" or "just learn FNO" in Phase 1. The purpose of 
     Stages 0-6 is to leave the decision open until the student has the 
     context to make it well.

---

