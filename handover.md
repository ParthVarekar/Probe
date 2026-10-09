# PROBE — Session Handover

> **Purpose:** let any Claude session pick up the PROBE founding-CTO planning assignment if the current session ends midway.
> **Branch:** `claude/pensive-ptolemy-huq2mx` · **Originating session:** https://claude.ai/code/session_01WyyihpQzFz2ZYAdX8EKpJj
> **Last updated:** 2026-10-09. See the update log at the bottom.

---

## 0. TL;DR for the next session

- **Phase: PLANNING ONLY.** A strict approval gate is in force (§1). Do **not** implement anything.
- **Only exception:** this `handover.md` file. The user explicitly asked for it so work survives session loss. No other repository file may be created or modified before the user approves the final plan.
- **Progress:** Stage 1 (reconnaissance) is **done**. Stage 2 (research) is **in progress**: an 11-dimension research workflow with adversarial source verification was launched in the originating session. Its results are **not yet recorded here**. If §7 has no research digests, re-run the workflow in §9.
- **Next:** finish research → run the design workflow (competing strategies, maths, benchmark, red team, threat model) → write the final plan (Parts A–Q, §3) → present it in Plan Mode → **stop and wait for explicit approval**.
- The original user brief is the authoritative source. §2 is a faithful condensation of it. If the user can paste the original, prefer it.

---

## 1. Hard rules (approval gate; these override everything else)

The current phase is **planning only**. Permitted: read-only repository inspection, external research (web search/fetch, reading docs, papers and source repos), and temporary in-memory analysis or calculations.

**Forbidden before explicit user approval of the final plan:**
- Creating application source files, editing source files, or creating implementation documentation in the repo. `handover.md` is the sole user-authorized exception.
- Installing or upgrading dependencies. Creating databases or persistent app state. Running migrations. Changing project configuration.
- Modifying tracked or untracked project files, other than `handover.md`.
- Starting implementation, or launching subagents that modify files.
- Committing changes, other than `handover.md` updates. Deploying.
- Provisioning paid cloud resources, purchasing subscriptions, or consuming paid credits on external services.
- Starting implementation because a promising existing project was found.
- Creating plan or scratch files or task trackers **in the repo**. Present the plan in the response or through Plan Mode's plan file, which lives outside the repo.

Also:
- If formal Plan Mode is available, use it for the assignment. It was exited only to commit this handover, with user approval. Re-enter it (`EnterPlanMode`) to present the final plan, then `ExitPlanMode` to request approval.
- Do not claim tests, benchmarks or experiments were run when they were not.
- Never fabricate research findings, citations, costs, project files, capabilities or benchmark results.
- If a research source or tool is unavailable, report it and make a clearly labelled provisional recommendation.
- No PR unless the user explicitly asks. Push only to `claude/pensive-ptolemy-huq2mx`.

---

## 2. The mission brief (faithful condensation)

The user plays founder. Claude acts as founding CTO, principal AI research engineer, scientific-method specialist, staff architect, product strategist and implementation lead for **PROBE**.

### 2.1 Ambition (brief §1)
- **Not** another chatbot, personal assistant, multi-agent framework, RAG app, AI research summarizer, agent dashboard, synthetic-data generator, or generic "automated scientist" that writes convincing reports. These may be components; they are not the product.
- **Core question:** can an AI system autonomously turn a question about the world into a well-designed experiment, obtain real or simulated evidence, distinguish between competing explanations, decide which observation to collect next, and produce a reproducible conclusion whose validity can be independently checked?
- It must **perform** the experimental workflow, not just recommend steps.
- It must distinguish what it **suspects**, **predicts**, **observed**, and what its evidence **supports**.
- **Outputs:** executable protocols, structured hypotheses, recorded observations, independently verifiable calculations, uncertainty estimates, experiment histories, reproducible research artifacts. It investigates questions rather than retrieving information.
- **Long-term vision:** a general-purpose experimental discovery engine and platform where humans and AI formulate questions, run experiments, share validated protocols, contribute observations and build on reproducible findings. Users could be scientists (environment), engineers (how an unfamiliar software system behaves), companies (product hypotheses), or AI systems (resolving uncertainty about an unfamiliar API/environment).
- **Do not** pretend v1 achieves universal scientific intelligence. Build a narrow, technically credible foundation that extends toward the vision.
- **Goal:** find out whether there is a meaningful capability we can build *better than existing alternatives*, demonstrate it with real evidence, and possibly turn it into a business. An impressive demo is not enough.

### 2.2 Central research questions (brief §2)
- **Hypothesis A (adaptive experimental efficiency):** can adaptive, evidence-driven experiment selection produce more reliable conclusions per unit of experimental cost than fixed or random sampling, while staying reproducible and representing uncertainty properly?
- **Hypothesis B (learning to investigate unfamiliar environments):** can a system learn to select useful experiments in an unfamiliar environment by reasoning about competing explanations and possible observations, rather than following a fixed plan?
- These are hypotheses, not results. Be willing to find the methods do not beat baselines. Define what would count as convincing evidence for or against each.
- "AI-driven scientific experimentation" in general is **not** a sufficient novelty claim. Find a more specific, experimentally testable, differentiated contribution. If existing work already covers the direction, recommend a sharper hypothesis or a stronger product direction. Do not defend the original idea just because it came first.

### 2.3 Outcome over implementation method (brief §3, non-negotiable)
- No need to build everything from scratch. Find and assemble the **best possible solution**. The method matters less than the quality of the outcome. Everything in the brief is a starting point, not a fixed decision.
- **3.1 Explore the full solution space:** mature OSS, research repos, scientific computing / statistics / DoE libraries, active learning and Bayesian optimization frameworks, AI-scientist and experiment-orchestration platforms, commercial APIs, models (API and local), agent frameworks, execution and workflow engines, sensor platforms, datasets, ingestion tools, scientific visualization, sandboxing and containerization, reproducibility tech, cloud and managed services, implementations of relevant papers, commercial products to integrate or extend, and entirely different approaches. Search beyond familiar tools.
- **3.2 For every major capability, evaluate build / buy / integrate / reuse / adapt** on: capability and quality, reliability and limitations, maturity and maintenance, compatibility, integration complexity, development time, operational complexity, performance and scalability, security and privacy, licensing and commercial-use rights, API availability, rate limits and service restrictions, initial and recurring cost, cost at scale, vendor dependence and portability, extensibility, reproducibility and inspectability, and impact on differentiation. Choose on evidence, not popularity.
- **3.3 Reuse aggressively but intelligently.** Use established components for commodity capabilities. Focus custom engineering on differentiation, closing capability gaps, defensibility, or results others can't reach. Don't hand-build basic infrastructure. Don't glue random APIs together and call it a breakthrough. Before recommending a component, check its real capabilities, source, limitations, license, maintenance, compatibility and commercial suitability. For services, check failure modes, pricing, rate limits, data handling and portability. Proprietary dependencies need a documented fallback or migration path.
- **3.4 Look for the strongest existing implementations** for adaptive experimental design, statistical inference, simulation, data validation, experiment orchestration, provenance, reproducible execution, AI orchestration, scientific workflow engines, report generation, security and sandboxing, and benchmarking. Read papers, docs, repos, benchmarks and commercial offerings, and inspect actual code where practical. Six routes: (1) use as-is; (2) integrate behind an abstraction; (3) adapt or extend; (4) combine several; (5) replace a small part with custom code; (6) build new because the alternatives are inadequate. Justify the route chosen. Unverified or inaccessible implementations do not count as validated.
- **3.5 Don't force novelty into every component.** A possible differentiator is the combination of experimental planning, adaptive information acquisition, evidence verification and reproducibility, but do not assume it is novel. Let prior art decide. Never inflate an integration into a "revolution". A novel algorithm is worthless if it doesn't beat alternatives or solve a real problem.
- **3.6 Commercial tech is allowed** if it materially improves the outcome. You may investigate pricing, limits, terms and free tiers, and compare with OSS and local options. **Never** purchase, consume paid credits, provision billable resources or commit the user financially without explicit authorization. Never assume keys, accounts, licenses, free tiers or credits exist. If the best option is paid, name the dependency, estimate its cost and give alternatives. The initial demo must be practical and affordable.
- **3.7 At least three materially different solution strategies**, compared on: achievable capabilities, technical risk, reliability, research differentiation, development complexity, performance, cost, operational complexity, time to a credible demo, product and startup potential, and long-term extensibility. They must differ in substance, not just library choice. If fewer than three are credible, explain why.
- **3.8 The planning package must include** (in the plan, **not** written to the repo):
  1. Technology and Implementation Landscape.
  2. Build-vs-Buy-vs-Integrate Decision Matrix.
  3. Three Competing Architecture Strategies.
  4. Reuse and Extension Plan (use directly / adapt / custom).
  5. Dependency and Commercial-Risk Register (licenses, vendor dependencies, access restrictions, recurring costs, maintenance risks, fallbacks).
  6. Differentiation Analysis (differentiating vs commodity vs what could make PROBE undifferentiated).
  7. Recommendation and Rejected Alternatives, plus the conditions that would change the recommendation.

### 2.4 Research before architecture (brief §4)
- **Investigate:** AI for scientific discovery; automated experiment design; self-driving labs; active learning; Bayesian experimental design; expected information gain; hypothesis generation and evaluation; agentic scientific workflows; sensor networks and participatory sensing; causal inference and intervention-based reasoning; AI-generated experimental protocols; autonomous data collection and QC; reproducible computational science; scientific workflow orchestration; AI that selects and executes its own next experiments; related commercial products, startups and OSS alternatives.
- **For each important precedent:** (1) what it actually does; (2) implemented vs proposed capabilities; (3) its central technical contribution; (4) evidence for its claims; (5) benchmarks and baselines; (6) remaining limitations; (7) what it doesn't address; (8) overlap with and difference from PROBE.
- Cover direct competitors and adjacent products that could expand into the space. Using LLMs, multiple agents, a new UI or a new brand does not make something innovative.
- Record source links, publication dates and research status, separating papers, preprints, product claims and proposals. Never fabricate or imply a source was reviewed if it wasn't.
- **Required outputs:**
  - A related-work matrix: title, authors/org, date, link, central contribution, evidence, limitations, relevance.
  - Proposed contents (in the plan, not the repo) for `RELATED_WORK.md`, `COMPETITIVE_LANDSCAPE.md`, `RESEARCH_HYPOTHESES.md` and `ARCHITECTURE_DECISIONS.md`.
  - An **explicit answer to:** *"What is the most defensible technical and commercial differentiation available to us, given what already exists?"* If the idea is weakly differentiated, recommend a sharper hypothesis before implementation.

### 2.5 Initial application: campus microclimate discovery (brief §5)
- **Question:** how do shade, surface type, location and time of day relate to measured pedestrian-level temperature and thermal conditions? Which additional measurements would best distinguish the plausible explanations?
- This is a **candidate**, not permanent. Its advantages: understandable measurements, competing hypotheses, confounders, real observations are possible, a synthetic environment with known ground truth, quantitative evaluation of adaptive sampling, intuitive visuals, and a path to a data service.
- **Must consider** whether a different initial domain gives a better demo of the core tech or stronger startup potential, and decide on evidence.
- **Example hypotheses (not conclusions):**
  - Shaded locations have lower measured surface temperatures than comparable exposed ones under specified conditions.
  - Surface material explains more variation than shade.
  - An apparent shade effect is confounded by time, weather, location or sensor calibration.
  - The available observations cannot distinguish the explanations.
- Distinguish descriptive, predictive and causal analysis. Never claim causation from association. Specify how confounders are identified, how limitations are communicated, and how unsupported causal claims are prevented.

### 2.6 Three data modes (brief §6)
- **Mode A, deterministic simulation:** a synthetic environment with documented ground-truth parameters. It should eventually support configurable relationships, measurement noise, confounders, missing observations and sensor-error mechanisms. Always label results as **simulated**. Evaluate reusing existing simulation frameworks.
- **Mode B, imported datasets:** CSV import with column mapping to variables, units, timestamps, conditions and sources. It should support schema validation, missing-value handling, duplicate detection, unit validation, source preservation, data-quality warnings and preview before import. Keep the original data separate from normalized and transformed data. Evaluate existing ingestion and validation libraries.
- **Mode C, new real-world observations:** manual measurement entry plus a documented integration interface for external sensors, e.g. generic HTTP ingestion. Further integrations must be justified. The core demo must not depend on hardware or paid services. Evaluate existing sensor platforms and public datasets.

### 2.7 The scientific engine (brief §7)
Recommend existing libraries where they beat custom code.
- **7.1 Structured, versioned experiment representation:** research question, competing hypotheses, variables and units, candidate conditions, experimental units and sampling plan, controls, assumptions and confounders, measurement procedure, data sources and collection method, hypothesis-specific predictions, statistical analysis plan, stopping criteria, safety classification and required permissions, protocol version and execution state, provenance, and final results and limitations. Explain representation, validation, storage, versioning and linking. Don't rely on unstructured LLM text.
- **7.2 Hypotheses:** precise statements, conditions of applicability, predicted observations, supporting and contradicting evidence, assumptions, confounders and inter-hypothesis relationships. Use explicit priors and posterior updates **only** when a real probabilistic model supports them. No decorative probabilities.
- **7.3 Adaptive design:** evaluate at least (1) random, (2) predetermined fixed, (3) stratified where appropriate, and (4) adaptive information-gain-based sampling. Check libraries and research implementations first, and evaluate Bayesian EIG as a candidate. The selector must account for competing hypotheses, current uncertainty, possible outcomes, EIG, cost, feasibility, safety and permissions, sampling constraints and stopping criteria. Define the maths, assumptions, compute needs and limitations. Explain when EIG is inappropriate or can't be estimated reliably, and give a justified fallback. Every recommendation must expose the proposed action, which hypotheses it targets, why it may be informative, its EIG or alternative score, cost and feasibility, the other candidates considered, and why it was chosen. Preserve the full sequence of recommendations.
- **7.4 Statistics:** descriptive stats, regression, condition comparisons, uncertainty intervals, Bayesian updating, diagnostics, confounder checks and data-quality analysis, using established libraries. Document the assumptions behind each procedure. Never choose a procedure because it is convenient or gives a nice result.
- **7.5 Provenance per observation:** observation ID, experiment ID, sampling-unit ID, timestamp and timezone, variable and unit, value, measurement condition, collection method, instrument or source ID, calibration info, quality flags, missingness status, origin (simulated / imported / manual / external), and transformation history. Keep raw data separate from processed data. Minimize private data.
- **7.6 Reproducibility package per finalized experiment:** versioned protocol, hypotheses, data and source references, data dictionary, provenance, model and software versions, configuration, seeds, analysis scripts or executable specs, results and uncertainty, figures and tables, execution logs, limitations and reproduction instructions. **A replay must never silently replace recorded results with new LLM interpretations.** Evaluate existing workflow, provenance and reproducibility tech first.

### 2.8 AI orchestration layer (brief §8)
- The LLM may turn NL questions into structured objectives, suggest hypotheses and confounders, draft protocols, explain options, suggest candidate observations, interpret **validated** analytical outputs, write evidence-grounded reports and identify gaps. It is **not** the authority for calculations.
- **Deterministic code owns** schema validation, maths, candidate ranking, data-quality checks, provenance and result computation.
- **Providers to investigate:** Anthropic API (when configured), local models via Ollama or similar, other APIs if materially better, orchestration frameworks, and a **deterministic development provider** for offline use. Verify current docs; don't assume model IDs, endpoints, structured-output formats or reasoning fields.
- **Claude Code access ≠ product API credentials.** The default demo must work **without any model API key**. The dev provider must be explicitly labelled and must not impersonate a frontier LLM.
- Validate model output before execution or persistence. **Never execute arbitrary generated code** just because a model returned it.
- Provenance must distinguish LLM, deterministic computation, human operator, external source and simulation. Evaluate existing orchestration products before building custom ones.

### 2.9 Product experience (brief §9)
- It should feel like a premium scientific instrument plus a sophisticated AI research product. Not a generic admin template with gradients, glowing borders or meaningless charts. Distinctive, polished, information-dense where appropriate, accessible.
- **Screens:**
  - Research Home: active experiments, unresolved questions, recent findings, observations, data-quality issues.
  - Experiment Workspace.
  - Hypothesis Explorer: competing hypotheses, expected observations, supporting and contradicting evidence.
  - Adaptive Experiment Planner: candidates, usefulness, feasibility, cost, uncertainty, ranking explanations.
  - Live Observation Stream.
  - Evidence and Analysis.
  - Experiment Replay.
  - Research Report.
  - Data and Integrations.
- For each screen, specify the underlying data and the interactions it needs. Evaluate UI, visualization and dashboard libraries.
- The default UI works without an API key. Simulated data is visibly marked. Plan loading, empty, error and recovery states. Polish does not equal completeness.

### 2.10 Required flagship demo (brief §10), all with real state and calculations
1. Open a seeded environmental question.
2. Inspect multiple hypotheses.
3. Generate or edit a structured protocol.
4. Validate the protocol.
5. Initialize the deterministic synthetic environment.
6. Collect initial observations.
7. Inspect data and uncertainty.
8. Rank candidate next measurements.
9. Select an action with the adaptive strategy.
10. Execute it in the simulator.
11. Update analysis from the new observation.
12. Repeat until the stopping condition.
13. Produce a report from computed results.
14. Export a reproducible package.
15. Rerun with the same seed.
16. Compare adaptive against the baseline strategies.

No prerecorded animations, hardcoded scores or fabricated logs. **Define acceptance criteria for every stage.**

### 2.11 Research benchmark (brief §11)
- Compare adaptive against valid non-adaptive baselines under equal budgets, across multiple synthetic environments with known ground truth.
- **Scenarios:** strong and weak relationships; multiple competing hypotheses; noise levels; missing observations; imbalanced sampling opportunities; confounders; misleading correlations; limited budgets; **and cases where adaptive should not win**.
- **Ground truth is hidden from the agent** and visible only to the evaluator. Documented seeds, repeated runs.
- **Metrics:** correct-hypothesis rate, calibration, log loss / Brier, parameter error, uncertainty reduction, cost, observations needed to reach criteria, inconclusive rate, sensitivity to noise and confounding, and runtime/compute. Evaluate correctness, calibration and efficiency **separately**. **Being confidently wrong must not look like success.**
- Numbers come from actual execution only. The report covers environments, assumptions, baselines, protocols, sample counts, results with intervals, failure cases and reproduction commands.
- Propose a realistic local benchmark size plus a route to larger runs. Don't presume adaptive wins. Investigate existing benchmark frameworks.

### 2.12 Technology (brief §12)
- **Candidate stack, not decided:** React+TS, Python+FastAPI, Pydantic, SQLite (with a Postgres path), scientific libraries, an established execution/workflow engine, a charting library, and model APIs where justified. Inspect alternatives, including reusing a scientific platform or extending a research implementation.
- **12.1 Subsystems:** frontend, API, scientific engine, hypothesis management, selection algorithms, simulated environment, ingestion and provenance, statistics, AI provider abstraction, job execution, persistence, benchmark runner, report generation. State the route for each and what custom code remains.
- **12.2 Background execution:** long jobs must not block the API. Job states: queued, running, completed, failed, cancelled. Needs progress, error records, controlled cancellation and recovery after interruption. **No Redis or distributed workers without evidence** they are needed.
- **12.3 Persistence:** experiments, hypotheses, protocol versions, observations, sources, execution events, results and benchmark runs, with real relations and constraints. Not one big JSON blob. Use migration libraries.
- **12.4 Typed, documented APIs:** experiment CRUD, protocol edit and validate, hypotheses, imports and observation ingestion, candidate-action generation, simulation execution, results and histories, jobs, reproducible exports, benchmark runs, AI-provider config. **The backend enforces scientific invariants regardless of the frontend.**
- **12.5 Developer experience:** explicit dependencies, env config, migrations, fixtures, tests, benchmark scripts, setup docs. **Windows 11 and Linux.** Runs locally with no mandatory cloud.

### 2.13 Security, safety, integrity (brief §13)
- **Security:**
  - Strict input validation; uploads and external responses are untrusted.
  - Time and resource limits; sandboxing for any supported generated code.
  - Restricted outbound network; API-key protection; explicit configuration for integrations.
  - Review of destructive or irreversible actions; consent for experiments involving people.
  - Careful handling of precise locations and sensitive records; a way to stop active experiments.
  - Simulations may run automatically within limits. People, danger, sensitive data or physical actuators require permissions, safeguards and human oversight.
  - Never run dangerous experiments, manipulate people without consent, access private systems or over-collect personal data.
- **Integrity:**
  - Preserve raw data; document transformations and exclusions.
  - Distinguish missing values from zero; flag insufficient sample sizes; track provenance.
  - Separate exploratory from confirmatory analysis; document post-collection protocol changes.
  - Distinguish correlation from causation; report contradictory and inconclusive findings; make results reproducible.
  - Never produce authoritative reports from unverified model-generated "observations".
- **The plan must include a threat model with mitigations.**

### 2.14 Future extensions to evaluate, not necessarily build (brief §14)
- A: advanced adaptive design.
- B: multi-source evidence integration with conflict resolution.
- C: autonomous replication within tolerances.
- D: a generalized second environment, e.g. a black-box API discovered through controlled queries.
- E: AI-assisted collaboration (critics and proposers). **Agreement between LLMs ≠ independent verification.**
- F: an experiment knowledge exchange (templates, protocols, benchmarks).
- G: a general scientific API.
- H: an experimental OS where new domains plug in through interfaces.
- I: alternative product directions that keep the core ambition, compared with PROBE.
- Prioritize by evidence and value, not feature count.

### 2.15 Resources and cost (brief §15)
- Assess what runs locally, what needs external services, what might need rented GPUs, recurring costs, the most valuable paid dependencies, free and self-hosted alternatives, and what gets expensive at scale.
- The local simulation, engine, benchmark and default demo work **without credentials**. Optional providers are configured separately.
- No purchases or cloud commitments without explicit authorization. If expensive resources are needed, justify them and propose a cheaper validation route first.

### 2.16 Required workflow stages (brief §16)
1. Read-only reconnaissance.
2. Research and competitive investigation.
3. ≥3 strategies plus the build/buy/integrate matrix.
4. Refined hypothesis, domain, demo and measurable success criteria; document uncertainties.
5. The comprehensive execution plan.
6. **STOP for approval.**

---

## 3. Final deliverable: required content of the plan (brief §17)

| Part | Content |
|---|---|
| A | **Executive summary:** what PROBE will be; initial capability; what makes it ambitious; the research contribution investigated; what the first deliverable does and won't do; startup potential; what could cause failure. Separate established, proposed and speculative. |
| B | **Research and competitive assessment:** related-work matrix, leading competitors, overlaps, limitations, strongest differentiation, source links, verified vs assumed. Recommend hypothesis changes if needed. |
| C | **Technology landscape:** best tools, libraries, research implementations, APIs, commercial products, models and infrastructure; what they provide; remaining gaps. |
| D | **Build/buy/integrate matrix** for every major subsystem: alternatives, recommended route, effects on quality, reliability, cost, complexity, research value and commercial viability; reuse directly / adapt / build from scratch. |
| E | **Three competing architecture strategies** compared on common criteria, with a recommendation (or an explanation if fewer than three are credible). |
| F | **Recommended architecture:** Mermaid diagram covering frontend, API, scientific engine, hypothesis manager, selection algorithms, simulator, ingestion and provenance, statistics, AI adapters, execution engine, DB, benchmark runner and report generator, with each component's purpose and ownership. |
| G | **Data model and API contracts:** entities, relations, key fields, validation rules, state transitions, and the boundaries between deterministic computation, AI-assisted reasoning and experiment execution. |
| H | **Scientific-method design:** the environment, hypotheses, variables, candidate actions, sampling methods, stopping criteria, data quality, statistics and uncertainty; **the maths of adaptive selection** (assumptions, equations, compute, limitations); prevention of data leakage and misleading conclusions. |
| I | **Benchmark and evaluation plan:** baselines, conditions, seeds, repetitions, metrics, budgets, success criteria, reproduction instructions, and which results support or falsify the hypotheses. |
| J | **Implementation roadmap:** ordered milestones, each with objective, deliverables, files and components affected, dependencies, required tests, acceptance criteria, main risks and definition of done. Ordered by dependency and risk reduction, not visual appeal. Identify the **earliest complete end-to-end vertical slice**. |
| K | **Repository and file-change plan:** based on the actual repo (existing vs new files), proposed tests and integration verification. |
| L | **UI and product design:** screens, navigation, interactions, design system, data dependencies, error states, responsiveness, and how the UI shows a *functioning* experiment. |
| M | **Security and reliability:** threat model, execution boundaries, data-integrity risks, scientific failure modes, safeguards, and license, vendor and privacy considerations. |
| N | **Resource and cost assessment.** |
| O | **Risk register, top 10:** each with probability, impact, early warning signs, mitigation and fallback. Covers technical, research, competitive and startup risk. |
| P | **Acceptance criteria:** scientific correctness, reproducibility, benchmark validity, app functionality, UX, security, reliability, cost, setup and installation. Runnable, testable, independently inspectable. |
| Q | **Approval decisions:** the small set of high-impact decisions that need user judgment, each with a recommended option and alternatives. No routine engineering questions. |

Also include the seven §2.3/3.8 items and the proposed contents of `RELATED_WORK.md`, `COMPETITIVE_LANDSCAPE.md`, `RESEARCH_HYPOTHESES.md` and `ARCHITECTURE_DECISIONS.md`, **inside the plan, not as repo files**.

**Operating principles (brief §19):**
1. Investigate before recommending.
2. Explore alternatives before committing.
3. Optimize for outcome, not custom code.
4. Prefer proven components where they are better.
5. Build custom only where it creates an advantage.
6. Make claims traceable to evidence.
7. Separate novelty, engineering value and business value.
8. State assumptions and uncertainty.
9. Decide where reasonable rather than asking unnecessary questions.
10. Preserve useful existing work.
11. A design doc is not an implemented capability.
12. Simulated is not real.
13. Don't prioritize polish over the scientific engine.
14. Don't assume the original direction is best.
15. Don't continue into implementation after presenting the plan.

If implementation is later approved, specialist subagents (prior-art, statistics, architecture, backend, frontend, testing, security) may be used with clear scopes and no conflicting concurrent edits. The CTO remains responsible for integration and verification.

**Definition of a successful plan (brief §20).** The user can determine:
- whether the problem is worth solving, and whether the direction is differentiated;
- what to reuse and what to build;
- the architecture, how the scientific method works, and how the adaptive algorithm is tested;
- how the demo works and how the benchmark validates or falsifies the hypothesis;
- the resources needed, the failure risks and the startup opportunity;
- what "done" means and what they approve by authorizing implementation.

The plan must be concrete enough to start without rediscovering basic decisions, while staying flexible where research hasn't settled them.

---

## 4. Status by stage

| Stage | Status | Notes |
|---|---|---|
| 1. Read-only reconnaissance | ✅ Done | See §5. Repo is effectively empty. |
| 2. Research and competitive investigation | 🔄 In progress | Workflow `probe-landscape-research` (run `wf_181ed593-9f9`) launched in the originating session. Results are not yet in §7. If they are missing when you resume, **re-run §9**. |
| 3. ≥3 strategies + build/buy/integrate matrix | ⏳ Not started | Planned as a second workflow (§8 step 3). |
| 4. Refine hypothesis, domain, demo, success criteria | ⏳ Not started | |
| 5. Comprehensive plan (Parts A–Q) | ⏳ Not started | |
| 6. Stop for approval | ⏳ | Present in Plan Mode, then `ExitPlanMode`. |

---

## 5. Verified environment and repository facts (2026-10-09)

- **Repo:** `parthvarekar/probe`. Branches: `main` and `claude/pensive-ptolemy-huq2mx`, both at commit `4fd56f5` ("Initial commit", Parth Varekar) before this handover. The only file is `README.md`, containing `# Probe` with no trailing newline. **No existing code, conventions or dependencies to preserve.** The architecture is unconstrained.
- **Cloud container used for planning:** Python 3.13.16, Node v22.22.0, npm 10.9.4, pnpm, uv 0.11.32, Docker CLI present, git. 4 vCPU, 15 GiB RAM, **no GPU**. Outbound HTTPS goes through an agent proxy. These are facts about the *planning* environment, not the user's machine. The target is a local Windows 11 and Linux developer machine.
- **GitHub tooling scope:** `mcp__github__*` tools are scoped to `parthvarekar/probe` only. Read other public repos through ordinary web fetches of github.com pages, not the GitHub MCP tools.

---

## 6. Verified facts gathered so far

### 6.1 Anthropic API (source: the bundled `claude-api` reference skill, model table cached 2026-10-06; re-verify before final cost numbers)

| Model | ID | Context | $ / 1M input | $ / 1M output |
|---|---|---|---|---|
| Claude Fable 5.1 | `claude-fable-5-1` | 1M | 10.00 | 50.00 |
| Claude Opus 5.5 (current Opus) | `claude-opus-5-5` | 1M | 4.00 | 20.00 |
| Claude Sonnet 5.5 (current Sonnet) | `claude-sonnet-5-5` | 1M | 2.00 | 10.00 |
| Claude Haiku 5.5 (current Haiku) | `claude-haiku-5-5` | 1M | 0.10 (≤100K-token prompts; 0.50 beyond) | 0.50 (2.50 beyond) |

- **Structured outputs:** `output_config: {format: {...}}` (JSON schema), with `client.messages.parse()` for validated parsing. **Strict tool use:** `strict: true` on the tool definition, which requires `additionalProperties: false`.
- **Forced `tool_choice` (`any`/`tool`) returns HTTP 400 on Opus 5.5, Sonnet 5.5 and Fable 5.1.** Use `auto` + `strict: true`, or structured outputs.
- Thinking cannot be disabled on Opus 5.5; control it with `output_config.effort` (default `medium`). No assistant prefill on current models. Handle `stop_reason: "refusal"`.
- **Message Batches API:** 50% cost, asynchronous. **Prompt caching:** prefix-based, up to 4 breakpoints; cache reads are cheap (e.g. $0.20/MTok on Opus 5.5 and Sonnet 5.5).
- **Implication for PROBE:** a high-quality optional provider exists, with schema-enforced outputs suitable for validated hypothesis and protocol drafting. **No API key is assumed for the product**, so the default demo must use a labelled deterministic dev provider.

---

## 7. Research results

> **Not yet recorded.** When the research workflow completes, add a compact, verified digest per dimension here: key findings, a verified source table (title, org, date, URL, contribution, evidence, limitations, relevance, verification status), tools with license and maintenance, and competitors. Mark anything unverified.

### 7.1 Unverified working hypotheses (pre-research; **do not cite as findings**)
These are hypotheses from the CTO's prior knowledge, recorded so the next session knows what the research is testing. **All must be confirmed or refuted against primary sources.**
- **Likely high-overlap prior art** to check carefully:
  - BoxingGym (Stanford, 2025): a benchmark for LLM agents doing experimental design and model discovery with EIG.
  - POPPER (Stanford, 2025): agentic sequential falsification with e-values.
  - BED-LLM (Rainforth group, 2025): LLMs combined with Bayesian experimental design.
  - Piriyakulkij et al. (NeurIPS 2024): LLM-proposed rules, Bayesian inference and active experimentation.
  - AI2 AutoDiscovery (Bayesian surprise).
  - FutureHouse/Edison Kosmos and Robin, Google AI co-scientist, Sakana AI Scientist v1/v2.
  - Sloman et al. 2022: adaptive designs can be *worse* than random under misspecification ("active learning bias").
- **Likely conclusion to test:** "adaptive EIG beats random/fixed" is *well known* in correctly specified settings, so Hypothesis A as stated is probably not novel. A **sharper, falsifiable candidate**: *misspecification-robust adaptive hypothesis discrimination*. The hypothesis set is generated (possibly by an LLM) and may be **incomplete or misspecified**. The question is whether a guarded adaptive design keeps its efficiency advantage without becoming confidently wrong. Candidate guards: an explicit "unmodeled / none-of-the-above" hypothesis, tempered or power posteriors, posterior-predictive checks that trigger hypothesis-space expansion, an exploration mixture, and anytime-valid (e-value) inference under optional stopping. The headline metric would be calibrated correctness per unit cost, with confident-wrong runs penalized.
- **Likely architecture direction (to be stress-tested by ≥3 alternatives):**
  - A Python deterministic scientific core: hypotheses as executable generative models, exact or Monte-Carlo EIG over finite candidate actions, cost-aware selection.
  - FastAPI with Pydantic v2; SQLite via SQLAlchemy and Alembic (Postgres-ready); a DB-backed job queue (no Redis).
  - Content-addressed raw data with a provenance graph, exported as RO-Crate/W3C PROV.
  - A React+TS SPA.
  - An optional LLM behind a provider abstraction, with a labelled deterministic offline dev provider.
- **Candidate second environment** for generalization (Hypothesis B): black-box API/system behavior discovery through controlled queries.

---

## 8. Next steps (ordered)

1. **Collect the research results.** In the originating session: read the workflow result or its `journal.jsonl`. In a new session: re-run the §9 workflow (needs Workflow-tool opt-in; ultracode was on in the originating session). Without workflows, run the 11 dimensions as parallel `Agent` research tasks with the same prompts.
2. **Record digests in §7** of this file. Commit and push (`handover.md` only).
3. **Run the design workflow** (read-only):
   - (a) three materially different strategy proposals (e.g. a custom lean BOED core; a platform built on an existing BO/DoE framework such as Ax/BoTorch or BayBE; an LLM-agent-centric approach on an existing AI-scientist/benchmark stack), plus a judge panel;
   - (b) a mathematical spec of the adaptive selector (EIG formulation, estimator, cost, stopping, misspecification diagnostics, fallback);
   - (c) benchmark design (environments, baselines, seeds, metrics, budgets, falsification criteria);
   - (d) a red-team critique of novelty, commercial claims and domain choice;
   - (e) a threat model.
4. **Write the final plan** covering Parts A–Q (§3), the seven §3.8 deliverables and the four proposed research-artifact contents.
5. **Present it:** `EnterPlanMode` → write the plan to the plan file → `ExitPlanMode`. **STOP. Wait for explicit approval.** The user may approve, request revisions, approve only some milestones, or reject.

---

## 9. Research workflow (re-runnable)

Run it with the **Workflow** tool by passing the script below as `script`. Its agents are instructed to be strictly read-only. Pattern: `pipeline(DIMENSIONS, research → adversarial verify+digest)`, then a completeness critic (barrier), then ≤5 gap-fill agents.

<details>
<summary>Full workflow script (as launched 2026-10-09)</summary>

```js
export const meta = {
  name: 'probe-landscape-research',
  description: 'Read-only research sweep for PROBE: 11 research dimensions, adversarial source verification, completeness critique and gap-filling',
  phases: [
    { title: 'Research', detail: 'one deep-research agent per landscape dimension' },
    { title: 'Verify', detail: 'adversarial fact-check + compact corrected digest per dimension' },
    { title: 'Critique', detail: 'completeness critic across all digests' },
    { title: 'Gap-fill', detail: 'research the most important gaps' },
  ],
}

const CONTEXT = `
You are a research analyst supporting the founding CTO of PROBE, an early-stage AI product. TODAY'S DATE IS 2026-10-09. Your training knowledge may be stale: actively search for 2025-2026 developments and verify current facts (releases, licenses, pricing, company status) from primary sources.

PROBE concept (under evaluation, not settled): an AI system that turns a question about the world into a structured experiment with competing hypotheses, obtains real or simulated evidence, chooses the next most informative observation (adaptive experimental design, e.g., Bayesian expected information gain), distinguishes what it suspects / predicts / observed / what evidence supports, and produces reproducible, independently verifiable conclusions (versioned protocols, provenance, uncertainty, replayable analysis). Deterministic algorithms (not the LLM) own calculations, ranking, validation and provenance; an LLM layer is optional and assists with structuring questions, hypotheses, confounders and reports. Candidate first domain: campus pedestrian-level microclimate (shade, surface type, location, time of day vs temperature/thermal conditions) with three data modes: deterministic synthetic simulator with hidden ground truth, CSV import, and manual/HTTP sensor observations. A reproducible benchmark must compare adaptive selection vs random, fixed and stratified baselines on correctness, calibration and efficiency (including cases where adaptive should NOT win). Must run locally on Windows 11 and Linux, with no API key or hardware required for the default demo. Candidate stack (not decided): React+TypeScript frontend, Python+FastAPI backend, Pydantic, SQLite.
Research hypotheses under evaluation: (A) adaptive, evidence-driven experiment selection yields more reliable conclusions per unit cost than fixed/random sampling while preserving reproducibility and calibrated uncertainty; (B) a system can learn/choose useful experiments in an unfamiliar environment by reasoning about competing explanations and possible observations.
The CTO must determine what is genuinely differentiated given existing work, so honesty about overlap is more valuable than enthusiasm.

RULES (strict):
- READ-ONLY. Do not create, edit or delete any files anywhere. Do not install packages. Do not run git commands that modify anything. Do not use any mcp__github__ tools (repository scope restriction) — read public GitHub pages via WebFetch instead. Bash may be used ONLY for read-only network metadata lookups such as: curl -s https://pypi.org/pypi/<pkg>/json | python3 -c '...' (print only, no files).
- Use WebSearch (mode "standard" by default; "extended" for niche, very recent, pricing, or hard-to-find facts) and WebFetch on primary sources: original papers (arXiv/journal pages), official docs, source repositories (README, LICENSE, releases page), official product pages, pricing pages.
- NEVER fabricate citations, authors, dates, numbers, licenses, prices or capabilities. If you could not access a source, set accessed=false and say so. If unsure, say unsure. Distinguish peer-reviewed papers, preprints, product claims, blog posts and proposals.
- Distinguish capabilities that are DEMONSTRATED with evidence from capabilities that are CLAIMED or PROPOSED.
- Be concrete and specific. Prefer depth on the most important items over shallow coverage of everything; aim for roughly 15-35 well-investigated items.
`

const DIMENSIONS = [
  { key: 'ai-scientist-systems', focus: `End-to-end AI-scientist and agentic research systems. Investigate at minimum: Sakana "The AI Scientist" v1 (2024) and v2 (2025) plus independent evaluations/critiques of it; Google "AI co-scientist" (2025); FutureHouse / Edison Scientific systems (Robin, Kosmos, PaperQA2, Aviary, data-analysis agents) and their 2025-2026 evolution; AI2 CodeScientist and related AI2 work; Agent Laboratory; Virtual Lab (Swanson et al.); Biomni (Stanford); Coscientist (Boiko et al., Nature 2023); ChemCrow; Google DeepMind AlphaEvolve and any 2025-2026 DeepMind science agents; Microsoft Discovery; Intology Zochi; Autoscience Carl; DeepScientist; Agents4Science 2025 conference outcomes; OpenAI science initiatives; AI-generated experimental protocols (BioPlanner, LAB-Bench ProtocolQA, protocol representations such as Autoprotocol); any notable 2026 systems (search "AI scientist 2026", "autonomous research agent 2026", "AI that runs experiments 2026"). For EACH system determine whether it (a) generates hypotheses, (b) designs experiments, (c) adaptively chooses the NEXT experiment based on explicit uncertainty/information, (d) executes experiments (wet lab, computational, simulated), (e) quantifies uncertainty with explicit probabilistic models, (f) controls error rates under adaptive stopping/multiple testing, (g) preserves provenance and reproducibility packages, (h) benchmarks against non-adaptive baselines. Separate demonstrated vs claimed. Summarize the common gaps across this category.` },
  { key: 'hypothesis-validation-benchmarks', focus: `Systems and benchmarks specifically about hypothesis testing/falsification, data-driven discovery, and LLM-driven experimental design / information gathering. At minimum: POPPER (Huang et al. 2025, "Automated Hypothesis Validation with Agentic Sequential Falsifications", e-values); AI2 AutoDiscovery / AutoDS (Bayesian-surprise-driven open-ended discovery, 2025); DiscoveryBench; DiscoveryWorld; BoxingGym (Gandhi et al. 2025, benchmarking automated experimental design and model discovery, uses EIG); ScienceAgentBench; BLADE; MLGym / MLAgentBench; "Automated Statistical Model Discovery with Language Models" (Li et al., ICML 2024); "Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning" (Piriyakulkij et al., NeurIPS 2024); BED-LLM (Choudhury, Rainforth et al. 2025); Uncertainty of Thoughts (Hu et al. 2024); LLAMBO; HypoGeniC and hypothesis-generation literature; Si et al. "Can LLMs generate novel research ideas"; LLM agents doing active learning / information-seeking / 20-questions with EIG; any 2025-2026 benchmarks of agents choosing experiments or interactive scientific discovery (search thoroughly). For BoxingGym specifically: exact environments, license, code availability and activity, how EIG and prediction error are used, headline results (do LLM agents beat random designs?), and whether PROBE could adopt it as an external benchmark. For POPPER: exact method, error-control guarantees, evaluation, code/license. State precisely how much of PROBE's Hypotheses A and B is already answered by existing work and what remains open.` },
  { key: 'boed-theory', focus: `Bayesian optimal experimental design (BOED), sequential/adaptive design theory, and anytime-valid inference. At minimum: Rainforth, Foster, Ivanova, Bickford Smith "Modern Bayesian Experimental Design" (Statistical Science 2024); Chaloner & Verdinelli 1995; Ryan et al. 2016 review; Foster et al. 2019 "Variational Bayesian optimal experimental design"; Foster et al. 2020 unified SGD approach; Deep Adaptive Design (Foster et al., ICML 2021), iDAD (Ivanova et al. 2021), RL-BOED (Blau et al. 2022); Sloman et al. 2022 "Characterizing the robustness of Bayesian adaptive experimental designs to active learning bias"; BOED under model misspecification (Overstall & McGree; Go & Isaac robust EIG; 2023-2026 work on misspecification-robust BED, generalized/power posteriors in BED); model-discrimination designs (Box & Hill 1967; Atkinson & Fedorov T-optimality; Cavagnaro, Myung & Pitt Adaptive Design Optimization 2010); BALD (Houlsby 2011); EIG estimators (nested Monte Carlo, PCE/ACE bounds, Laplace, importance sampling, exact computation for discrete hypotheses) with their bias/variance/compute; cost-aware/budgeted and non-myopic design; stopping rules; anytime-valid inference / e-values / safe testing (Grunwald, de Heide & Koolen "Safe Testing"; Ramdas et al. "Game-theoretic statistics and safe anytime-valid inference"; Howard et al. confidence sequences) and their interaction with adaptive data collection; Bayes-factor optional stopping; documented failure modes where adaptive designs underperform random or fixed designs; active intervention design for causal discovery. Deliver a concrete mathematical recommendation for PROBE: (i) EIG formulation when competing hypotheses are discrete generative models each with continuous nuisance parameters (EIG on the model indicator vs on parameters vs joint), with equations; (ii) cost-penalized selection; (iii) estimator choice with compute cost for ~10-200 candidate actions per step on a laptop; (iv) diagnostics for when EIG is unreliable (misspecification, poor posterior approximations); (v) principled fallback strategy; (vi) how to keep inference valid under adaptive selection and optional stopping; (vii) what kinds of benchmark environments should make adaptive design LOSE.` },
  { key: 'boed-software', focus: `Software libraries for adaptive experimental design, active learning, Bayesian optimization, probabilistic programming and simulation-based inference. Evaluate at minimum: Pyro (pyro.contrib.oed), NumPyro, PyMC (+ pymc-extras), Stan/CmdStanPy, BlackJAX, Meta Ax and BoTorch (incl. information-theoretic acquisition functions), Emukit, Trieste, BayBE (Merck KGaA), NIST optbayesexpt, ADOpy, pyDOE3, sbi (mackelab), BayesFlow, ELFI, pyabc, dynesty and UltraNest (Bayesian evidence for model comparison), Optuna, SMAC3, HEBO, scikit-learn GaussianProcessRegressor, GPyTorch, OSS Vizier, Olympus/Atlas (Aspuru-Guzik group), Summit, and any 2024-2026 BOED-specific libraries (search e.g. "bayesian experimental design python library 2025", "expected information gain python package"). For each: license, latest release date and recent activity (check https://pypi.org/pypi/<name>/json via curl and the GitHub page), Python version support (3.12/3.13), Windows support, dependency weight (PyTorch/JAX), whether it supports discrete model discrimination with EIG, sequential design, cost-aware selection, and whether it is suitable as PROBE's engine core, a component, or only a reference implementation. Inspect actual source where it matters (e.g., pyro.contrib.oed estimators). Recommend a concrete stack for PROBE's EIG/inference engine with justification and identify exactly what custom code remains.` },
  { key: 'sdl-and-commercial', focus: `Self-driving laboratories, experiment-orchestration platforms, and commercial competitors / adjacent products. At minimum: Lila Sciences, Periodic Labs, Edison Scientific / FutureHouse commercial offerings, Emerald Cloud Lab, Strateos (status), Atinary SDLabs, Citrine Informatics, ChemOS 2.0, Bluesky (NSLS-II), Acceleration Consortium, A-Lab (Berkeley), IvoryOS, Chemspeed, Synthace, Benchling AI, Owkin, causaLens (causal AI), literature-AI tools (Elicit, Consensus, Scite, Undermind, SciSpace) as adjacent but different, product-experimentation platforms (Statsig — check acquisition status; Eppo — check Datadog acquisition; GrowthBook; Optimizely; LaunchDarkly) as adjacent adaptive-experimentation markets, Meta Ax and Google Vizier as internal platforms, Sakana AI, Intology, Autoscience, and any 2025-2026 startups pitching autonomous experimentation, AI scientists, experiment agents, hypothesis engines, or "AI for empirical research" (search YC 2025-2026 batches and major VC announcements). For each: what they actually sell, evidence of capability, pricing model if public, funding/scale (with source), target buyer, overlap with PROBE, how they could expand into PROBE's space. Conclude where the market is crowded and where real white space exists, and which business models look viable for PROBE.` },
  { key: 'workflow-provenance-infra', focus: `Execution, workflow, persistence, provenance, reproducibility and sandboxing infrastructure for a locally runnable Python + web application on Windows 11 and Linux without mandatory cloud. Evaluate: durable execution / background jobs — DBOS Transact Python (check SQLite support and license), Temporal, Prefect 3, Dagster, Hatchet, Procrastinate, Huey, Dramatiq, RQ, arq, Celery, and a plain DB-backed job table with asyncio/thread workers; scientific workflow systems — Snakemake, Nextflow, CWL, Airflow (fit or not); provenance — W3C PROV-DM/PROV-O and the "prov" Python package, RO-Crate and Workflow Run RO-Crate profiles (ro-crate-py), ProvONE, noWorkflow; experiment tracking — MLflow, DVC, Sacred, Aim, W&B; reproducible environments — uv lockfiles, pixi, conda-lock, Docker, Nix (Windows implications); persistence — SQLite (WAL, constraints, JSON), DuckDB, PostgreSQL, SQLAlchemy 2 + Alembic, SQLModel; content-addressed storage of raw files and hashing; data validation — Pydantic v2, Pandera, Great Expectations, Frictionless (Table Schema / Data Package), pint and pint-pandas (units), Polars/pandas; report generation — Quarto, Jinja2 + WeasyPrint, marimo/Jupyter export; sandboxing for any generated analysis code — E2B, Modal, Daytona, gVisor, Firecracker, Docker with limits, nsjail/bubblewrap (Linux-only), Pyodide/WASM (cross-platform), RestrictedPython (and its weaknesses) — and whether PROBE v1 should avoid executing generated code entirely. For each: license, maintenance, Windows support, operational complexity, fit. Recommend a concrete minimal-but-strong stack.` },
  { key: 'llm-providers-orchestration', focus: `LLM providers and AI orchestration frameworks as of October 2026. Verify CURRENT official documentation (do not assume model names or features): Anthropic API (current model lineup, structured outputs / JSON schema enforcement, tool use, pricing, rate limits, prompt caching, batch API); OpenAI API (structured outputs, Responses API, pricing); Google Gemini API (structured output, pricing, free-tier data-use terms); Ollama (structured outputs via JSON schema "format", supported local models, Windows support); llama.cpp (GBNF grammars / JSON schema); vLLM (guided decoding); LM Studio; abstraction layers — LiteLLM, PydanticAI, Instructor, Outlines, BAML, LangChain/LangGraph, DSPy, OpenAI Agents SDK, Claude Agent SDK, Model Context Protocol (MCP); local open-weight models current in 2026 (e.g., Qwen3-family, Llama 4, gpt-oss, Gemma, Mistral) for structured extraction/reasoning and their hardware needs (CPU-only 16 GB RAM laptop vs consumer GPU). Also: deterministic/offline development-provider patterns (record/replay e.g. vcrpy, fixture providers); LLM provenance/observability (OpenTelemetry GenAI semantic conventions, Langfuse, Arize Phoenix — licenses); prompt-injection risks from ingested data and mitigations. Recommend a provider abstraction design, default offline behavior, which frameworks to use vs avoid (with reasons), and rough per-session cost estimates for a typical PROBE experiment session (state token assumptions explicitly).` },
  { key: 'microclimate-domain', focus: `The proposed initial domain: campus/urban pedestrian-level microclimate (shade, surface materials, location, time of day vs air temperature, surface temperature, mean radiant temperature, thermal comfort). Investigate: established physics and empirical findings with effect sizes (urban heat island; surface vs air temperature differences; tree shade vs built shade; albedo/material effects; sky view factor; wind); thermal-comfort indices (UTCI, PET, MRT, WBGT) and the pythermalcomfort library (license, maintenance); key confounders and measurement artifacts (radiation-shield errors in low-cost sensors, sensor time constants, calibration drift, time-of-day and weather confounding, spatial autocorrelation); simulators/models — ENVI-met (license/cost), SOLWEIG/UMEP (open source, license), PALM-4U, Ladybug Tools, SUEWS/SuPy, CityFFD, simple surface energy-balance models — and whether any could serve as PROBE's synthetic ground-truth environment vs a custom structural causal simulator calibrated to literature; public data — NOAA ISD/GHCNh/ASOS, Open-Meteo (terms), Meteostat, Netatmo (API terms), Weather Underground PWS, PurpleAir, Sensor.Community, openSenseMap, SAGE/Waggle, Array of Things archives, NOAA/CAPA Heat Watch campaign data, Landsat/ECOSTRESS LST, open campus microclimate datasets; QC tools — CrowdQC+, TITAN/titanlib; low-cost sensors (BME280/SHT4x with radiation shields, IR thermometers, Kestrel, HOBO) and costs. Assess market potential: who pays for microclimate/heat evidence (cities, campuses, insurers, landscape architects, public health), existing companies (CAPA Strategies, Climate Central, Jupiter Intelligence, others), and whether microclimate is a strong initial domain for demonstrating adaptive experimentation (e.g., can a user actually choose where/when to measure? are interventions possible?).` },
  { key: 'alternative-domains', focus: `Alternative initial domains and product directions that preserve the core ambition (automated empirical discovery via adaptive, hypothesis-discriminating experiments). Evaluate at least: (a) black-box software/API behavior discovery — active automata learning (LearnLib, AALpy), property-based testing (Hypothesis), fuzzing, specification mining, LLM agents probing unfamiliar APIs; (b) hypothesis-driven investigation of production software — AI SRE startups (Traversal, Resolve.ai, Cleric, others) and chaos-engineering platforms (Gremlin, Steadybit, LitmusChaos) where experiments are interventions; (c) experiments on AI models / LLM behavior — Transluce Docent, Anthropic Petri, UK AISI Inspect, active testing (Kossen et al.), IRT-based adaptive evaluation (tinyBenchmarks etc.), automated interpretability agents; (d) product/growth A/B testing with bandits; (e) cognitive science adaptive design optimization (ADOpy) and psychophysics (QUEST, Psi method); (f) agriculture / field trials; (g) materials/chemistry via simulators (Olympus) or cloud labs; (h) performance engineering / configuration tuning; (i) synthetic causal environments (BoxingGym, causal discovery with interventions). For each assess: cost and latency per experiment, availability of REAL (not simulated) evidence without hardware, ground truth availability for benchmarking, naturally competing hypotheses, realism of confounding, buyer and willingness to pay, competition intensity, demo intuitiveness. Rank them and recommend whether the flagship should remain microclimate, switch, or pair microclimate with a second environment (and which), with evidence.` },
  { key: 'statistics-causal-integrity', focus: `Statistical analysis, causal inference, and scientific-integrity tooling and practice. Evaluate: statsmodels, SciPy stats, pingouin, linearmodels, PyMC + Bambi, ArviZ (LOO/WAIC, posterior predictive checks), CmdStanPy, NumPyro; causal — DoWhy/PyWhy (identification + refutation API), EconML, causal-learn, pgmpy, y0, Ananke, dagitty (R and JS), sensemakr (Python), E-value sensitivity analysis (VanderWeele & Ding); active intervention design for causal discovery; anytime-valid tooling — confseq (Howard/Ramdas), safestats (R), any Python e-value libraries; multiplicity control (statsmodels multipletests, e-BH); multiverse/specification-curve tools (Boba, specification_curve, multiverse R); preregistration (OSF, AsPredicted), reporting standards (STROBE etc.), FAIR principles; spatial statistics for sensor data (PySAL/esda for spatial autocorrelation, PyKrige, GSTools); measurement-error models; calibration metrics and proper scoring rules (Brier, log loss, reliability diagrams, ECE; scoringrules/properscoring packages); approaches for programmatically constraining claim strength (descriptive vs predictive vs causal) based on study design. For each tool: license, maintenance, Python 3.12/3.13 and Windows support, fit. Recommend a concrete analysis stack and a "claim-guard" policy mapping evidence design (observational, stratified observational, controlled intervention, randomized) to the strongest permitted claim language.` },
  { key: 'frontend-viz', focus: `Frontend, scientific visualization and UI stack for a premium, information-dense, scientific-instrument-style web app that runs locally. Evaluate: React + TypeScript + Vite SPA vs Next.js vs SvelteKit vs SolidStart; accessible component primitives (Radix UI, shadcn/ui, React Aria, Mantine, Ark UI, Base UI) — check current maintenance status of each in 2026; charting — Observable Plot, Vega-Lite/react-vega, Plotly.js, Apache ECharts, visx, D3, uPlot, Nivo, Recharts (scientific capability: uncertainty bands, small multiples, posterior distributions, performance, license, bundle size); maps — MapLibre GL JS, deck.gl, Leaflet, OpenStreetMap tile usage policy, Protomaps/PMTiles for offline basemaps; graph/hypothesis visualization — React Flow (xyflow), Cytoscape.js, ELK.js, Sigma.js; data grids — TanStack Table, AG Grid Community, Glide Data Grid; data fetching and typed clients — TanStack Query, openapi-typescript, Orval, Hey API; real-time — SSE vs WebSockets with FastAPI; testing — Playwright, Vitest, Testing Library; Python-native UI alternatives (Streamlit, Panel/HoloViz, Dash, NiceGUI, Reflex, Solara, marimo apps) and whether they can deliver a premium, distinctive UI; backend frameworks FastAPI vs Litestar vs Django Ninja (typed OpenAPI, SSE, maturity in 2026). Draw design inspiration from serious analytical/scientific products (Observable, Linear, Benchling, Weights & Biases, Bloomberg Terminal, Grafana) and articulate design principles for a credible scientific instrument UI. Recommend a concrete stack with license notes.` },
]

const ITEM = { type: 'object' }
const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    dimension: { type: 'string' },
    summary: { type: 'string', description: 'Key takeaways for PROBE, 400-900 words' },
    sources: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, title: { type: 'string' }, authors_or_org: { type: 'string' }, date: { type: 'string' }, url: { type: 'string' },
      kind: { type: 'string', description: 'peer_reviewed | preprint | official_docs | source_repo | product_page | blog | news | benchmark | dataset | other' },
      status: { type: 'string' }, accessed: { type: 'boolean' },
      what_it_does: { type: 'string' }, central_contribution: { type: 'string' }, evidence: { type: 'string' },
      baselines_or_benchmarks: { type: 'string' }, limitations: { type: 'string' }, does_not_address: { type: 'string' },
      relevance_to_probe: { type: 'string' }, overlap_with_probe: { type: 'string', description: 'none | low | medium | high | critical' },
      demonstrated_vs_claimed: { type: 'string' },
    }, required: ['title', 'url', 'accessed'] } },
    tools: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, url: { type: 'string' }, category: { type: 'string' }, license: { type: 'string' },
      latest_release: { type: 'string' }, maintenance: { type: 'string' }, maturity: { type: 'string' },
      capabilities: { type: 'string' }, limitations: { type: 'string' }, commercial_use_ok: { type: 'string' },
      windows_support: { type: 'string' }, fit_for_probe: { type: 'string' },
      recommended_route: { type: 'string', description: 'use_as_is | integrate_behind_abstraction | adapt_extend | combine | replace_part | build_new | reference_only | reject' },
      accessed: { type: 'boolean' },
    }, required: ['name', 'url'] } },
    competitors: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, org: { type: 'string' }, url: { type: 'string' }, type: { type: 'string' },
      funding_or_scale: { type: 'string' }, what_they_do: { type: 'string' }, overlap: { type: 'string' },
      threat_level: { type: 'string' }, expansion_path: { type: 'string' }, accessed: { type: 'boolean' },
    }, required: ['name', 'url'] } },
    key_insights_for_probe: { type: 'array', items: { type: 'string' } },
    recommendations: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
    unverified_claims: { type: 'array', items: { type: 'string' } },
  },
  required: ['dimension', 'summary', 'sources', 'key_insights_for_probe', 'recommendations'],
}

const DIGEST_SCHEMA = {
  type: 'object',
  properties: {
    dimension: { type: 'string' },
    digest_markdown: { type: 'string', description: 'Corrected digest, max ~1800 words: verified key findings, implications for PROBE, concrete recommendations, explicit uncertainty' },
    sources: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, authors_or_org: { type: 'string' }, date: { type: 'string' }, url: { type: 'string' },
      kind: { type: 'string' }, status: { type: 'string' },
      contribution: { type: 'string', description: '<=40 words' }, evidence: { type: 'string', description: '<=40 words' },
      limitations: { type: 'string', description: '<=40 words' }, relevance: { type: 'string', description: '<=40 words' },
      overlap: { type: 'string' },
      verification: { type: 'string', description: 'verified | corrected | unverified | removed' }, correction_note: { type: 'string' },
    }, required: ['title', 'url', 'verification'] } },
    tools: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, url: { type: 'string' }, license: { type: 'string' }, latest_release: { type: 'string' },
      maintenance: { type: 'string' }, windows: { type: 'string' }, route: { type: 'string' }, fit: { type: 'string', description: '<=40 words' },
      verification: { type: 'string' },
    }, required: ['name', 'url', 'verification'] } },
    competitors: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, url: { type: 'string' }, what: { type: 'string', description: '<=40 words' }, funding_scale: { type: 'string' },
      overlap: { type: 'string' }, threat: { type: 'string' }, expansion_path: { type: 'string' }, verification: { type: 'string' },
    }, required: ['name', 'url', 'verification'] } },
    recommendations: { type: 'array', items: { type: 'string' } },
    corrections_made: { type: 'array', items: { type: 'string' } },
    not_verified: { type: 'array', items: { type: 'string' } },
    missing_items: { type: 'array', items: { type: 'string' } },
  },
  required: ['dimension', 'digest_markdown', 'sources', 'recommendations', 'corrections_made', 'not_verified'],
}

const GAPS_SCHEMA = {
  type: 'object',
  properties: {
    gaps: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, why_it_matters: { type: 'string' }, research_instructions: { type: 'string' },
    }, required: ['title', 'why_it_matters', 'research_instructions'] } },
    contradictions: { type: 'array', items: { type: 'string' } },
    overall_assessment: { type: 'string' },
  },
  required: ['gaps', 'contradictions', 'overall_assessment'],
}

const results = await pipeline(
  DIMENSIONS,
  d => agent(
    `${CONTEXT}\n\nYOUR RESEARCH DIMENSION: ${d.key}\n\n${d.focus}\n\nReturn your findings via the structured output. In 'summary', state the most important conclusions for PROBE, including uncomfortable ones (e.g., if existing work already covers PROBE's idea).`,
    { label: `research:${d.key}`, phase: 'Research', schema: FINDINGS_SCHEMA }
  ),
  (findings, d) => findings ? agent(
    `${CONTEXT}\n\nYou are an ADVERSARIAL FACT-CHECKER and editor for research dimension "${d.key}". Another agent produced the findings JSON below. Assume it may contain hallucinated papers, wrong authors/dates, wrong licenses, outdated maintenance or company status, inflated capability claims, or wrong benchmark numbers.\n\nTasks:\n1. Verify every load-bearing item: all sources with overlap high/critical, every tool whose route is not reject/reference_only, every competitor, and every surprising quantitative claim. Actually fetch the URL (WebFetch) or search (WebSearch). Check existence, title, authors/org, date, license, latest release/maintenance, company status/funding, and whether the claimed contribution/evidence is accurate. Verify at least 20 items (or all, if fewer). List anything you did not verify in not_verified — do not silently skip.\n2. Correct errors; mark fabricated or unresolvable items as removed. Never introduce new unverified claims.\n3. Note important missing items you encounter (missing_items) and add any you verify yourself.\n4. Produce the compact corrected digest per the schema (digest_markdown max ~1800 words; per-item text fields <=40 words).\n\nFINDINGS JSON:\n${JSON.stringify(findings)}`,
    { label: `verify:${d.key}`, phase: 'Verify', schema: DIGEST_SCHEMA }
  ) : null,
)

const digests = results.filter(Boolean)
log(`${digests.length}/${DIMENSIONS.length} dimensions produced verified digests`)

phase('Critique')
const critique = await agent(
  `${CONTEXT}\n\nYou are a COMPLETENESS CRITIC. Below are verified research digests across ${digests.length} dimensions for PROBE. Identify the most important gaps that would change the CTO's architecture, differentiation, or domain decision if filled: missing prior work (especially anything that already does adaptive hypothesis-discriminating experimentation with provenance), missing competitors, missing critical tools, unverified load-bearing claims, and contradictions between dimensions. Use WebSearch to sanity-check that a gap is real before listing it. Return at most 6 gaps, ordered by importance, each with precise research instructions.\n\nDIGESTS:\n${JSON.stringify(digests.map(x => ({ dimension: x.dimension, digest: x.digest_markdown, source_titles: (x.sources || []).map(s => s.title), tools: (x.tools || []).map(t => t.name), competitors: (x.competitors || []).map(c => c.name), missing_items: x.missing_items, not_verified: x.not_verified })))}`,
  { label: 'completeness-critic', phase: 'Critique', schema: GAPS_SCHEMA }
)

const gaps = (critique && critique.gaps) ? critique.gaps : []
const GAP_CAP = 5
if (gaps.length > GAP_CAP) log(`Gap-fill capped at ${GAP_CAP}; dropped: ${gaps.slice(GAP_CAP).map(g => g.title).join('; ')}`)

phase('Gap-fill')
const gapFills = await parallel(gaps.slice(0, GAP_CAP).map((g, i) => () => agent(
  `${CONTEXT}\n\nGAP TO RESEARCH: ${g.title}\nWhy it matters: ${g.why_it_matters}\nInstructions: ${g.research_instructions}\n\nResearch this gap thoroughly from primary sources and verify each item you report. Return a compact corrected digest per the schema.`,
  { label: `gap:${i + 1}`, phase: 'Gap-fill', schema: DIGEST_SCHEMA }
)))

return { digests, critique, gapFills: gapFills.filter(Boolean) }

```

</details>

---

## 10. How to resume in a new session

1. Check out the branch: `git fetch origin claude/pensive-ptolemy-huq2mx && git checkout claude/pensive-ptolemy-huq2mx`.
2. Read this file end to end, especially §1 (hard rules), §4 (status), §8 (next steps) and the update log.
3. If formal Plan Mode is available, enter it (`EnterPlanMode`). Do all work read-only, except updates to this `handover.md`, which need the user's consent to leave Plan Mode briefly for the commit.
4. Continue from the first unfinished item in §8. Keep §4, §7 and the update log current, and commit and push `handover.md` after each meaningful milestone.
5. End by presenting the plan for approval. **Never** begin implementation without explicit approval.

---

## 11. Update log

| Date | Session | Change |
|---|---|---|
| 2026-10-09 | session_01WyyihpQzFz2ZYAdX8EKpJj | Created handover. Stage 1 done; Stage 2 research workflow `wf_181ed593-9f9` running; Anthropic API facts recorded; pre-research working hypotheses recorded as unverified. |
