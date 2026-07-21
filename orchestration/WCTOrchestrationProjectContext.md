# WCT-GNN Project Orchestration Context

This is the project-specific runtime fragment for WCT-GNN work. It is not a
standalone orchestration role, research plan, or setup procedure. Reusable local
orchestration utilities combine it with a package assignment and resolved local
runtime/access paths in each generated worker context.

## WCT initializer settings

The WCT project profile requires:

- `EnvironmentRoot`: the existing shared Python environment;
- `DataRoot`: the existing shared EEG dataset root;
- `CoherentMultiplexRoot`: the selected read-only wavelet source;
- `GpuVisibleDevices`: the explicitly selected `CUDA_VISIBLE_DEVICES` value for
  queued GPU runs.

The initializer receives these values explicitly and records the resolved
paths only in parent-local runtime/context state. Tracked project instructions
must not embed a maintainer's machine paths. In particular, the initializer
must not guess a data root from historical or partial MNE-style directories.

## Shared Python environment

WCT experiments reuse one existing Python environment because reinstalling
PyTorch and the scientific stack per worktree is expensive. The environment is
a shared dependency resource, not a writable extension of a worker branch.

Workers must not install, upgrade, remove, or repair packages there. A missing
or incompatible dependency is reported to the orchestrator and becomes a
serialized human decision. The adapter's initialization probe imports the
required scientific stack without changing it.

The environment may contain an editable MOABB installation bound to another
checkout. Therefore using its Python executable alone does not prove which
MOABB source ran.

`ai_docs/RepoInfo.md` describes ordinary single-checkout development. During an
orchestrated package, this generated context overrides its environment,
external-source, data-root, result-path, and experiment-launch instructions.
RepoInfo remains useful for architecture and normal command orientation, but a
worker must not use its plain Python command as managed experiment evidence.

## MOABB and WCT source provenance

Managed WCT execution places the assigned worktree first on `PYTHONPATH` and
verifies that:

- `moabb.__file__` resolves below the assigned worktree;
- `coheriqs_contributions.moabb_pipelines.common` resolves below that worktree;
  and
- `utils.coherence_utils` resolves below the configured
  `CoherentMultiplexRoot`.

Any mismatch is a hard stop. A worker does not continue with an installed,
integration, source-checkout, or sibling-worktree substitute and merely note
the difference.

`CoherentMultiplexRoot` is read-only to ordinary packages. A genuine change to
that source requires a separately planned, serialized package with explicit
ownership. `WCT_COHERENT_MULTIPLEX_ROOT` selects the configured source during a
managed run.

## EEG dataset contract

Every worker that runs an EEG experiment requires read access to the configured
`DataRoot`. Keeping that path visible only to the orchestrator would prevent
the worker from running the assigned experiment and encourage private copies.

Workers must not:

- download a missing dataset;
- move, rename, repair, convert, or deduplicate the shared dataset tree;
- change user-wide MNE configuration;
- create a silent worktree-local data fallback; or
- treat failure to access a subject/session as permission to bootstrap data.

The managed launcher sets `MNE_DATA` and `MNE_DATASETS_BNCI_PATH` to this root
and exports `MOABB_DISABLE_DOWNLOADS=1`. Missing or inaccessible data is a
stopping condition, not permission to acquire, copy, or repair it.

The adapter allocates an execution-specific `_MNE_FAKE_HOME_DIR` and optional
`MNE_CACHE_DIR` below `shared/execution-state` so concurrent workers do not race
through one user configuration or mutable scratch state. The MNE cache
directory is created only if MNE actually uses it. Python bytecode writes are
disabled so imports do not leave `__pycache__` trees throughout the worker
checkout. These variables do not make the dataset physically read-only; the
no-mutation rule still applies.

If the current agent sandbox cannot read the environment, dataset, or wavelet
root, the package stops and asks the human to expose the required root. It does
not copy the resource into the worktree.

## MOABB results

MOABB's evaluation result-storage layer reads `MOABB_RESULTS` when constructing
its HDF5 path. The adapter sets it to an execution-specific directory so
different packages and executions do not share one result store.

`run_wct_gnn.py` derives a run ID automatically from the managed execution ID;
manual runs receive a timestamped ID unless `--run-id` is supplied. Every
evaluation/inner-group combination receives a distinct HDF5 suffix, preventing
one group from overwriting another. Beside each HDF5 store the runner writes a
CSV result table and Markdown summary containing the configured data root,
outer scores, means, and run-grid configuration. Workers link these
human-readable files instead of decoding HDF5 manually. HDF5 remains MOABB's
structured store;
it contains result rows and model metadata, not another copy of the raw EEG.

The managed launcher separately captures complete child stdout and stderr in
the execution result directory's `console.log`. The control log remains a
short lifecycle/resource trace.

## Derived-cache status

No reusable CWT or noise-bank cache exists yet. `shared/execution-state` is
mutable run isolation and must not be repurposed as that cache. Unless a package
assignment explicitly owns an accepted cache implementation, workers do not
create ad hoc per-worktree or machine-wide substitutes.

## WCT managed execution

The WCT adapter configures the shared Python executable and accepts a Python
script below the assigned worktree or standard Python module execution as the
managed command. For a module, pass `-Command '-m'` and put the module name
first in `-CommandArguments`, for example `@('pytest', '<focused-test-path>')`.
It verifies the source and MNE path contract before launching either form.

WCT experiments, evaluations, and project-importing tests or smoke runs used as
evidence use the managed-command launcher at the absolute path in the
assignment.

For orchestrator-owned combined validation, the same program-root launcher can
target the exact integration worktree with `-ExecutionScope Integration`; this
is preferred when the controlled environment and durable provenance are useful.
The orchestrator may still run an experiment directly when its judgment favors
the simpler path, but then it records the environment, configuration, result
location, and limitations itself and does not describe the run as equivalent
managed evidence.

WCT managed execution is CPU-only by default: the adapter sets
`CUDA_VISIBLE_DEVICES=-1`, so model configurations using `device="auto"` do not
silently contend for a GPU. A run intended to use CUDA passes `-Resource gpu`
to the general launcher. The launcher queues that program-wide exclusive
resource; after acquisition the adapter exposes the configured
`GpuVisibleDevices` value and verifies that PyTorch can see CUDA before the
experiment begins. Configuring this resource mechanism does not imply that a
GPU exists or is usable on every system; a requested GPU run fails preflight
when the configured environment cannot expose CUDA.

The execution record includes the selected data root, wavelet root, Python
executable, MOABB result directory, execution-state paths, child console log,
and launcher identity. Scientific conclusions and uncertainty remain in the
experiment ledger or package report; resolved scores and configuration are
also written automatically by the WCT runner.

## WCT-specific stopping conditions

Stop and report when:

- the shared Python environment is incomplete or incompatible;
- MOABB, contributions, or wavelet imports resolve from the wrong source;
- the EEG dataset root is empty, inaccessible, or missing required data;
- a run would download, repair, or mutate the shared dataset;
- a package would need to modify Coherent_Multiplex without ownership;
- user-wide MNE configuration would be changed.
