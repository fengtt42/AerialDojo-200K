# AerialDojo Code Usage

[中文](README.md) | English

This project provides a ProjectAirSim UAV environment, scene management, online policy
evaluation, and trajectory RGB-D recording. Built-in policies are `trajectory` for
action playback and `cliph` for text-goal navigation.

See the parent [dataset README](../README.en.md) for directory structure and episode
correspondence. This document covers installation, configuration, execution, and APIs.

## 1. Code layout

```text
AerialDojo/                         # Code project root; run commands here
├── README.md / README.en.md
├── AerialDojo/                     # Python package
│   ├── env_uav.py                 # Reset, step, observations, episode state
│   ├── env_utils_uav.py           # Poses, actions, and distances
│   ├── eval_policy.py             # Online evaluation entry point
│   ├── policies/                  # Interfaces, factory, CLIP-H, trajectory policy
│   ├── projectairsim_plugin/      # Scene server and ProjectAirSim client
│   └── trajectory_recording/      # Recording, resume, and index rebuilding
├── config/                        # Server, policy, and sensor configuration
├── scripts/                       # Launch scripts
├── tests/
└── requirements.txt
```

On this server the code root is `/DATA/DATANAS1/AerialDojo/AerialDojo`; its parent is
the dataset root. Replace these paths when installing elsewhere.

## 2. Environment and installation

Run UE `.sh` launchers on a Linux GPU server with access to the dataset. NAS may provide
storage only; the server running UE performs rendering and image capture.

Tests currently use Python 3.10. Python 3.12 has not been validated for this project.

```bash
cd /DATA/DATANAS1/AerialDojo/AerialDojo
conda create -n aerialdojo python=3.10
conda activate aerialdojo
python -m pip install -r requirements.txt
```

Activate an existing environment instead of creating another one if appropriate.
Install the ProjectAirSim client from a source tree matching the plugin in your maps:

```bash
python -m pip install -e /path/to/ProjectAirSim/client/python/projectairsim
```

Replace the source path with its actual location. The ProjectAirSim SDK is not installed
automatically by this project's requirements.txt. CLIP-H also requires model weights;
trajectory recording does not load CLIP.

Basic checks do not launch UE or download models:

```bash
python -B -m unittest discover -s tests -v
```

## 3. Record published tasks and trajectories

The recorder directly supports SemanticOGS, TrajectoryDATA, and AerialENVS. A task with
`episode_id=n` uses `n.json` in the same trajectory partition. Replanning, resplitting,
and additional route indexes are not required.

### 3.1 Configuration

Edit `config/record_jobs.yaml`:

| Setting | Default or purpose |
| --- | --- |
| `dataset_root` | `..`, relative to the code project root |
| `server.root_path` | AerialENVS under the dataset root |
| `server.host` / `server.port` | 127.0.0.1 / 38000, scene-manager RPC |
| `server.endpoint_base_port` | 38100, first scene endpoint port |
| `recording.output_root` | VideoRECORD under the dataset root |
| `task_source.map_name` | N_Island_0 by default; overridable on the command line |
| `task_source.splits` | IID_TRAINS and OOD_TRAINS by default |
| `task_source.tasks` | base, standard, and long by default |
| `gpus` | GPU and scene slots; GPU 0 by default |

### 3.2 Start and record

Activate the same Python environment and enter the code root in both terminals.
Start the recording server in terminal 1:

```bash
bash scripts/record_server.sh
```

In terminal 2, inspect the selected task, then record one complete trajectory:

```bash
bash scripts/record.sh --map_name N_Island_0 --tasks base --limit 1 --dry_run
bash scripts/record.sh --map_name N_Island_0 --tasks base --limit 1
```

Record all training tasks for the map:

```bash
bash scripts/record.sh --map_name N_Island_0
```

Select a test split:

```bash
bash scripts/record.sh --map_name N_Island_1 --splits OOD_TESTS
```

| Argument | Purpose |
| --- | --- |
| `--map_name` | Select a map |
| `--splits` | Select IID_TRAINS, IID_TESTS, OOD_TRAINS, or OOD_TESTS |
| `--tasks` | Select base, standard, or long |
| `--episode_ids 0 5 12` | Select partition-local IDs; usually also specify split and task |
| `--limit` | Maximum selected tasks; 0 means unlimited, applied before resume scanning |
| `--max_steps` | Maximum captured frames per task; 0 records the full trajectory |
| `--gpus` | GPU list, with one scene per GPU |
| `--dataset_root` | Override the dataset root; also supported by the server script |
| `--output_root` | Override the recording output root |
| `--dry_run` | Read tasks and print assignments without UE or recording writes |
| `--overwrite` | Rerecord selected tasks; resume is the default |

Use matching GPU pools in both terminals:

```bash
# Terminal 1
bash scripts/record_server.sh --gpus 2,4
# Terminal 2
bash scripts/record.sh --map_name N_Island_0 --gpus 2 4
```

One recorder can use multiple GPUs. Independent processes recording different maps
concurrently need separate scene-manager services and non-overlapping RPC/scene ports.

### 3.3 Output, resolution, and recovery

The default output is `../VideoRECORD/<split>/<task>/<map>_<B|S|L>/<episode_id>/`.
Front, left, right, and down cameras all default to **640×640**. RGB is PNG; depth is
float32 NPY in meters. Poses, actions, and task metadata are saved alongside frames.

Recording uses non-physics pose playback with collisions disabled:

- `config/sim_config/scene_aerialdojo_drone_nonphysics.jsonc`
- `config/sim_config/robot_aerialdojo_quadrotor_nonphysics.jsonc`

Change the relevant cameras' RGB and depth capture-settings in the robot configuration
to change image resolution. The UE display-window size is a separate setting.

Rerun the same command to resume. Completed tasks are skipped; partial tasks continue
from the first incomplete frame. Frames are staged in `/tmp/aerialdojo_record_stage`
and copied asynchronously to NAS by default. Use `--local_stage_root ''` to write
directly to the output directory. Indexes merge by recording_id and preserve earlier
recording batches.

After all recording processes have stopped, rebuild the global index if needed:

```bash
python -m AerialDojo.trajectory_recording.rebuild_collected_index \
  --record-root /DATA/DATANAS1/AerialDojo/VideoRECORD
```

## 4. Online policy evaluation

### 4.1 Scene-server configuration

Set `server.root_path` in `config/server_config.yaml` to the actual AerialENVS path,
`/DATA/DATANAS1/AerialDojo/AerialENVS` on this machine, and select the GPU pool.
The default RPC port is 36000. Scene endpoints start at 36100, with a topic/service
port pair for each scene.

Use `show_game: false` for offscreen rendering. On a server with a desktop, set it to
true to display a window. A visible window is not required for client image capture.

### 4.2 Input formats and policy configuration

The online evaluator currently uses its own task/trajectory input format. The recorder
directly accepts the published format. Old `example/...` paths in online configurations
refer to data not included in the current code package; update them before running.

| Input | Online interface requirement |
| --- | --- |
| Task file | JSON array with map_name, start_pose, goal_pose, and related fields |
| Start position | `start_pose.start_position`, NED meters |
| Start quaternion | `start_pose.start_quaternionr`, xyzw order |
| Goal position | `goal_pose.goal_position`, NED meters |
| CLIP-H text | `description` |
| Trajectory-policy file | JSONL, one trajectory per line with trajectory_id and actions |
| Trajectory matching | Task trajectory_id takes priority; object IDs are also supported |

Published tasks use `start_quaternion_xyzw`. Your loader or conversion step must provide
the same values under `start_quaternionr` for the online interface, without reordering.

A published per-episode trajectory JSON is not the trajectory policy's JSONL format.
When preparing actions, omit the initial start, retain later actions, and append stop.
If supplying internal steps, use position_m and quaternion_wxyz. Still locate the original
published route through the same partition's episode_id; do not infer filenames from
actor names or array positions.

Configure `config/policy_config.yaml`:

| Setting | Purpose |
| --- | --- |
| `task_file` | Task file matching the online input format |
| `policy` | trajectory, cliph, or a custom policy |
| `policy_config.trajectory_file` | JSONL used by the trajectory policy |
| `gpu_id` | GPU used by the UE scene |
| `episodes` | Number of tasks; 0 runs all tasks in the file |
| `max_actions` | Per-episode action limit, 300 by default |
| `output` | Evaluation JSONL with one summary per task |

### 4.3 Run evaluation

After preparing inputs and configuration, start the online server in terminal 1:

```bash
bash scripts/server.sh
```

Run one trajectory-policy episode in terminal 2:

```bash
bash scripts/run_policy.sh --config config/policy_config.yaml --episodes 1
```

CLIP-H uses a separate configuration. Set its task_file and gpu_id before running:

```bash
bash scripts/run_policy.sh --config config/policy_config_cliph.yaml --episodes 1
```

CLIP-H uses four current images, depth, and description. Its default model is
`openai/clip-vit-base-patch16`; model_name in policy_config can point to local weights.
The current CLIP-H implementation does not consume ImageOGS target reference images.

Online evaluation defaults to `scene_collision.jsonc`: reset places the vehicle directly,
and later movement performs a volume collision sweep. The policy must stop within the
goal success radius to succeed under the stopping rule. Collisions and step limits also
end episodes. Recording completion is not an online navigation success metric.

## 5. Integrate your own program

### 5.1 Custom policies and the factory

Subclass `NavigationPolicy`, reset episode state in `reset(observation)`, and return a
`PolicyAction` enum value from `forward(observation)`.

Import `NavigationPolicy`, `PolicyAction`, and `PolicyFactory` from
`AerialDojo.policies`.

`PolicyFactory` resolves a name to a policy class and constructs it with the supplied
parameters. It supports two ways to load your policy:

- **Load the class directly:** set `--policy my_package.my_policy:MyPolicy`. The module
  only needs to be importable; registration and edits to policies/__init__.py are unnecessary.
- **Load by a registered name:** decorate your class with
  `@PolicyFactory.register("my_policy")`, then use
  `--policy-module my_package.my_policy --policy my_policy`.
  The module is imported first so registration happens before construction.

Parameters from policy_config or `--policy-config` are passed to the constructor.
Replace old parameters when switching policies. For a no-argument policy, use
`--policy-config '{}'` so the trajectory policy's trajectory_file is not passed through.

Your own program can call `PolicyFactory.create("my_policy", **config)` to construct
a registered policy and `PolicyFactory.available()` to list registered names. The
evaluator then calls reset and forward. The factory does not execute actions; the
environment handles scenes, observations, actions, and episode termination.

### 5.2 Use the environment directly

For an existing training or control loop, construct SimpleUAVEnv with task_file or tasks,
then call reset, step, and close. Construction does not launch UE; reset starts or reuses
the scene.

| Method | Return values or purpose |
| --- | --- |
| `reset(indices=...)` | Initial observation list |
| `step(actions)` | observations, rewards, dones, infos |
| `step_gymnasium(actions)` | observations, rewards, terminated, truncated, infos |
| `observe(include_images=False)` | Read state without capturing images |
| `close()` | Clean up connections and scenes managed by this environment |

Results are batch lists even for batch_size=1. The action-list length must match the
current batch. SimpleUAVEnv offers Gymnasium-style return values but is not a complete
gymnasium.Env with observation_space/action_space definitions.

To match the online evaluator, explicitly set
`UAVEnvConfig.terminate_on_success=False` and wait for the policy to stop.

### 5.3 Observations and actions

| Observation field | Meaning |
| --- | --- |
| `rgb` | Front, left, right, down images; PNG bytes by default |
| `depth` | Same camera order, float32 arrays in meters |
| `pose` | NED position plus xyzw quaternion, seven numbers |
| `task` | Current task and normalized fields |
| `state` / `imu` | State and IMU information |
| `trajectory` | Episode history of poses, actions, and distances |
| `step` / `move_distance` | Action count and traveled distance |
| `distance_to_goal` | Distance to the goal anchor |
| `done` / `success` / `collision` | Episode state |
| `oracle_success` | Whether the success radius was entered at any point |
| `terminated` / `truncated` / `termination_reason` | Termination, truncation, and reason |

Choose model inputs according to your evaluation protocol. Avoid unintentionally feeding
evaluation ground truth, such as target coordinates or distance, into a perception model.
An image-goal policy can load task.image, resolving its path correctly.

| PolicyAction | Environment action string | Default behavior |
| --- | --- | --- |
| MoveForward / MoveLeft / MoveRight | forward / left / right | Move 1 meter relative to heading |
| MoveUp / MoveDown | ascend / descend | Move up/down 1 meter |
| TurnLeft / TurnRight | rotl / rotr | Turn left/right 15 degrees |
| Stop | stop | Stop explicitly |

The evaluator requires enum values from policies. Direct env.step calls use a list of
action strings. To import this package from another project, add the code project root
to PYTHONPATH and use explicit task/configuration paths.

## 6. Results and troubleshooting

Evaluation JSONL contains one row per task with task ID, policy, steps, success, collision,
oracle_success, termination reason, distances, and reward. Reusing an output path
overwrites that evaluation file. Frame recording uses separate VideoRECORD indexes
and resume behavior.

| Symptom | What to check |
| --- | --- |
| Missing example path | Update task_file and trajectory_file in the online policy YAML |
| Missing start-pose field | Check the online start_quaternionr field versus the published field name |
| Missing trajectory_id / actions | The online trajectory policy requires the JSONL action format |
| Missing projectairsim | Install the SDK matching the map plugin in the active environment |
| Server listening without UE | A client reset or recording request launches the map |
| Missing map launcher | Check root_path and same-name .sh files under IID_ENVS/OOD_ENVS |
| Connection failure | Check RPC, topic/service ports, GPUs, and server configuration |
| No visible window | Offscreen rendering still supports capture; window and sensor sizes are separate |

Unit tests and dry-run do not launch UE. Rendering, collision interfaces, and model
inference must be exercised with compatible maps, SDK, GPU, and model configuration.
