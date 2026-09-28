# AerialDojo 代码使用说明

中文 | [English](README.en.md)

本项目提供 ProjectAirSim 无人机环境、地图管理服务、在线策略评测和轨迹 RGB-D
录制。内置 `trajectory` 动作回放与 `cliph` 文本目标导航策略。

数据目录和 episode 对应关系见上一级[数据集 README](../README.md)。本文集中
说明安装、配置、运行及程序接口。

## 1. 代码目录

```text
AerialDojo/                         # 代码项目根目录，命令在这里执行
├── README.md / README.en.md     # 中英文代码使用说明
├── AerialDojo/                     # Python 包
│   ├── env_uav.py                 # reset、step、观测与任务状态
│   ├── env_utils_uav.py           # 位姿、动作与距离计算
│   ├── eval_policy.py             # 在线评测入口
│   ├── policies/                  # 策略接口、工厂、CLIP-H、轨迹策略
│   ├── projectairsim_plugin/      # 地图服务和 ProjectAirSim 客户端
│   └── trajectory_recording/      # 回放录制、续录及索引汇总
├── config/                        # 服务、策略和传感器配置
├── scripts/                       # 启动脚本
├── tests/
└── requirements.txt
```

本机代码目录为 `/DATA/DATANAS1/AerialDojo/AerialDojo`，数据根目录为它的父目录。

## 2. 运行环境与安装

UE `.sh` 地图在能访问数据的 Linux GPU 服务器上运行。NAS 可以只提供存储，
实际渲染与采图由运行 UE 的服务器完成。

当前测试使用 Python 3.10；Python 3.12 尚未作为本项目验证环境。

```bash
cd /DATA/DATANAS1/AerialDojo/AerialDojo
conda create -n aerialdojo python=3.10
conda activate aerialdojo
python -m pip install -r requirements.txt
```

已有环境时直接激活。还需从与地图插件匹配的 ProjectAirSim 源码目录安装客户端：

```bash
python -m pip install -e /path/to/ProjectAirSim/client/python/projectairsim
```

将上述路径替换为实际源码位置。ProjectAirSim SDK 不由本项目 requirements.txt
自动提供。CLIP-H 另需可读取的模型权重；轨迹录制不加载 CLIP 模型。

基础检查不启动 UE，也不下载模型：

```bash
python -B -m unittest discover -s tests -v
```

## 3. 直接录制现有任务与轨迹

录制入口已支持当前发布目录，可直接使用 SemanticOGS、TrajectoryDATA 和
AerialENVS。每个分区的任务 `episode_id=n` 对应相同分区下的 `n.json`；无需
重新规划、重新划分任务或准备其他索引。

### 3.1 配置

修改 `config/record_jobs.yaml`：

| 配置项 | 默认值或作用 |
| --- | --- |
| `dataset_root` | `..`，相对于代码项目根目录 |
| `server.root_path` | 数据根目录下的 AerialENVS |
| `server.host` / `server.port` | 127.0.0.1 / 38000，地图管理 RPC |
| `server.endpoint_base_port` | 38100，场景端口分配起点 |
| `recording.output_root` | 数据根目录下的 VideoRECORD |
| `task_source.map_name` | 默认 N_Island_0，可由命令行覆盖 |
| `task_source.splits` | 默认 IID_TRAINS、OOD_TRAINS |
| `task_source.tasks` | 默认 base、standard、long |
| `gpus` | GPU 和场景实例配置，默认 GPU 0 |

### 3.2 启动与录制

两个终端均激活同一 Python 环境并进入代码项目目录。

终端 1 启动录制服务：

```bash
bash scripts/record_server.sh
```

终端 2 先检查所选任务的目录与分配，再录制一条完整轨迹：

```bash
bash scripts/record.sh --map_name N_Island_0 --tasks base --limit 1 --dry_run
bash scripts/record.sh --map_name N_Island_0 --tasks base --limit 1
```

录制该地图的全部训练任务：

```bash
bash scripts/record.sh --map_name N_Island_0
```

指定测试集：

```bash
bash scripts/record.sh --map_name N_Island_1 --splits OOD_TESTS
```

| 参数 | 作用 |
| --- | --- |
| `--map_name` | 选择地图 |
| `--splits` | 选择 IID_TRAINS、IID_TESTS、OOD_TRAINS、OOD_TESTS |
| `--tasks` | 选择 base、standard、long |
| `--episode_ids 0 5 12` | 选择分区内任务编号，通常同时指定 split 和 task |
| `--limit` | 最多选择多少条任务；0 表示不限，在续录扫描前应用 |
| `--max_steps` | 每条最多采多少帧；0 表示完整录制 |
| `--gpus` | GPU 列表，每张 GPU 一个场景 |
| `--dataset_root` | 覆盖数据根目录；服务脚本也支持该参数 |
| `--output_root` | 覆盖录制输出根目录 |
| `--dry_run` | 只读取任务并显示分配，不启动 UE、不写录制数据 |
| `--overwrite` | 重录选中的任务；默认使用断点续录 |

多 GPU 时，两个终端使用一致的 GPU 池：

```bash
# 终端 1
bash scripts/record_server.sh --gpus 2,4
# 终端 2
bash scripts/record.sh --map_name N_Island_0 --gpus 2 4
```

同一录制进程可并行使用多个 GPU；多个独立进程同时录制不同地图时，应使用
不同的地图管理服务与不重叠的 RPC/场景端口配置。

### 3.3 输出、分辨率与恢复

默认输出为 `../VideoRECORD/<split>/<task>/<map>_<B|S|L>/<episode_id>/`。
前、左、右、下四个相机默认均为 **640×640**，RGB 为 PNG，深度为米制
float32 NPY，另存逐帧位置、姿态、动作与任务元数据。

录制采用无碰撞、非物理位姿回放，配置为：

- `config/sim_config/scene_aerialdojo_drone_nonphysics.jsonc`
- `config/sim_config/robot_aerialdojo_quadrotor_nonphysics.jsonc`

更改图像尺寸时，修改机器人配置中对应相机 RGB 与深度的 capture-settings。
UE 显示窗口尺寸不等于传感器输出分辨率。

重新执行同一命令即可续录：完成任务跳过，部分任务从未完成帧继续。默认先写
`/tmp/aerialdojo_record_stage`，再异步提交 NAS；`--local_stage_root ''` 可直接
写最终目录。分批录制按唯一 recording_id 合并索引，不丢弃前面批次的记录。

全部录制进程停止后，可重建全局索引：

```bash
python -m AerialDojo.trajectory_recording.rebuild_collected_index \
  --record-root /DATA/DATANAS1/AerialDojo/VideoRECORD
```

## 4. 在线策略评测

### 4.1 服务配置

在 `config/server_config.yaml` 中将 `server.root_path` 改为实际 AerialENVS
目录，本机为 `/DATA/DATANAS1/AerialDojo/AerialENVS`，并设置所用 GPU。
默认 RPC 为 36000，场景端口从 36100 开始分配，每个场景使用 topic/service
端口对。

后台渲染使用 `show_game: false`。有图形桌面的服务器可设为 true；显示窗口
不影响客户端是否能读取图像。

### 4.2 输入格式与配置

当前在线评测使用自身的任务/轨迹输入格式，录制入口则直接支持发布格式。
在线配置中的旧 `example/...` 数据不随当前代码提供，运行前必须修改相关路径。

| 内容 | 在线入口要求 |
| --- | --- |
| 任务文件 | JSON 数组，包含 map_name、start_pose、goal_pose 等字段 |
| 起点位置 | `start_pose.start_position`，NED 米 |
| 起点四元数 | `start_pose.start_quaternionr`，顺序为 xyzw |
| 目标位置 | `goal_pose.goal_position`，NED 米 |
| CLIP-H 文本 | `description` |
| 轨迹策略输入 | JSONL，每行一条轨迹，包含 trajectory_id、actions |
| 轨迹匹配 | 优先使用任务的 trajectory_id，也支持起终点对象 ID |

公开任务的起点字段是 `start_quaternion_xyzw`。给在线入口使用时，需要在自己的
数据加载或转换过程中提供同值的 `start_quaternionr`，不改变四元数顺序。

公开的逐任务轨迹 JSON 不能直接当作 trajectory 策略的 JSONL。整理动作时去掉
初始 start，保留后续动作并在末尾补 stop；如提供内部 steps，位置字段为
position_m，四元数为 quaternion_wxyz。匹配任务时仍以同分区 episode_id 找到
原始轨迹，不使用数组下标或 actor 名称猜测发布文件。

在 `config/policy_config.yaml` 中配置：

| 配置项 | 用途 |
| --- | --- |
| `task_file` | 符合在线输入格式的任务文件路径 |
| `policy` | trajectory、cliph 或自定义策略 |
| `policy_config.trajectory_file` | trajectory 策略使用的 JSONL 路径 |
| `gpu_id` | UE 场景使用的 GPU |
| `episodes` | 运行任务数，0 为该文件全部任务 |
| `max_actions` | 每回合动作数上限，默认 300 |
| `output` | 每任务一行的评测结果 JSONL |

### 4.3 启动评测

完成输入与配置后，终端 1 启动在线服务：

```bash
bash scripts/server.sh
```

终端 2 运行一个轨迹策略任务：

```bash
bash scripts/run_policy.sh --config config/policy_config.yaml --episodes 1
```

CLIP-H 使用独立配置，也需要先设置其中的 task_file 与 gpu_id：

```bash
bash scripts/run_policy.sh --config config/policy_config_cliph.yaml --episodes 1
```

CLIP-H 使用实时四路图像、深度和 description，默认模型为
`openai/clip-vit-base-patch16`，也可在 policy_config 中设置本地 model_name。
ImageOGS 的目标图片不是当前 CLIP-H 的输入。

在线环境默认使用 `scene_collision.jsonc`：reset 直接放置起点，随后移动执行
体积碰撞扫描。策略需在目标成功半径内主动 Stop 才按停止规则判成功；碰撞或
超过步数上限也会结束回合。录制完成状态不能代替在线导航评测结果。

## 5. 接入自己的程序

### 5.1 自定义策略与工厂

自己的策略继承 `NavigationPolicy`，用 `reset(observation)` 重置回合状态，
在 `forward(observation)` 中返回 `PolicyAction` 枚举。

`NavigationPolicy`、`PolicyAction` 和 `PolicyFactory` 均可从
`AerialDojo.policies` 导入。

`PolicyFactory` 负责按名字找到策略类，再用参数创建实例，支持两种接入方式：

- **直接加载类**：设置 `--policy my_package.my_policy:MyPolicy`，模块可被 Python
  导入即可，无需注册，也无需修改项目的 `policies/__init__.py`。
- **按注册名加载**：在自己的类上使用 `@PolicyFactory.register("my_policy")`，
  运行时设置 `--policy-module my_package.my_policy --policy my_policy`。
  `--policy-module` 会先导入模块，让注册生效，再由工厂创建策略。

`policy_config` 或 `--policy-config` 中的参数传给策略构造函数。切换策略时应同时
替换旧参数，例如不需要参数的策略设置 `--policy-config '{}'`，避免继续传入
trajectory 策略的 `trajectory_file`。

自己的程序也可用 `PolicyFactory.create("my_policy", **参数字典)` 创建已注册策略，
用 `PolicyFactory.available()` 查看注册名。评测入口随后依次调用策略的 reset 和
forward；工厂本身不负责执行动作，环境负责地图、采图、动作和结束判定。

### 5.2 直接使用环境接口

已有训练或控制循环时，使用 `SimpleUAVEnv`：创建时提供 task_file 或 tasks，
然后依次调用 reset、step、close。构造对象不会立即打开 UE，reset 才启动或
复用场景。

| 接口 | 返回值或作用 |
| --- | --- |
| `reset(indices=...)` | 初始观测列表 |
| `step(actions)` | observations、rewards、dones、infos |
| `step_gymnasium(actions)` | observations、rewards、terminated、truncated、infos |
| `observe(include_images=False)` | 只读取状态，不获取图像 |
| `close()` | 清理连接和环境管理的场景 |

所有回合结果按 batch 返回列表，即使 batch_size=1 也如此；动作列表长度应与
当前 batch 一致。SimpleUAVEnv 提供 Gymnasium 风格返回值，但不是带完整
observation_space/action_space 的 gymnasium.Env。

自己的循环如需与在线评测一致，应显式设置
`UAVEnvConfig.terminate_on_success=False`，等待策略 Stop。

### 5.3 观测与动作

| observation 字段 | 含义 |
| --- | --- |
| `rgb` | 前、左、右、下四路图像，默认 PNG bytes |
| `depth` | 同顺序的 float32 深度数组，单位米 |
| `pose` | NED 位置与 xyzw 四元数，共 7 个数 |
| `task` | 当前任务及归一化字段 |
| `state` / `imu` | 状态和 IMU 信息 |
| `trajectory` | 本回合历史位姿、动作和距离 |
| `step` / `move_distance` | 已执行步数与累计移动距离 |
| `distance_to_goal` | 到目标锚点的距离 |
| `done` / `success` / `collision` | 回合状态 |
| `oracle_success` | 本回合是否曾进入成功半径 |
| `terminated` / `truncated` / `termination_reason` | 终止、截断及原因 |

策略应根据实验协议选择模型输入；目标真实位置和距离等评估字段不应被无意当作
感知输入。图片目标策略可自行读取 task.image，但需要正确解析目标图片路径。

| PolicyAction | 环境动作字符串 | 默认行为 |
| --- | --- | --- |
| MoveForward / MoveLeft / MoveRight | forward / left / right | 相对当前朝向移动 1 米 |
| MoveUp / MoveDown | ascend / descend | 上升/下降 1 米 |
| TurnLeft / TurnRight | rotl / rotr | 左转/右转 15° |
| Stop | stop | 主动停止 |

在线评测要求策略返回枚举；直接调用 env.step 时传入动作字符串列表。
在其他项目中导入本包时，将代码项目根目录加入 PYTHONPATH，并使用明确的任务
和配置路径。

## 6. 结果与常见问题

评测 JSONL 每个任务一行，包含任务 ID、策略、步数、成功、碰撞、oracle_success、
终止原因、距离和奖励。再次使用相同输出路径会覆盖评测文件。逐帧录制使用
VideoRECORD 的独立索引和续录机制。

| 现象 | 检查项 |
| --- | --- |
| 找不到 example 路径 | 修改在线策略 YAML 的 task_file 和 trajectory_file |
| 缺少起点姿态字段 | 检查在线输入的 start_quaternionr 与公开字段名的区别 |
| 缺少 trajectory_id / actions | 在线轨迹策略需要 JSONL 动作格式 |
| 缺少 projectairsim | 在当前环境安装与地图插件匹配的 SDK |
| 服务监听但没有 UE | 地图在客户端 reset 或开始录制时启动 |
| 找不到地图启动脚本 | 检查 root_path 和 IID_ENVS/OOD_ENVS 下的同名 .sh |
| 连接失败 | 检查管理 RPC 与场景 topic/service 端口、GPU 和服务配置 |
| 无可见窗口 | 默认后台渲染仍能采图；窗口显示与传感器分辨率分别配置 |

单元测试和 dry-run 不启动 UE。实际渲染、碰撞接口与模型推理需在匹配的地图、
ProjectAirSim SDK、GPU 和模型环境下运行。
