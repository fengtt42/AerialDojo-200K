# AerialDojo 数据集结构

中文 | [English](README.en.md)

本文说明完整数据集在本地的目录结构，包括地图、导航任务、规划轨迹和录制数据。
代码使用方法见 [AerialDojo/README.md](AerialDojo/README.md)。

**当前 GitHub 发布包含代码、中英文 README 和数据目录占位文件。** 地图、任务、
轨迹及图像数据将另行发布到 Hugging Face，下载链接会补充在本页。克隆 GitHub
仓库不会下载这些数据；获得数据后，将各数据目录放到仓库根目录，保持下面的
相对路径即可。录制和在线评测需要先准备对应地图与任务数据。

## 1. 总体目录

```text
AerialDojo/
├── README.md / README.en.md     # 中英文数据集结构说明
├── AerialENVS/                  # UE / ProjectAirSim 打包地图
│   ├── IID_ENVS/
│   └── OOD_ENVS/
├── SemanticOGS/                 # 文本目标导航任务
├── ImageOGS/                    # 图片目标导航任务及目标参考图片
├── TrajectoryDATA/              # 与任务一一对应的预计算轨迹
├── VideoRECORD/                 # 沿轨迹录制的 RGB、深度、位姿和动作
├── BENCHMARK/                   # 当前预留目录
└── AerialDojo/                  # 代码项目，内部 README 说明使用方法
```

| 目录 | 内容 | 使用方式 |
| --- | --- | --- |
| `AerialENVS` | 可执行的 UE 地图及 ProjectAirSim 插件 | 在 Linux GPU 服务器上运行场景 |
| `SemanticOGS` | 起点、目标、文本描述、地标和任务属性 | 文本目标导航、任务分析 |
| `ImageOGS` | 与语义任务对应的图片目标任务和参考图片 | 图片目标导航 |
| `TrajectoryDATA` | 规划位姿与动作序列 | 路线分析、回放、录制 |
| `VideoRECORD` | 执行录制后生成的逐帧观测与元数据 | 训练、可视化和后处理 |

SemanticOGS 与 ImageOGS 是同一导航任务的两种目标表示，统计任务总量时不应
重复相加。TrajectoryDATA 不包含沿途 RGB-D；VideoRECORD 按需生成，目录存在
不代表已经录制完毕。离线读取数据文件不需要启动 UE。

## 2. 地图与数据划分

地图名称采用 `<环境类别>_<环境家族>_<场景编号>`，例如 `N_Island_0`。

| 前缀 | 类别 | 地图例子 |
| --- | --- | --- |
| `N` | 自然环境 Natural | `N_Island_0`、`N_Coast_0` |
| `U` | 城市环境 Urban | `U_Mall_0`、`U_Factory_0` |
| `I` | 基础设施 Infrastructure | `I_Bridge_0`、`I_RailCorridor_0` |
| `D` | 灾害环境 Disaster | `D_Earthquake_0`、`D_Flood_0` |

当前发布规则将 `_0` 地图放入 IID 组，`_1` 地图放入 OOD 组：

| 任务划分目录 | 对应地图目录 | 含义 |
| --- | --- | --- |
| `IID_TRAINS` | `AerialENVS/IID_ENVS` | IID 地图上的训练任务 |
| `IID_TESTS` | `AerialENVS/IID_ENVS` | IID 地图上的测试任务 |
| `OOD_TRAINS` | `AerialENVS/OOD_ENVS` | OOD 地图上提供的训练/适配任务 |
| `OOD_TESTS` | `AerialENVS/OOD_ENVS` | OOD 地图上的测试任务 |

同一地图的 Train/Test 使用同一个 UE 场景。划分针对有向任务点对，不保证目标
actor、语义类别或地标完全互斥。使用 OOD_TRAINS 适配时，应与未使用适配数据的
跨场景泛化实验区分。

地图启动脚本位于 `AerialENVS/<IID_ENVS 或 OOD_ENVS>/<地图>/<地图>.sh`。

## 3. 任务类型与分区命名

SemanticOGS、ImageOGS、TrajectoryDATA 使用相同的三级分区：

```text
<数据划分>/<任务类型>/<地图>_<B|S|L>_<Train|Test>/
```

| 任务类型目录 | 缩写 | JSON 的 task 字段 | 当前生成器的完成长度区间 |
| --- | --- | --- | --- |
| `1_BaseTasks` | `B` | `base_task` | `[5, 30)` 米 |
| `2_StandardTasks` | `S` | `standard_task` | `[30, 60)` 米 |
| `3_LongHorizonTasks` | `L` | `long_task` | `>= 60` 米，上限可按地图配置 |

完成长度为路径平移长度加终点到目标的残余距离。发布数据已经完成分类、筛选和
划分，读取时使用现有目录与任务字段即可，无需重新分类。

## 4. 任务与轨迹的 episode 一一对应关系

**每个分区独立编号，从 `0` 到 `N-1`。在同一分区内，SemanticOGS 的任务、
ImageOGS 的任务和 TrajectoryDATA 的轨迹按 `episode_id` 严格一一对应。**

例如分区为 `IID_TRAINS/1_BaseTasks/N_Island_0_B_Train`：

```text
SemanticOGS/IID_TRAINS/1_BaseTasks/N_Island_0_B_Train/Task.json
ImageOGS/IID_TRAINS/1_BaseTasks/N_Island_0_B_Train/Task.json
TrajectoryDATA/IID_TRAINS/1_BaseTasks/N_Island_0_B_Train/
├── 0.json
├── 1.json
├── 2.json
└── ...
```

| SemanticOGS/Task.json 中的任务 | ImageOGS/Task.json 中的任务 | TrajectoryDATA 中的轨迹 |
| --- | --- | --- |
| `episode_id = "0"` | `episode_id = "0"` | `0.json` |
| `episode_id = "1"` | `episode_id = "1"` | `1.json` |
| `episode_id = "2"` | `episode_id = "2"` | `2.json` |
| `episode_id = "n"` | `episode_id = "n"` | `n.json` |

因此，一个完整分区有 N 条语义任务，就应有 N 条图片任务和 N 个对应轨迹文件。
两个 Task.json 都是任务数组；读取时按 `episode_id` 找到相应记录，再用该编号
拼接轨迹文件名即可。

需要区分以下几种编号：

- **episode_id 是分区内编号。** 不同地图、不同任务类型、Train/Test 之间分别
  从 0 开始；它们的 `0.json` 是不同任务。全局定位需要完整分区路径加 episode_id。
- **轨迹内部 task_id 是规划阶段编号。** 它可以与 episode_id 不同，不能用它
  替代公开目录中的 `n.json` 文件名。
- **目标图片名称不是 episode 编号。** 图片位置以 ImageOGS 任务的 `image`
  字段为准；多个任务可以引用同一张目标图片。
- **actor 名称不是任务编号。** 同一 actor 可能参与多个任务或锚点，不能只根据
  起终点 actor 名称推断某个 episode。

例如 `N_Island_0_B_Train` 中的 episode 7，仅对应三个数据目录下**这个分区**
中的 episode 7；与 `N_Island_0_B_Test` 或 `N_Island_0_S_Train` 中的 episode 7
没有编号上的关联。

## 5. SemanticOGS 任务字段

每个分区的 `Task.json` 为 JSON 数组，每个元素描述一条导航任务。

| 字段 | 含义 |
| --- | --- |
| `episode_id` | 分区内任务编号，字符串 |
| `map_name` | 地图名称 |
| `coordinate_system` | 坐标系统，公开数据为 ProjectAirSim NED |
| `start_true_name` / `goal_true_name` | 起点、目标的可读对象名称 |
| `start_object_name` / `goal_object_name` | UE actor 名称 |
| `start_pose.start_position` | 起点 xyz，单位米 |
| `start_pose.start_quaternion_xyzw` | 起点四元数，顺序为 xyzw |
| `goal_pose.goal_position` | 目标锚点 xyz，单位米 |
| `category` | 目标语义类别 |
| `Landmark` | 目标关联地标的文字描述 |
| `Direction` | 方向描述 |
| `description` | 目标对象的文本描述 |
| `task` | base_task、standard_task 或 long_task |
| `used-in-train` | 1 为训练任务，0 为测试任务 |
| `info.geodesic_distance` | 规划路径平移长度，米 |
| `info.euclidean_distance` | 起点到目标锚点的直线距离，米 |
| `info.complexity` | 完成长度除以直线距离得到的复杂度 |

复杂度计算包含终点残余距离，可能与 geodesic_distance / euclidean_distance
略有区别。地图之间的筛选阈值可以不同，不应使用一个固定阈值重新解释全部任务。

## 6. ImageOGS 图片目标

ImageOGS 与 SemanticOGS 保持相同的任务划分、episode_id、起点和目标。图片任务
以 `image` 字段提供目标参考图片，不包含语义任务中的 `description` 和 `Direction`。

`image` 相对于**当前 Task.json 所在目录**解析，通常为 `Images/<文件名>.png`。
复制图片目标数据时需要一起复制被引用的 Images 文件。参考图片用于描述目标，
不表示无人机沿轨迹采集的实时画面。

## 7. TrajectoryDATA 轨迹字段

每个 `<episode_id>.json` 为一个 JSON 对象。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 轨迹格式版本，如 hybrid_astar_trajectory_v6 |
| `map_name` | 所属地图 |
| `task_id` | 原始规划任务编号 |
| `status` | 规划状态 |
| `start` | 起点 actor、位置和姿态 |
| `goal` | 目标 actor、目标位置、目标半径与边界定义 |
| `trajectory` | 按顺序排列的位姿和动作列表 |
| `trajectory[].position_ned_m` | 当前位姿的位置，NED 米 |
| `trajectory[].quaternion_xyzw` | 当前姿态，xyzw |
| `trajectory[].action` | 到达当前位姿使用的动作 |

第一项动作为 `start`。后续动作描述从前一位姿到当前位姿的变化，末项不一定有
`stop`。目标通常是一个半径范围，所以末点不一定精确等于目标锚点。

公开任务和轨迹都使用 **NED 米制位置、[x, y, z, w] 四元数**。NED 的 z 正方向
向下，z 不能直接当作离地高度；回放已有数据不需要再次进行 UE→NED 变换。

## 8. VideoRECORD 录制数据

录制保留 split 和 task 目录，分区文件夹省去末尾的 `_Train` / `_Test`：

```text
VideoRECORD/IID_TRAINS/1_BaseTasks/N_Island_0_B/
├── 0/                           # 对应原分区 episode_id="0"
│   ├── step_*_camera_*.png       # 前、左、右、下四路 RGB
│   ├── depth/*.npy              # 同帧对应的深度
│   ├── trajectory_*_collected.jsonl
│   └── .recording_complete.json
└── collected_episodes.json      # 分区录制索引
```

默认 RGB 和深度为 640×640；RGB 保存为 PNG，DepthPerspective 深度保存为米制
float32 NPY。元数据包含原始 episode_id、规划 task_id、任务来源、每帧位姿和动作。

全局 `VideoRECORD/collected_episodes.jsonl` 汇总录制信息；跨分区使用唯一的
`recording_id` 区分任务，不能只使用 episode_id。录制帧动作表示“看到当前观测后
要执行的下一动作”，最后一帧为 stop，与原始轨迹项的动作排列方式有一位偏移。

录制输出为逐帧数据，不自动合成 MP4。图像引用可能包含原机器的绝对路径，移动
录制目录后需相应更新元数据路径。

## 9. 当前数据检查与已知限制

2026-09-28 对发布前的本地完整数据目录进行了只读检查：42 张地图的 252 个分区，共 102,866 条
导航任务及对应轨迹。语义任务、图片任务和轨迹文件的 episode 对应关系、共享
字段与轨迹起点/目标绑定均通过检查；9,504 个被引用的目标图片文件均可读，
尺寸均为 640×640。本次未重新验证规划结果、碰撞或实际启动 UE。

- **缺少一张地图包**：`AerialENVS/OOD_ENVS/U_Community_1` 不存在。该地图
  的 394 条任务与轨迹已提供，但在线运行或录制前需要补齐地图包。其余 41 张
  地图的启动脚本、脚本引用的主程序及 pak 文件存在。
- **12 个长程分区为空**：下表所列分区的 Task.json 是空数组，对应轨迹目录
  也为空；不属于任务与轨迹错配。批量读取与统计时应跳过空分区，并避免除零。
- **VideoRECORD 尚无录制结果**：当前只有预建目录，没有 episode 录制数据。
  如需沿途 RGB-D，请按照代码 README 执行录制。
- **在线评测需要准备输入**：在线 YAML 仍引用未提供的 `example/...` 文件，
  在线入口的起点四元数字段和轨迹格式也与发布格式不同。转换说明见
  [代码 README 第 4 节](AerialDojo/README.md#4-在线策略评测)。录制入口直接支持
  当前发布格式。

| 空的长程分区 | 地图 |
| --- | --- |
| Train 和 Test 均为空 | D_Earthquake_0、D_Explosion_0、D_Flood_0、N_Forest_0、U_ParkingLot_1 |
| 仅 Train 为空 | N_Mountain_1 |
| 仅 Test 为空 | N_Snowfield_0 |
