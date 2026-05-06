# Mooncake rollout 数据传输优化报告

## 背景

ROLL 的 RLVR 训练流程中，rollout 侧会生成较大的 `DataProto`，随后需要把生成结果传回 driver / training 侧继续做 reference log-prob、old log-prob、reward、advantage 和 train step。原先主要依赖 Ray object store 传输，数据会经历序列化、对象存储 put/get、反序列化等阶段。

这次优化的目标是：在不改变上层 `DataProto` 语义的前提下，引入 Mooncake RDMA/store 作为 rollout transfer backend，降低 rollout 数据跨节点传输的 put/get 开销，并通过 profiling 明确瓶颈到底在后端传输还是在 `DataProto` 编解码。

## 做了哪些优化

### 1. 统一 transfer payload 格式

Mooncake backend 复用了 `ray_optimized` 的 transfer payload 格式：

- tensor 数据和 non-tensor 数据统一编码为 transfer payload；
- metadata 和 bulk payload 分离；
- 对 tensor / non-tensor decode 阶段增加细粒度 profiling；
- 在 materialize 后把 payload 内的 transfer stats 合并回 `data.meta_info["metrics"]`，使真实 pipeline 日志中可以直接看到各 domain 的 transfer 指标。

这样 Ray 和 Mooncake 的对比只反映 backend put/get 差异，尽量避免 payload 格式不同造成的干扰。

### 2. Mooncake store backend

新增 / 完善 Mooncake rollout transfer backend，核心路径是：

1. producer 侧把 `DataProto` 编码为 metadata + bulk chunks；
2. 使用 Mooncake store 写入对象；
3. consumer 侧先读取 metadata，得到 bulk layout；
4. 使用 `get_into_ranges` 把多个 bulk ranges scatter 到预分配 buffer；
5. 从 buffer 中 decode 回 `DataProto`。

关键点：

- 使用预注册 buffer，减少反复分配和注册；
- bulk payload 通过 range read 写入目标 buffer，避免额外拼接；
- range read 支持一个 key 下多个 fragment/range；
- profiling 中拆分 metadata range get、bulk range get、backend get、payload decode 等阶段。

### 3. 对齐新版 Mooncake `get_into_ranges` 语义

新版 Mooncake `get_into_ranges` 支持一个 key 对应多段 ranges。ROLL 侧调用形态从旧的 repeated-key 方式调整为：

```python
self.store.get_into_ranges([ptr], [[key]], [[dest_offsets]], [[src_offsets]], [[sizes]])
```

也就是一个 key group 下携带多个 offsets / sizes。对应的 fake store 和测试也同步改成相同语义。

### 4. 移除生产路径中的 `batch_get_buffer_ranges` fallback

`batch_get_buffer_ranges` 只是临时分析接口，不作为最终 ROLL-facing API。生产 ROLL 路径使用 `get_into_ranges`。

为了定位性能差异，曾临时加过环境变量切换，用于同环境对比：

- `batch_get_buffer_ranges`
- generic `get_into_ranges`
- fast-path `get_into_ranges`

最终 warmed profiling 显示 generic `get_into_ranges` 已经没有明显劣势，因此不需要再为 ROLL 增加新的公开 API。

### 5. Mooncake 内部 fast path 验证

在 Mooncake 侧实现并验证了针对 ROLL hot path 的内部 fast path：单 destination buffer、单 key、多 ranges，直接构造 `BatchTransferReadRanges` 输入，绕过部分通用路径处理。

但在充分 warmup 后，fast path 没有稳定优于 generic path。结果显示 bulk RDMA scatter transfer 本身耗时几乎一致，差异主要来自 metadata/client/Ray 调度抖动。因此当前推荐是继续使用 generic `get_into_ranges`，fast path 可作为内部优化保留，但不是必须依赖的接口。

### 6. 真实 pipeline 可观测性

在完整 ROLL RL pipeline 中，transfer metrics 会按 domain 写入最终 step metrics，例如：

- `{domain}/transfer/time/serialize`
- `{domain}/transfer/time/put`
- `{domain}/transfer/time/get`
- `{domain}/transfer/time/deserialize`
- `{domain}/transfer/profile/time_seconds/backend_get`
- `{domain}/transfer/profile/time_seconds/from_transfer_payload`
- `{domain}/transfer/profile/time_seconds/decode_tensors`
- `{domain}/transfer/profile/time_seconds/decode_non_tensors`
- `{domain}/transfer/throughput/e2e_mbps`

这使得 synthetic benchmark 和真实 RL 训练可以用同一套指标对齐分析。

## 测试环境

主要双机环境：

- Producer / Ray head: `192.168.22.70`
- Consumer / worker: `192.168.22.72`
- Ray address: `192.168.22.70:6391`
- Mooncake master: `192.168.22.70:50053`
- Mooncake HTTP metadata: `http://192.168.22.70:8083/metadata`
- Mooncake protocol: `rdma`

Mooncake generic `get_into_ranges` 测试条件：

- `ROLL_MOONCAKE_DISABLE_FAST_PATH=1`
- `ROLL_MOONCAKE_RANGE_GET_MODE` unset
- full RL pipeline 中使用 `MOONCAKE_GLOBAL_SEGMENT_SIZE=1GB`

注意：完整 RL pipeline 会同时拉起多个 Mooncake client。若每个 client 使用 `MOONCAKE_GLOBAL_SEGMENT_SIZE=16GB`，会出现 RDMA MR 注册耗尽，报错类似 `Failed to register memory ... Input/output error` / `Mooncake store setup failed with code -600`。最终 full RL 对比使用 1GB segment，避免并发 client setup 时注册过大的 RDMA segment。

## 单测 / benchmark 数据

### 1. range read 路径对比

每条路径都先跑完整 warmup，再跑 measured。对比对象：

- 临时 `batch_get_buffer_ranges`
- generic `get_into_ranges`
- fast-path `get_into_ranges`

平均结果：

| path | avg backend_get_s | avg get_s | avg metadata_range_get_s | avg bulk_range_get_s |
|---|---:|---:|---:|---:|
| batch_get_buffer_ranges | 0.043881 | 0.012624 | 0.003989 | 0.008499 |
| generic_get_into_ranges | 0.042767 | 0.011940 | 0.003316 | 0.008496 |
| fast_get_into_ranges | 0.044175 | 0.013357 | 0.004698 | 0.008535 |

结论：

- bulk scatter transfer 基本一致，`bulk_range_get_s` 都在约 `0.0085s`；
- generic `get_into_ranges` 平均略好于 batch 和 fast path；
- fast path 没有稳定收益；
- 旧版观察到的 batch 明显优势，在统一 warmup 和同一 rebuilt binary 后没有复现；
- 不需要为 ROLL 新增额外公开 range-read API。

### 2. realistic-shape 双机 synthetic benchmark

该 benchmark 构造接近真实 RL rollout 形态的 `DataProto`：

- 11 个 tensor fields；
- 12 个 non-tensor fields；
- 多 domain batch size；
- 双机 producer / consumer；
- Ray backend 与 Mooncake generic `get_into_ranges` 对比；
- 每个 backend 都做 full warmup 后再 measured。

平均结果：

| backend | avg serialize_s | avg put_s | avg get_s | avg deserialize_s | avg backend_get_s | avg e2e_mbps |
|---|---:|---:|---:|---:|---:|---:|
| ray_optimized | 1.080286 | 0.095956 | 0.086608 | 0.030341 | 0.116949 | 531.9 |
| mooncake | 1.045302 | 0.016749 | 0.014197 | 0.030632 | 0.044829 | 1355.7 |

加速比：

- e2e throughput: `1355.7 MB/s` vs `531.9 MB/s`，Mooncake 为 Ray 的 **2.55x**；
- put: `0.016749s` vs `0.095956s`，Mooncake **5.73x faster**；
- get: `0.014197s` vs `0.086608s`，Mooncake **6.10x faster**；
- backend_get: `0.044829s` vs `0.116949s`，Mooncake **2.61x faster**；
- deserialize 基本一致，说明 decode 不是 Mooncake/Ray backend 差异导致。

### 3. transfer backend regression tests

核心 transfer regression suite 已通过：

```bash
PYTHONPATH=/root/ROLL python -m pytest \
  /root/ROLL/tests/distributed/scheduler/test_rollout_transfer_backend.py -q
```

结果：

```text
8 passed
```

该测试覆盖了：

- rollout transfer backend 基本 round trip；
- fake Mooncake store 的新版 `get_into_ranges` 语义；
- 一个 key 多 ranges 的 scatter read 行为；
- Mooncake / Ray backend 兼容 transfer payload 的基本正确性。

## 真实 ROLL RL pipeline e2e 数据

### 测试数据说明

本节已有 e2e 指标来自完整 `RLVRPipeline`，不是 synthetic microbenchmark；但它使用的是 8 卡 text RLVR 小模型配置，用于验证真实训练链路中的 transfer metrics 能否被采集和对比。

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- config: `examples/qwen2.5-7B-rlvr_megatron/rlvr_config_8gpus`
- domains: `math_rule`、`code_sandbox`、`llm_judge`、`crossthinkqa`、`ifeval`
- `rollout_batch_size=256`
- `num_return_sequences_in_group=8`

另外按 Qwen2.5-VL-7B detection 数据配置启动过 16 卡 `RLVRVLMPipeline` 验证：

- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- train data: `/data/oss_bucket_0/yuzhao/data/One-RL-to-See-Them-All/Orsta-Data-47k/train/train_detection_v3det_4000.parquet`
- validation data: `/data/oss_bucket_0/yuzhao/data/One-RL-to-See-Them-All/Orsta-Data-47k/test/test_detection_coco_test_multi_2000.parquet`
- domain: `cv_detection=1.0`
- `rollout_batch_size=256`
- `num_return_sequences_in_group=8`
- actor train: GPU 0-7；actor infer: GPU 8-15；reference: GPU 0-7

该 VLM detection 数据已成功下载并加载，日志中确认 train parquet 为 `4000` rows。但本轮 16 卡 VLM run 在 vLLM worker 初始化阶段失败，原因是环境里的 `torch_c_dlpack_ext/libtorch_c_dlpack_addon_torch26-cuda.so` 与当前 torch/CUDA ABI 不匹配，报错 `undefined symbol: _ZNK3c106Device3strB5cxx11Ev`。因此下面的 e2e 表格暂时不包含 VLM detection 指标。

### Text RLVR e2e 指标

真实测试使用完整 `RLVRPipeline`：

- pipeline: `RLVRPipeline`
- `max_steps=1`
- 每个 backend 先跑一个完整 warmup run，再跑 measured run；
- Ray measured log: `/tmp/roll_full_rl_transfer/ray_measured.log`
- Mooncake measured log: `/tmp/roll_full_rl_transfer/mooncake_generic_measured_1gbseg.log`
- 汇总报告: `/tmp/roll_full_rl_transfer/full_rl_report.md`

Mooncake 条件：generic `get_into_ranges`，fast path disabled。

### Per-domain transfer metrics

| domain | backend | payload_mb | serialize_s | put_s | get_s | deserialize_s | backend_get_s | e2e_mbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| code_sandbox | ray_optimized | 118.312653 | 0.195400 | 0.164142 | 0.068976 | 0.044090 | 0.113066 | 1046.401823 |
| crossthinkqa | ray_optimized | 38.219781 | 0.053408 | 0.055768 | 0.019759 | 0.015481 | 0.035240 | 1084.548321 |
| ifeval | ray_optimized | 42.929008 | 0.054858 | 0.039054 | 0.024657 | 0.017910 | 0.042567 | 1008.494140 |
| llm_judge | ray_optimized | 38.071601 | 0.049089 | 0.034990 | 0.021359 | 0.015915 | 0.037274 | 1021.387262 |
| math_rule | ray_optimized | 155.227303 | 0.261514 | 0.217967 | 0.092985 | 0.051329 | 0.144314 | 1075.624560 |
| code_sandbox | mooncake_generic | 118.315521 | 0.134540 | 0.088209 | 0.012718 | 0.042642 | 0.055360 | 2137.201449 |
| crossthinkqa | mooncake_generic | 38.222206 | 0.051336 | 0.031917 | 0.004338 | 0.015738 | 0.020077 | 1903.825313 |
| ifeval | mooncake_generic | 42.930946 | 0.047569 | 0.057768 | 0.005344 | 0.017792 | 0.023136 | 1855.600619 |
| llm_judge | mooncake_generic | 38.073097 | 0.041103 | 0.034861 | 0.006365 | 0.016486 | 0.022852 | 1666.084153 |
| math_rule | mooncake_generic | 155.230812 | 0.183257 | 0.138288 | 0.114491 | 0.047894 | 0.162385 | 955.940901 |

### 平均 transfer 指标

| backend | avg payload_mb | avg serialize_s | avg put_s | avg get_s | avg deserialize_s | avg backend_get_s | avg from_payload_s | avg e2e_mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ray_optimized | 78.552069 | 0.122854 | 0.102384 | 0.045547 | 0.028945 | 0.074492 | 0.028798 | 1047.291221 |
| mooncake_generic | 78.554517 | 0.091561 | 0.070208 | 0.028652 | 0.028110 | 0.056762 | 0.027954 | 1703.730487 |

真实 full RL pipeline 中的效果：

- e2e throughput: Mooncake `1703.7 MB/s` vs Ray `1047.3 MB/s`，提升 **1.63x**；
- put time: Mooncake `0.070208s` vs Ray `0.102384s`，Mooncake **1.46x faster**；
- get time: Mooncake `0.028652s` vs Ray `0.045547s`，Mooncake **1.59x faster**；
- backend_get: Mooncake `0.056762s` vs Ray `0.074492s`，Mooncake **1.31x faster**；
- deserialize: Mooncake `0.028110s` vs Ray `0.028945s`，基本一致。

### Step-level 指标

| backend | time/step_generate | time/ref_log_probs_values | time/old_log_probs | time/step_train |
|---|---:|---:|---:|---:|
| ray_optimized | 96.713005 | 141.962311 | 148.076554 | 652.042706 |
| mooncake_generic | 91.547826 | 140.729243 | 146.652971 | 647.777688 |

说明：

- Mooncake 明显降低了 rollout transfer 的 put/get/backend_get；
- full step 端到端时间仍主要由 generation、reference / old log-prob 和 train step 主导；
- transfer 优化在真实 pipeline 中是可观测的，但不会线性转化为整体 step time 的同等比例下降，因为训练主路径还有大量非 transfer 计算。

## 结论

1. Mooncake RDMA/store backend 已经能作为 ROLL rollout transfer 的高性能 backend 使用。
2. 在 realistic-shape 双机 benchmark 中，Mooncake generic `get_into_ranges` 相比 Ray backend：
   - e2e throughput 提升 **2.55x**；
   - put 提升 **5.73x**；
   - get 提升 **6.10x**；
   - backend_get 提升 **2.61x**。
3. 在完整 ROLL RL pipeline 中，Mooncake generic `get_into_ranges` 相比 Ray backend：
   - e2e transfer throughput 提升 **1.63x**；
   - put 提升 **1.46x**；
   - get 提升 **1.59x**；
   - backend_get 提升 **1.31x**。
4. serialization / deserialization 基本是 `DataProto` 编解码成本，Mooncake 和 Ray 差异不大；Mooncake 的主要收益来自 backend transport put/get。
5. `batch_get_buffer_ranges` 不需要作为 ROLL 生产接口保留；generic `get_into_ranges` 在 warmup 后表现已经足够好。
6. fast-path `get_into_ranges` 没有稳定优于 generic path，不应作为 ROLL 侧新增 API 的依据。
7. full pipeline 使用 Mooncake 时需要合理设置 `MOONCAKE_GLOBAL_SEGMENT_SIZE`，避免多 client 并发 setup 时 RDMA memory registration 资源耗尽。

## 建议

- ROLL 生产路径使用 Mooncake generic `get_into_ranges`。
- 保留 profiling metrics，便于后续追踪真实训练中的 transfer 变化。
- 不新增 ROLL-facing `batch_get_buffer_ranges` API。
- Mooncake full RL 配置建议从 `MOONCAKE_GLOBAL_SEGMENT_SIZE=1GB` 起步，根据并发 client 数和 RDMA MR 限制再调大。
- 后续若继续优化，应优先看 `DataProto` 编解码和 full pipeline 计算阶段，而不是继续拆 range-read API。
