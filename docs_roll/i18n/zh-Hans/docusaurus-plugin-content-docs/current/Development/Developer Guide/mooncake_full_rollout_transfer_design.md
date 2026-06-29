# Mooncake DataProto Rollout 传输

ROLL 可以使用 Mooncake 作为大规模 rollout `DataProto` 的传输后端。目标是让 Ray 只承担控制面和轻量引用传递，让 Mooncake 承载重 payload。

## 适用场景

当 rollout batch 很大、完整 `DataProto` 继续通过 Ray object 序列化传输成本很高时，适合启用 Mooncake transfer。典型场景是 VLM 或多模态任务，因为大量数据可能位于 `non_tensor_batch`，而不只是 `batch` tensor 字段。

Mooncake 路径保持 ROLL 原有 `DataProto` 结构：

- `batch`
- `non_tensor_batch`
- `meta_info`

reward、reference、training worker 最终仍然消费本地 materialized 后的普通字段。Mooncake 是传输后端，不应该暴露成业务代码的新对象模型。

## 工作方式

在 rollout handoff 时，ROLL 把重 payload 写入 Mooncake，并通过 Ray 传递轻量 remote reference。下游 worker 根据需要 materialize 对应字段和行。

适配层使用 Mooncake structured-object API：

| ROLL 操作 | Mooncake API |
|---|---|
| 存储 rollout payload | `put_dataproto` |
| 追加派生字段 | `append_dataproto_fields` |
| 按字段/行读取 | `get_dataproto` |
| 通过 Ray 传引用 | `export_dataproto_ref` / `import_dataproto_ref` |
| 清理 owned remote payload | `cleanup_dataproto` |

ROLL 不创建 Mooncake `BufferPool`。buffer 注册、可复用 staging buffer、copy fallback 和 multi-buffer transfer policy 都属于 Mooncake 内部职责。

## 配置

通过 `transfer_backend` 启用：

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    client_scope: node
    protocol: rdma
    device_name: erdma_1
    local_hostname: 192.168.22.70
    metadata_server: P2PHANDSHAKE
    master_server_address: 192.168.22.70:50053
    global_segment_size: 34359738368
    local_buffer_size: 42949672960
    default_chunk_bytes: 536870912
    namespace: roll_rl
    key_prefix: roll_rl
    policy:
      max_inflight_put: 1
      put_mode: auto
      copy_mode: auto
```

关键字段：

- `master_server_address` / `master_server_addr`：已启动的 Mooncake master 地址。
- `local_hostname`：Mooncake 可见的本机地址。
- `metadata_server`：Mooncake metadata 模式，例如 `P2PHANDSHAKE`。
- `protocol`：性能验证应使用 `rdma`。
- `device_name` / `rdma_devices`：`protocol: rdma` 时必填。
- `client_scope`：`node` 表示每个 Ray 节点一个 Mooncake client actor；`process` 表示每个进程一个 client。
- `require_native_tensors`：校验 tensor 字段尽量走 Mooncake native tensor payload。

当配置里提供显式 master 地址时，ROLL 会直接把 `backend_config` 映射给 `MooncakeDistributedStore.setup`。Mooncake 环境变量/config-file 模式只作为没有显式 master 地址时的兼容 fallback。

## 部署注意事项

启动 ROLL job 前先启动 Mooncake master。生产配置不要依赖临时 Ray actor 环境变量，应把 store 参数放在 `transfer_backend.backend_config`。

RDMA 性能验证时需要确认：

- Mooncake master 和 client 版本匹配。
- 配置的 RDMA device 存在。
- `local_buffer_size` 足够覆盖待测 payload，避免 fallback registration 干扰结果。
- benchmark 尽量分开报告 setup/registration 成本和在线 put/get 成本。

## 验收清单

一次 Mooncake rollout transfer 验证应确认：

1. Ray 正常初始化。
2. Mooncake transfer backend 从显式 `backend_config` 初始化。
3. scheduler `DataProto.to_remote` 成功。
4. `put_dataproto` 存储 `batch`、`non_tensor_batch` 和 metadata。
5. 下游 worker 通过 `get_dataproto` materialize 所需字段。
6. 派生字段可以 append，不需要重写原始 rollout payload。
7. owning remote batch 只清理一次 Mooncake payload。
