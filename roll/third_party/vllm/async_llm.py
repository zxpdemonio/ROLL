from vllm.v1.engine.async_llm import AsyncLLM


class CustomAsyncLLM(AsyncLLM):
    async def abort(self, request_id, *args, **kwargs):
        if not getattr(self, "_roll_mooncake_aux_enabled", False):
            return await super().abort(request_id, *args, **kwargs)
        from roll.distributed.scheduler.mooncake_direct import add_cleanup_note

        external = [request_id] if isinstance(request_id, str) else list(request_id)
        internal = set(external)
        for identity in external:
            internal.update(self.output_processor.external_req_ids.get(identity, ()))
        pending = getattr(self, "_roll_pending_aux_discards", set())
        pending.update(internal)
        self._roll_pending_aux_discards = pending
        failure = None
        try:
            return await super().abort(external if not isinstance(request_id, str) else request_id, *args, **kwargs)
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                await self.settle_aux_outputs()
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                add_cleanup_note(failure, cleanup_error)

    async def settle_aux_outputs(self, **kwargs):
        pending = set(getattr(self, "_roll_pending_aux_discards", ()))
        if pending:
            await self.engine_core.collective_rpc_async(
                method="settle_aux_outputs", kwargs={"request_ids": list(pending), "discard": True},
            )
            self._roll_pending_aux_discards.difference_update(pending)
        if kwargs:
            await self.engine_core.collective_rpc_async(method="settle_aux_outputs", kwargs=kwargs)

    async def custom_init_worker(self):
        await self.engine_core.collective_rpc_async(method="custom_init_worker")

    async def load_states(self):
        await self.engine_core.collective_rpc_async(method="load_states")

    async def offload_states(self, level):
        if getattr(self, "_roll_pending_aux_discards", None):
            await self.settle_aux_outputs()
        await self.reset_prefix_cache()
        await self.engine_core.collective_rpc_async(method="offload_states", args=(level,))

    async def setup_collective_group(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="setup_collective_group", args=args, kwargs=kwargs)

    async def broadcast_parameter(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="broadcast_parameter", args=args, kwargs=kwargs)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        await self.engine_core.collective_rpc_async(method="update_parameter_in_bucket", args=(serialized_named_tensors, is_lora))

    async def add_lora(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="custom_add_lora", args=args, kwargs=kwargs)

    async def process_weights_after_loading(self):
        await self.engine_core.collective_rpc_async(method="process_weights_after_loading")
