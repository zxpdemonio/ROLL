"""Inference-role packing must not import the Megatron training backend."""
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize('experts,indices',[(256,[0,255]),(512,[256,511])])
def test_postprocess_router_artifacts_without_megatron(experts,indices):
    script=textwrap.dedent('''
        import importlib.abc
        import json
        import sys
        class NoMegatron(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'megatron' or fullname.startswith('megatron.'):
                    raise ModuleNotFoundError('Megatron is unavailable in inference role', name=fullname)
        sys.meta_path.insert(0,NoMegatron())
        import torch
        from tensordict import TensorDict
        from roll.distributed.scheduler.protocol import DataProto
        from roll.utils.execution_artifacts import get_expert_dtype
        from roll.utils.functionals import postprocess_generate
        experts=int(sys.argv[1]);indices=json.loads(sys.argv[2]);dtype=get_expert_dtype(experts)
        prompts=DataProto(batch=TensorDict({'input_ids':torch.tensor([[10,11]]),
            'attention_mask':torch.ones((1,2),dtype=torch.int64),
            'position_ids':torch.tensor([[0,1]])},batch_size=[1]))
        routes=torch.tensor([[indices],[indices],[indices]],dtype=dtype)
        packed=postprocess_generate(prompts,torch.tensor([[10,11,12,13]]),1,4,-2,-1,
            routed_experts=[routes],num_moe_experts=experts)
        actual=packed.batch['routed_experts']
        assert actual.dtype==dtype and tuple(actual.shape)==(1,4,1,2)
        torch.testing.assert_close(actual[0,:3],routes,atol=0,rtol=0)
        assert actual[0,3,0].tolist()==[0,1]
        assert not any(name=='megatron' or name.startswith('megatron.') for name in sys.modules)
    ''')
    result=subprocess.run([sys.executable,'-c',script,str(experts),str(indices)],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
