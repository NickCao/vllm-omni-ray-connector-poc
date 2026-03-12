import pytest
import torch
import uuid
import ray

from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
    MooncakeTransferEngineConnector,
)
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)
from .ray_connector import RayConnector


@pytest.fixture(autouse=True, scope="session")
def ray_init():
    ray.init()


def transfer(tx, rx):
    from_stage = "stage_0"
    to_stage = "stage_1"
    req_id = str(uuid.uuid4())

    tx_hash, metadata = ray.get(tx.send.remote(from_stage, to_stage, req_id))
    rx_hash = ray.get(rx.recv.remote(from_stage, to_stage, req_id, metadata))
    assert tx_hash == rx_hash


@pytest.mark.benchmark(warmup=True, warmup_iterations=5)
@pytest.mark.parametrize(
    "connector, config",
    [
        (SharedMemoryConnector, {}),
        (
            MooncakeTransferEngineConnector,
            {
                "host": "auto",
                "zmq_port": 50051,
                "protocol": "rdma",
                "memory_pool_size": 2147483648,
                "memory_pool_device": "cpu",
            },
        ),
        (RayConnector, {"rdt": False}),
        (RayConnector, {"rdt": True}),
    ],
)
def test_connector(connector, config, benchmark):
    @ray.remote(num_gpus=1)
    class Tx:
        def __init__(self):
            self.conn = connector(config)

        def send(self, from_stage, to_stage, req_id):
            device = torch.device("cuda:0")
            data = torch.rand(4096, 4096, dtype=torch.float32, device=device)
            success, _, metadata = self.conn.put(
                from_stage, to_stage, req_id, {"tensor": data}
            )
            assert success
            return torch.hash_tensor(data).item(), metadata

    @ray.remote(num_gpus=1)
    class Rx:
        def __init__(self):
            self.conn = connector(config)

        def recv(self, from_stage, to_stage, req_id, metadata):
            device = torch.device("cuda:0")
            data, _ = self.conn.get(from_stage, to_stage, req_id, metadata)
            return torch.hash_tensor(data["tensor"].to(device=device)).item()

    tx = Tx.remote()
    rx = Rx.remote()

    benchmark(transfer, tx, rx)
