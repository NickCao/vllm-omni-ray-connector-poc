import pytest
import torch
import uuid
import ray

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

    tx_hash = ray.get(tx.send.remote(from_stage, to_stage, req_id))
    rx_hash = ray.get(rx.recv.remote(from_stage, to_stage, req_id))
    assert tx_hash == rx_hash


@pytest.mark.benchmark(warmup=True, warmup_iterations=5)
@pytest.mark.parametrize(
    "connector, config",
    [
        (SharedMemoryConnector, {}),
        (RayConnector, {"rdt": False}),
        (RayConnector, {"rdt": True}),
    ],
)
def test_connector(connector, config, benchmark):
    @ray.remote(num_gpus=0.5)
    class Tx:
        def __init__(self):
            self.conn = connector({})

        def send(self, from_stage, to_stage, req_id):
            device = torch.device("cuda:0")
            data = torch.rand(4096, 4096, dtype=torch.float32, device=device)
            success, _, _ = self.conn.put(
                from_stage, to_stage, req_id, {"tensor": data}
            )
            assert success
            return torch.hash_tensor(data).item()

    @ray.remote(num_gpus=0.5)
    class Rx:
        def __init__(self):
            self.conn = connector({})

        def recv(self, from_stage, to_stage, req_id):
            device = torch.device("cuda:0")
            data, _ = self.conn.get(from_stage, to_stage, req_id, None)
            return torch.hash_tensor(data["tensor"].to(device=device)).item()

    tx = Tx.remote()
    rx = Rx.remote()

    benchmark(transfer, tx, rx)
