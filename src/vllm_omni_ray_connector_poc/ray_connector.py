from typing import Any

import ray
from ray import ObjectRef

from vllm_omni.distributed.omni_connectors.utils.logging import get_connector_logger
from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase

logger = get_connector_logger(__name__)


@ray.remote
class RayRefStore:
    def __init__(self):
        self._store: dict[str, ObjectRef] = {}

    def put(self, key: str, ref: list[ObjectRef]) -> None:
        self._store[key] = ref[0]

    def get(self, key: str) -> ObjectRef | None:
        return self._store.get(key)


class RayConnector(OmniConnectorBase):
    def __init__(self, config: dict[str, Any]):
        self._store = RayRefStore.options(
            name="ray_ref_store", get_if_exists=True
        ).remote()

    def cleanup(self, request_id: str) -> None:
        pass

    def health(self) -> dict[str, Any]:
        pass

    def close(self) -> None:
        pass

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            key = self._make_key(put_key, from_stage, to_stage)
            ref = ray.put(data, _tensor_transport="nixl")
            self._store.put.remote(key, [ref])

            return True, 0, None
        except Exception as e:
            logger.error(f"RayConnector: put failed for key {put_key}: {e}")
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        try:
            key = self._make_key(get_key, from_stage, to_stage)
            ref = self._store.get.remote(key)
            data = ray.get(ray.get(ref))

            return data, 0
        except Exception as e:
            logger.error(f"RayConnector: get failed for key {get_key}: {e}")
            return None
