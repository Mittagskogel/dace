#!/usr/bin/env python3

from dace.transformation import transformation as xf
from dace.transformation.subgraph import helpers
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace import data, dtypes

class PromiseGpucopy(xf.SingleStateTransformation):
    """ Removes Host to Device array copies and vice versa. """

    access_src = xf.PatternNode(nodes.AccessNode)
    access_dst = xf.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.access_src, cls.access_dst)]

    def can_be_applied(self, state: SDFGState, expr_idx: int, sdfg: SDFG,
                       permissive=False) -> bool:
        desc_src = sdfg.arrays[self.access_src.data]
        desc_dst = sdfg.arrays[self.access_dst.data]
        if not isinstance(desc_src, data.Scalar) and not isinstance(desc_dst, data.Scalar):
            # Find Host->Device or Device->Host
            storage_src = desc_src.storage == dtypes.StorageType.GPU_Global
            storage_dst = desc_dst.storage == dtypes.StorageType.GPU_Global

            if storage_src != storage_dst:
                print(f'Found {self.access_src.label}->{self.access_dst.label}')
                return True
        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        # Host to Device
        if sdfg.arrays[self.access_dst.data].storage == dtypes.StorageType.GPU_Global:
            access_host = self.access_src
            access_device = self.access_dst
        # Device to Host
        else:
            access_host = self.access_dst
            access_device = self.access_src

        state.remove_node(access_host)
        sdfg.arrays[access_device.data].transient = False

        # Check if remaining device node is isolated
        if state.in_degree(access_device) + state.out_degree(access_device) == 0:
            state.remove_node(access_device)
