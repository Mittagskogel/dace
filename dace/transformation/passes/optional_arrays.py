# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Dict, Iterator, Optional, Set, Tuple

from dace import SDFG, data, properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class OptionalArrayInference(ppl.Pass):
    """
    Infers the ``optional`` property of arrays, i.e., if they can be given None, throughout the SDFG and all nested
    SDFGs.

    An array will be modified to optional if its previous ``optional`` property was set to ``None`` and:
    * it is transient;
    * it is in a nested SDFG and its parent array was transient; or
    * it is definitely (unconditionally) read or written in the SDFG.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If connectivity or any edges were changed, some new descriptors may be marked as optional
        return modified & (ppl.Modifies.States)

    def apply_pass(self,
                   sdfg: SDFG,
                   _,
                   parent_arrays: Optional[Dict[str, bool]] = None) -> Optional[Set[Tuple[int, str]]]:
        """
        Infers the ``optional`` property of arrays in the SDFG and its nested SDFGs.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :param parent_arrays: If not None, contains values of determined arrays from the parent SDFG.
        :return: A set of the modified array names as a 2-tuple (CFG ID, name), or None if nothing was changed.
        """
        result: Set[Tuple[int, str]] = set()
        parent_arrays = parent_arrays or {}

        cfg_id = sdfg.cfg_id

        # Set information of arrays based on their transient and parent status
        for aname, arr in sdfg.arrays.items():
            if not isinstance(arr, data.Array):
                continue
            if arr.transient:
                if arr.optional is not False:
                    result.add((cfg_id, aname))
                arr.optional = False
            if aname in parent_arrays:
                if arr.optional is not parent_arrays[aname]:
                    result.add((cfg_id, aname))
                arr.optional = parent_arrays[aname]

        # Change unconditionally-accessed arrays to non-optional
        for state in self.traverse_unconditional_blocks(sdfg, recursive=True):
            for anode in state.data_nodes():
                desc = anode.desc(sdfg)
                if isinstance(desc, data.Array) and desc.optional is None:
                    desc.optional = False
                    result.add((cfg_id, anode.data))

        # Propagate information to nested SDFGs
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    # Create information about parent arrays
                    pinfo: Dict[str, bool] = {}
                    for e in state.in_edges(node):
                        if e.data.is_empty():
                            continue
                        opt = sdfg.arrays[e.data.data].optional
                        if opt is not None:
                            pinfo[e.dst_conn] = opt
                    for e in state.out_edges(node):
                        if e.data.is_empty():
                            continue
                        opt = sdfg.arrays[e.data.data].optional
                        if opt is not None:
                            pinfo[e.src_conn] = opt

                    # Apply pass recursively
                    rec_result = self.apply_pass(node.sdfg, _, pinfo)
                    if rec_result:
                        result.update(rec_result)

        return result or None

    def traverse_unconditional_blocks(self,
                                      cfg: ControlFlowRegion,
                                      recursive: bool = True,
                                      produce_nonstate: bool = False) -> Iterator[ControlFlowBlock]:
        """
        Traverse CFG and keep track of whether the block is executed unconditionally.
        """
        ipostdom = sdutil.postdominators(cfg)
        curblock = cfg.start_block
        out_degree = cfg.out_degree(curblock)
        while out_degree > 0:
            if produce_nonstate:
                yield curblock
            elif isinstance(curblock, SDFGState):
                yield curblock

            if recursive and isinstance(curblock, ControlFlowRegion) and not isinstance(curblock, LoopRegion):
                yield from self.traverse_unconditional_blocks(curblock, recursive, produce_nonstate)
            if out_degree == 1:  # Unconditional, continue to next state
                curblock = cfg.successors(curblock)[0]
            elif out_degree > 1:  # Conditional branch
                # Conditional code follows, use immediate post-dominator for next unconditional state
                curblock = ipostdom[curblock]
            # Compute new out degree
            if curblock in cfg.nodes():
                out_degree = cfg.out_degree(curblock)
            else:
                out_degree = 0
        # Yield final state
        if produce_nonstate:
            yield curblock
        elif isinstance(curblock, SDFGState):
            yield curblock

    def report(self, pass_retval: Set[Tuple[int, str]]) -> str:
        return f'Inferred {len(pass_retval)} optional arrays.'
