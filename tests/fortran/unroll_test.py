# Copyright 2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests loop unrolling functionality."""
import numpy as np

from typing import Dict, Optional, Iterable

from fparser.two.Fortran2003 import Program
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.ast_desugaring import correct_for_function_calls, deconstruct_enums, \
    deconstruct_interface_calls, deconstruct_procedure_calls, deconstruct_associations, \
    assign_globally_unique_subprogram_names, assign_globally_unique_variable_names, prune_branches, \
    const_eval_nodes, prune_unused_objects, inject_const_evals, ConstTypeInjection, ConstInstanceInjection, \
    make_practically_constant_arguments_constants, make_practically_constant_global_vars_constants, \
    exploit_locally_constant_variables, create_global_initializers, convert_data_statements_into_assignments, \
    deconstruct_statement_functions, deconstuct_goto_statements, SPEC, remove_access_and_bind_statements, \
    identifier_specs, alias_specs, consolidate_uses, consolidate_global_data_into_arg, prune_coarsely, \
    unroll_loops
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string, construct_full_ast
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str], entry_points: Optional[Iterable[SPEC]] = None):
    parser = ParserFactory().create(std="f2008")
    ast = construct_full_ast(sources, parser, entry_points=entry_points)
    ast = correct_for_function_calls(ast)
    assert isinstance(ast, Program)
    return ast


def test_fortran_frontend_loop_unroll():
    """Tests whether basic unrolling transformation works."""
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  integer :: idx, d(5)
  do, idx=1,3
    d(1) = d(1) + 1
  end do
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = unroll_loops(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  INTEGER :: idx, d(5)
  d(1) = d(1) + 1
  d(1) = d(1) + 1
  d(1) = d(1) + 1
END SUBROUTINE main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()
    # g = create_singular_sdfg_from_string(sources, 'main')
    # g.validate()
    # g.simplify()
    # d = np.full([5], 5, order="F", dtype=np.int32)
    # g(d=d)
    # assert(d[0] == 8)

def test_fortran_frontend_loop_unroll_index():
    """Tests whether unrolling transformation correctly replaces indices."""
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  integer :: idx, d(5)
  do, idx=1,3
    d(1) = d(1) + idx
  end do
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = unroll_loops(ast)
    print(ast.children)
    ast = exploit_locally_constant_variables(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  INTEGER :: idx, d(5)
  idx = 1
  d(1) = d(1) + 1
  idx = 2
  d(1) = d(1) + 2
  idx = 3
  d(1) = d(1) + 3
END SUBROUTINE main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_fortran_frontend_local_assigns():
    """
    Tests that local assignment statements are respected by the const replacer.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  integer :: idx, d(5)
  idx = 1
  d(1) = d(1) + idx
  idx = 2
  d(1) = d(1) + idx
  idx = 3
  d(1) = d(1) + idx
end subroutine main
""", 'main').check_with_gfortran().get()
    ast = parse_and_improve(sources)
    print(ast.children)
    ast = exploit_locally_constant_variables(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  INTEGER :: idx, d(5)
  idx = 1
  d(1) = d(1) + 1
  idx = 2
  d(1) = d(1) + 2
  idx = 3
  d(1) = d(1) + 3
END SUBROUTINE main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


if __name__ == "__main__":
    test_fortran_frontend_loop_unroll()
