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
  idx = 1
  d(1) = d(1) + 1
  idx = 2
  d(1) = d(1) + 1
  idx = 3
  d(1) = d(1) + 1
END SUBROUTINE main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


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


def test_fortran_frontend_loop_unroll_index_step():
    """Tests whether unrolling transformation correctly replaces indices."""
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  integer :: idx, d(5)
  do, idx=1,5,2
    d(1) = d(1) + idx
  end do
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = unroll_loops(ast)
    ast = exploit_locally_constant_variables(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  INTEGER :: idx, d(5)
  idx = 1
  d(1) = d(1) + 1
  idx = 3
  d(1) = d(1) + 3
  idx = 5
  d(1) = d(1) + 5
END SUBROUTINE main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_fortran_frontend_loop_unroll_fancy():
    """Tests whether unrolling transformation correctly replaces indices."""
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  integer, parameter :: inc(3) = [3, 4, 2]
  type :: t
    integer :: a, b
  end type t
  type(t) :: arr(3)
  integer :: idx, jdx, tmp, d(5)
  idx = inc(1)
  do, jdx=1,d(2)
    do, idx=1,3
      tmp = inc(idx)
      d(idx) = d(idx) + tmp
    end do
  end do
  idx = 3
  arr(idx)%a = 5
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = unroll_loops(ast)
    ast = exploit_locally_constant_variables(ast)
    ast = const_eval_nodes(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  INTEGER, PARAMETER :: inc(3) = [3, 4, 2]
  TYPE :: t
    INTEGER :: a, b
  END TYPE t
  TYPE(t) :: arr(3)
  INTEGER :: idx, jdx, tmp, d(5)
  idx = 3
  DO , jdx = 1, d(2)
    idx = 1
    tmp = 3
    d(1) = d(1) + 3
    idx = 2
    tmp = 4
    d(2) = d(2) + 4
    idx = 3
    tmp = 2
    d(3) = d(3) + 2
  END DO
  idx = 3
  arr(3) % a = 5
END SUBROUTINE main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_fortran_frontend_loop_unroll_fancy():
    """Tests whether unrolling transformation correctly replaces indices."""
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  type t_qx_ptr
    integer, pointer :: ptr(:, :)
    integer, pointer :: qtr
  end type t_qx_ptr
  type(t_qx_ptr) :: q(3)
  integer, target :: x(2,3)
  integer :: d

  q(1) % ptr => x(:,:)
  q(1) % ptr(d,1) = 5
  q(1) % ptr(2,2) = 2
  q(1) % qtr => x(1,2)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = unroll_loops(ast)
    ast = exploit_locally_constant_variables(ast)
    ast = const_eval_nodes(ast)
    ast = prune_unused_objects(ast, [("main",)])

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  TYPE :: t_qx_ptr
  END TYPE t_qx_ptr
  TYPE(t_qx_ptr) :: q(3)
  INTEGER, TARGET :: x(2, 3)
  INTEGER :: d
  x(d, 1) = 5
  x(2, 2) = 2
END SUBROUTINE main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


if __name__ == "__main__":
    test_fortran_frontend_loop_unroll()
    test_fortran_frontend_loop_unroll_index()
    test_fortran_frontend_loop_unroll_index_step()
    test_fortran_frontend_loop_unroll_fancy()
