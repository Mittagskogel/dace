# Copyright 2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests loop unrolling functionality."""
import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_loop_unroll():
    """Tests that unrolling transformation works."""
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  integer :: idx, d(5)
  do, idx=1,3
    d(1) = d(1) + 1
  end do
end subroutine main
""").check_with_gfortran().get()
    g = create_singular_sdfg_from_string(sources, 'main')
    g.validate()
    g.simplify()
    d = np.full([5], 5, order="F", dtype=np.int32)
    g(d=d)
    assert(d[0] == 8)


if __name__ == "__main__":
    test_fortran_frontend_loop_unroll()
