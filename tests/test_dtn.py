"""Tests for DTN (Dirichlet-to-Neumann) method — Issue #23."""

from __future__ import annotations

import numdiff as nd
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix
from stratix._types import Method


def _smatrix_ref(stack, wavelength, kx, polarization):
    return stratix.solve(stack, wavelength, kx, polarization, method=Method.SMATRIX)


def _dtn_solve(stack, wavelength, kx, polarization):
    return stratix.solve(stack, wavelength, kx, polarization, method=Method.DTN)


class TestDtnSingleInterface:
    def test_matches_smatrix_air_glass_TE(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12
        assert res.method_used == Method.DTN

    def test_matches_smatrix_air_glass_TM(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TM)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TM)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_air_silicon(self, set_backend):
        n_air, n_si = 1.0, 3.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_si**2),
        )
        wavelength = 5e-7
        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_off_normal_TE(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength

        for theta_deg in [15, 30, 60]:
            theta_rad = nd.array(nd.pi * theta_deg / 180)
            kx = float(n_air * k0 * nd.sin(theta_rad))
            ref = _smatrix_ref(stack, wavelength, kx=kx, polarization=Polarization.TE)
            res = _dtn_solve(stack, wavelength, kx=kx, polarization=Polarization.TE)

            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, (
                f"theta={theta_deg}: R mismatch"
            )
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, (
                f"theta={theta_deg}: T mismatch"
            )

    def test_matches_smatrix_off_normal_TM(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength

        for theta_deg in [15, 30, 60]:
            theta_rad = nd.array(nd.pi * theta_deg / 180)
            kx = float(n_air * k0 * nd.sin(theta_rad))
            ref = _smatrix_ref(stack, wavelength, kx=kx, polarization=Polarization.TM)
            res = _dtn_solve(stack, wavelength, kx=kx, polarization=Polarization.TM)

            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, (
                f"theta={theta_deg}: R mismatch"
            )
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, (
                f"theta={theta_deg}: T mismatch"
            )

    def test_matches_smatrix_metal_TE(self, set_backend):
        """Metallic substrate: high R, DTN matches S-matrix."""
        n_air = 1.0
        n_metal = 0.05 + 3.5j
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_metal**2),
        )
        wavelength = 5e-7
        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_total_internal_reflection(self, set_backend):
        n_glass, n_air = 1.5, 1.0
        stack = Stack(
            superstrate=Material(epsilon=n_glass**2),
            substrate=Material(epsilon=n_air**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength
        kx = 1.1 * n_air * k0
        ref = _smatrix_ref(
            stack, wavelength, kx=float(kx), polarization=Polarization.TE
        )
        res = _dtn_solve(stack, wavelength, kx=float(kx), polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_energy_conservation_lossless(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=4.0),
        )
        wavelength = 5e-7
        for kx in [0.0, 5e6, 1e7]:
            res = _dtn_solve(stack, wavelength, kx=kx, polarization=Polarization.TE)
            assert abs(float(res.R[0, 0]) + float(res.T[0, 0]) - 1.0) < 1e-12


class TestDtnMultiLayer:
    def test_matches_smatrix_ar_coating_TE(self, set_backend):
        n_air, n_mgf2, n_glass = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d_ar = wavelength / (4 * n_mgf2)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
            layers=[Layer(thickness=d_ar, material=Material(epsilon=n_mgf2**2))],
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_ar_coating_TM(self, set_backend):
        n_air, n_mgf2, n_glass = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d_ar = wavelength / (4 * n_mgf2)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
            layers=[Layer(thickness=d_ar, material=Material(epsilon=n_mgf2**2))],
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TM)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TM)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_two_layer(self, set_backend):
        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        wavelength = 5e-7
        d_a = wavelength / (4 * n_a)
        d_b = wavelength / (4 * n_b)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=d_a, material=Material(epsilon=n_a**2)),
                Layer(thickness=d_b, material=Material(epsilon=n_b**2)),
            ],
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_bragg_mirror_TE(self, set_backend):
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)
        n_air, n_sub = 1.0, 1.5

        layers = []
        for _ in range(5):
            layers.append(Layer(thickness=d_high, material=Material(epsilon=n_high**2)))
            layers.append(Layer(thickness=d_low, material=Material(epsilon=n_low**2)))

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=layers,
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_bragg_mirror_TM(self, set_backend):
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)
        n_air, n_sub = 1.0, 1.5

        layers = []
        for _ in range(5):
            layers.append(Layer(thickness=d_high, material=Material(epsilon=n_high**2)))
            layers.append(Layer(thickness=d_low, material=Material(epsilon=n_low**2)))

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=layers,
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TM)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TM)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_off_normal_multilayer_TE(self, set_backend):
        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        wavelength = 5e-7
        d_a = 100e-9
        d_b = 200e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=d_a, material=Material(epsilon=n_a**2)),
                Layer(thickness=d_b, material=Material(epsilon=n_b**2)),
                Layer(thickness=50e-9, material=Material(epsilon=n_a**2)),
            ],
        )

        for kx in [0.0, 5e6, 1e7]:
            ref = _smatrix_ref(stack, wavelength, kx=kx, polarization=Polarization.TE)
            res = _dtn_solve(stack, wavelength, kx=kx, polarization=Polarization.TE)

            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, (
                f"kx={kx}: R mismatch"
            )
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, (
                f"kx={kx}: T mismatch"
            )

    def test_matches_smatrix_off_normal_multilayer_TM(self, set_backend):
        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        wavelength = 5e-7
        d_a = 100e-9
        d_b = 200e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=d_a, material=Material(epsilon=n_a**2)),
                Layer(thickness=d_b, material=Material(epsilon=n_b**2)),
                Layer(thickness=50e-9, material=Material(epsilon=n_a**2)),
            ],
        )

        for kx in [0.0, 5e6, 1e7]:
            ref = _smatrix_ref(stack, wavelength, kx=kx, polarization=Polarization.TM)
            res = _dtn_solve(stack, wavelength, kx=kx, polarization=Polarization.TM)

            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, (
                f"kx={kx}: R mismatch"
            )
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, (
                f"kx={kx}: T mismatch"
            )

    def test_matches_smatrix_metallic_layer(self, set_backend):
        """Metal layer on glass substrate."""
        n_air, n_sub = 1.0, 1.5
        n_metal = 0.05 + 3.5j
        wavelength = 5e-7
        d_metal = 50e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d_metal, material=Material(epsilon=n_metal**2))],
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_zero_layers(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _dtn_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_energy_conservation_lossless_multilayer(self, set_backend):
        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        wavelength = 5e-7

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=100e-9, material=Material(epsilon=n_a**2)),
                Layer(thickness=200e-9, material=Material(epsilon=n_b**2)),
                Layer(thickness=50e-9, material=Material(epsilon=n_a**2)),
            ],
        )

        for kx in [0.0, 5e6, 1e7]:
            res = _dtn_solve(stack, wavelength, kx=kx, polarization=Polarization.TE)
            assert abs(float(res.R[0, 0]) + float(res.T[0, 0]) - 1.0) < 1e-12, (
                f"kx={kx}: R+T={float(res.R[0, 0]) + float(res.T[0, 0])}"
            )


class TestDtnIsItsOwnFormulation:
    """Issue #54: DTN must be a distinct algorithm, not a renamed copy.

    ``Method.DTN`` shipped as a line-for-line duplicate of the admittance
    recursion with ``Y`` spelled ``Z``.  Every numeric test in this module
    passed, because a rename cannot change an answer -- so the numbers
    cannot be what guards this.  The guard has to look at the code.
    """

    @staticmethod
    def _shape(func) -> str:
        """Return the function's structure with every identifier erased.

        Parses the source, drops the docstring, and rewrites each name,
        attribute and constant to a fixed placeholder.  What survives is
        the shape of the statements and expressions alone, so two
        functions compare equal exactly when one is the other with its
        variables renamed -- the failure this test exists to catch.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))

        class _Erase(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.Name:
                return ast.Name(id="_", ctx=node.ctx)

            def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
                self.generic_visit(node)
                return ast.Attribute(value=node.value, attr="_", ctx=node.ctx)

            def visit_arg(self, node: ast.arg) -> ast.arg:
                return ast.arg(arg="_", annotation=None)

            def visit_Constant(self, node: ast.Constant) -> ast.Constant:
                return ast.Constant(value=None)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                if ast.get_docstring(node) is not None:
                    node.body = node.body[1:]
                node.name = "_"
                node.decorator_list = []
                node.returns = None
                self.generic_visit(node)
                return node

        return ast.dump(_Erase().visit(tree))

    def test_dtn_body_is_not_the_admittance_body_renamed(self):
        from stratix.methods._admittance import _admittance_solve
        from stratix.methods._dtn import _dtn_solve

        assert self._shape(_dtn_solve) != self._shape(_admittance_solve), (
            "DTN and admittance have identical statement structure -- DTN is "
            "the admittance recursion with its variables renamed, not a "
            "Dirichlet-to-Neumann formulation (issue #54)."
        )


class TestDtnDirichletResonance:
    """Layers whose phase thickness is a multiple of pi.

    A layer with ``sin(kz*d) == 0`` is at a Dirichlet resonance: its
    Dirichlet-to-Neumann map does not exist, because the interior problem
    with prescribed face values is singular there.  The cot/csc kernel is
    infinite entry by entry, yet the stack's R and T are perfectly
    ordinary -- a half-wave layer is an absentee layer.  The assembled
    system is scaled so those infinities are never evaluated.
    """

    @staticmethod
    def _resonant_stack(indices, detune=0.0):
        wavelength = 5e-7
        return wavelength, Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=1.5**2),
            layers=[
                Layer(
                    thickness=wavelength / (2 * n) * (1 + detune),
                    material=Material(epsilon=n**2),
                )
                for n in indices
            ],
        )

    def test_near_resonant_single_layer(self, set_backend):
        """A half-wave layer detuned by a part in a billion.

        The detuning is far below any thickness a deposition process
        controls, so this is a half-wave layer for every practical
        purpose -- and it separates the singular point itself from its
        neighbourhood, where the scaled system is well conditioned and
        DTN is accurate to the last bits.  Without the scaling the
        cot/csc entries here are of order 1e9 and the answer is lost.
        """
        wavelength, stack = self._resonant_stack([1.5], detune=1e-9)

        for pol in (Polarization.TE, Polarization.TM):
            ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=pol)
            res = _dtn_solve(stack, wavelength, kx=0.0, polarization=pol)
            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, pol
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, pol

    def test_stack_of_near_resonant_layers(self, set_backend):
        """Three near-resonant layers at once, so none can be lucky."""
        wavelength, stack = self._resonant_stack([1.5, 2.0, 2.5], detune=1e-9)

        for pol in (Polarization.TE, Polarization.TM):
            ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=pol)
            res = _dtn_solve(stack, wavelength, kx=0.0, polarization=pol)
            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, pol
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, pol

    def test_exactly_resonant_layers(self, set_backend):
        """On the singular point itself, detuned back to an answer.

        A layer at exact Dirichlet resonance leaves the assembled
        operator rank deficient, so the solve has nothing to find and
        raises -- taking the rest of the sweep down with the one point
        that landed there.  Such a layer is rotated off the resonance by
        ``sqrt(eps)`` in phase instead, which caps the error at about
        ``1e-8``.  The bound is what is asserted here rather than the
        ``1e-15`` the same stacks happen to come back with, because
        which side of it a backend's factorisation lands on is not
        something the method promises.
        """
        for indices in ([1.5], [1.5, 2.0, 2.5]):
            wavelength, stack = self._resonant_stack(indices)
            for pol in (Polarization.TE, Polarization.TM):
                ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=pol)
                res = _dtn_solve(stack, wavelength, kx=0.0, polarization=pol)
                assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-8, (
                    f"{indices} {pol}"
                )
                assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-8, (
                    f"{indices} {pol}"
                )

    def test_zero_thickness_layer(self, set_backend):
        """A layer an optimiser has removed is the resonance at phi = 0.

        ``thicknesses=`` is the autodiff override, so a search over layer
        thicknesses reaches zero routinely.  A zero-thickness layer has
        no DtN map for the same reason a half-wave layer has none, and it
        is the one that turns up in practice.
        """
        wavelength = 5e-7
        material = Material(epsilon=1.9)
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=1.5**2),
            layers=[
                Layer(thickness=100e-9, material=material),
                Layer(thickness=80e-9, material=material),
            ],
        )
        bare = Stack(
            superstrate=stack.superstrate, substrate=stack.substrate, layers=[]
        )

        res = stratix.solve(
            stack,
            wavelength,
            0.0,
            Polarization.TE,
            method=Method.DTN,
            thicknesses=nd.array([0.0, 0.0]),
        )
        ref = _smatrix_ref(bare, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-8
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-8

    def test_thick_absorbing_layer_does_not_overflow(self, set_backend):
        """A layer many skin depths thick.

        The kernel is carried on ``exp(i phi)`` rather than on
        ``sin(phi)`` and ``cos(phi)``, which both grow like
        ``exp(Im phi) / 2``: written the direct way, the entries of a
        20 um metal layer overflow to infinity, equilibration turns that
        into NaN, and the solve refuses the whole sweep.
        """
        wavelength = 5e-7
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=1.5**2),
            layers=[
                Layer(thickness=20e-6, material=Material(epsilon=(0.05 + 3.5j) ** 2))
            ],
        )

        for pol in (Polarization.TE, Polarization.TM):
            ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=pol)
            res = _dtn_solve(stack, wavelength, kx=0.0, polarization=pol)
            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, pol
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, pol


class TestDtnLossyStack:
    def test_energy_balance_with_absorbing_layer(self, set_backend):
        """R + T + lumped absorption closes for an absorbing stack."""
        wavelength = 5e-7
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=1.5**2),
            layers=[
                Layer(thickness=20e-9, material=Material(epsilon=(0.05 + 3.5j) ** 2)),
                Layer(thickness=120e-9, material=Material(epsilon=2.0**2)),
            ],
        )
        for pol in (Polarization.TE, Polarization.TM):
            ref = _smatrix_ref(stack, wavelength, kx=5e6, polarization=pol)
            res = _dtn_solve(stack, wavelength, kx=5e6, polarization=pol)
            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, pol
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, pol


class TestDtnThicknessOverride:
    def test_thicknesses_override_is_honoured(self, set_backend):
        """Passing ``thicknesses`` reproduces the stack built with them."""
        wavelength = 5e-7
        material = Material(epsilon=2.0**2)
        overridden = [130e-9, 65e-9]

        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=1.5**2),
            layers=[
                Layer(thickness=100e-9, material=material),
                Layer(thickness=200e-9, material=material),
            ],
        )
        rebuilt = Stack(
            superstrate=stack.superstrate,
            substrate=stack.substrate,
            layers=[Layer(thickness=d, material=material) for d in overridden],
        )

        res = stratix.solve(
            stack,
            wavelength,
            0.0,
            Polarization.TE,
            method=Method.DTN,
            thicknesses=nd.array(overridden),
        )
        ref = stratix.solve(
            rebuilt, wavelength, 0.0, Polarization.TE, method=Method.DTN
        )

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-14
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-14
