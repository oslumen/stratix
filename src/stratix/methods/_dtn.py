r"""Dirichlet-to-Neumann (DtN) map method.

Every other method in this package walks the stack, folding one layer at
a time into a running object -- a characteristic matrix, a scattering
matrix, an input admittance.  This one does not walk the stack.  It
writes down one Dirichlet-to-Neumann operator per layer, assembles them
into a single operator on the tangential field values *at the
interfaces*, and solves that system once.  The unknowns are physical
field amplitudes rather than a derived impedance, and the assembly is
the substructuring picture of a multilayer rather than a recursion.

Derivation
----------
Let ``u`` be the tangential field that stays continuous across every
interface: ``E_y`` for TE, ``H_y`` for TM.  Inside a homogeneous layer it
satisfies the Helmholtz equation ``u'' + kz^2 u = 0``, so on a layer of
thickness ``d`` occupying ``0 <= z <= d``::

    u(z) = u(0) cos(kz z) + B sin(kz z)

Write the companion quantity ``v = u' / (i * denom)``, with ``denom = mu``
for TE and ``denom = epsilon`` for TM.  ``v`` is the *other* tangential
field up to one fixed constant (``-H_x`` for TE, ``E_x`` for TM), so it
is continuous across interfaces too, and a single forward-going plane
wave ``exp(i kz z)`` has ``v = Y u`` with ``Y = kz / denom`` the wave
admittance -- the same ``Y`` the Fresnel coefficients are written in.
This is the ``exp(-i omega t)`` convention of ADR 0005, in which the wave
travelling *into* the stack (``+z``) carries ``exp(+i kz z)``.

Eliminating ``B`` between the two faces gives the layer's DtN map --
Dirichlet data ``(u(0), u(d))`` in, Neumann data ``(v(0), v(d))`` out::

    / v(0) \        /  cot(phi)  -csc(phi) \  / u(0) \
    |      |  = iY  |                      |  |      |,   phi = kz d
    \ v(d) /        \  csc(phi)  -cot(phi) /  \ u(d) /

the cot/csc kernel the method is named after.

Assembly
--------
Number the interfaces ``0 .. N`` for ``N`` layers, with ``u_j`` the
tangential field at interface ``j``.  Layer ``j+1`` contributes its
kernel to interfaces ``j`` and ``j+1``.  Requiring ``v`` to be continuous
at each interior interface -- the Neumann data the layer above delivers
must equal the Neumann data the layer below demands -- gives one equation
per interior interface, and each layer touches only its two faces, so the
assembled operator is tridiagonal.  The two ends close the system with
the radiation conditions: in the superstrate ``u_0 = 1 + r`` and
``v_0 = Y_super (1 - r)``, which eliminates ``r`` in favour of ``u_0`` and
leaves the incident wave as the system's only source term; in the
substrate a single outgoing wave gives ``v_N = Y_sub u_N``.

Solving ``A u = b`` -- ``A`` is the stack's discrete Poincare-Steklov
operator -- yields every interface field at once, and ``r = u_0 - 1`` and
``t = u_N`` are read straight off the ends.  There is no second pass to
recover the transmitted amplitude, because the amplitudes were the
unknowns.

Scaling away the Dirichlet resonances
-------------------------------------
``cot`` and ``csc`` are both infinite where ``sin(phi) = 0``: a layer
whose phase thickness is a multiple of pi is at a Dirichlet resonance and
has no DtN map, because its interior problem with prescribed face values
is singular there.  The stack's R and T are perfectly ordinary -- a
half-wave layer is an absentee layer -- so the blow-up belongs to the
representation, not to the physics.

The denominators are cleared exactly.  Each row is multiplied by the
phase factors of the layers it touches, which turns every ``cot`` and
``csc`` into a bounded combination of ``p = exp(i phi)``: the resonance
survives only as the factor ``p**2 - 1`` going to zero, never as an
entry going to infinity.  Writing the scaling on ``p`` rather than on
``sin(phi)`` matters twice over -- ``sin`` and ``cos`` themselves
overflow once a layer is thick enough to absorb, since both grow like
``exp(Im phi) / 2``, and it is the substitution the S-matrix layer loop
already makes.

Two more things keep the scaled system solvable.  Rows touching one
layer and rows touching two carry different powers of that factor, so
near a resonance they differ in magnitude by many orders; they are
equilibrated by their absolute row sum, which cannot change the solution
but does let a factorisation comparing pivots by absolute value see past
the spread.  And the solve goes through ``nd.linalg.solve`` rather than
an elimination straight down the diagonal, which would divide by
whichever pivot the scaling happened to drive to nearly nothing and
return a badly wrong answer for a stack of half-wave layers.

That leaves the resonance itself, where the operator is genuinely rank
deficient: the scaled determinant vanishes with ``sin(phi)``.  This is
the interior resonance of the DtN formulation showing through, not an
artefact of the assembly -- the layer's map does not exist, so there is
nothing to assemble -- and a solve of it raises rather than returning
anything.  An exactly half-wave layer is easy to write down, and a zero
thickness, which is the same resonance at ``phi = 0``, is where an
optimiser goes whenever it wants a layer gone.  Such a layer is
therefore rotated off the resonance by ``sqrt(eps)`` in phase, about a
femtometre of thickness, by :func:`_detune_resonance`.

``sqrt(eps)`` is the offset that balances the two errors either side of
it: rotating by ``delta`` perturbs the answer by order ``delta``, while
declining to rotate leaves a system conditioned like ``1 / delta`` and
an answer good to ``eps / delta``.  They meet at ``delta = sqrt(eps)``,
which caps the error over the whole resonant neighbourhood at about
``1e-8`` -- measured at ``2e-9`` across a kx sweep through one -- and
holds everywhere else to the ``1e-16`` the method otherwise delivers.

The pivoted solve makes DTN cubic in the layer count where the other
methods are linear.  For the stacks this library targets the constant is
small, and it is one more reason ``Method.AUTO`` resolves to the S-matrix
(ADR 0001) -- linear in the layers, with no resonance to work around and
no thickness where it drops to eight digits -- while DTN stays opt-in.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from ._medium_params import _medium_params
from ._util import _interface_coeffs
from ._util import _layer_phases
from ._util import _rel_tol
from ._util import _resolve_thicknesses
from ._util import _safe_R
from ._util import _safe_T
from ._util import _wave_admittances

#: How far a resonant layer is moved off its resonance, as a phase.
#: ``sqrt(eps)`` is the same floor :func:`~stratix.methods._util._rel_tol`
#: draws elsewhere, and it is the largest offset still invisible in the
#: answer: it shifts R by order ``1e-8``, against the ``1e-16`` the method
#: otherwise delivers, and it does so only at the isolated thicknesses
#: that have no answer at all without it.
_DETUNE = _rel_tol


def _detune_resonance(p: nd.ndarray) -> nd.ndarray:
    """Move a layer off an exact Dirichlet resonance.

    At ``p**2 == 1`` -- phase thickness an exact multiple of pi, which
    includes the zero thickness an optimiser reaches when it removes a
    layer -- the assembled operator is rank deficient.  The DtN map of
    such a layer does not exist, so this is the formulation's interior
    resonance rather than anything the assembly can be rearranged to
    avoid, and the system has no solution to find: ``nd.linalg.solve``
    raises on it, taking every other point of the sweep down with the one
    that landed there.

    The layer is therefore rotated by ``sqrt(eps)`` in phase, which is
    what a thickness change of about a femtometre would do.  R and T come
    back correct to roughly ``1e-8`` instead of not coming back at all.
    Away from a resonance the comparison is false and ``p`` is returned
    untouched, so this costs the ordinary case nothing -- neither
    accuracy nor a different value than the unguarded expression.

    Parameters
    ----------
    p : Layer phase factor ``exp(i * kz * d)``.

    Returns
    -------
    ``p``, rotated only where it sits on a resonance.
    """
    squared = p * p
    # ``p**2 - 1`` cancels, so its trustworthy scale is ``1 + |p**2|``.
    resonant = nd.abs(squared - 1.0) <= _DETUNE() * (1.0 + nd.abs(squared))
    return nd.where(resonant, p * nd.exp(1j * _DETUNE()), p)


def _assemble(
    Y_super: nd.ndarray,
    Y_layers: list[nd.ndarray],
    Y_sub: nd.ndarray,
    phis: list[nd.ndarray],
) -> tuple[nd.ndarray, nd.ndarray]:
    """Build the stack's scaled Poincare-Steklov system.

    Row ``j`` states that the tangential ``v`` delivered at interface
    ``j`` by the layer above equals the one demanded by the layer below,
    with the superstrate and substrate radiation conditions standing in
    for the missing layers at the two ends.  Every row is pre-multiplied
    by the phase factors of the layers it touches, so no ``cot`` or
    ``csc`` is ever evaluated.

    The scaling is carried in ``p = exp(i phi)`` rather than in
    ``sin(phi)``, which is the same clearing of the denominators written
    on the decaying exponential instead of on the difference of the two.
    A layer thick enough to absorb has a large ``Im(phi)``, where ``sin``
    and ``cos`` both grow like ``exp(Im phi) / 2`` and overflow to
    infinity well before the layer is optically thick by any practical
    measure; the physical branch of ``kz`` puts ``Im(phi) >= 0``, so
    ``|p| <= 1`` and every entry below is a bounded combination of
    ``p``, ``p**2 - 1`` and ``p**2 + 1``.  This is the same substitution
    the S-matrix layer loop makes, and for the same reason.

    The rows are stacked into a dense matrix rather than kept as three
    diagonals: the sweep axes sit in front of the two operator axes, so
    one assembly covers every wavelength and kx, and ``nd.linalg.solve``
    batches over them.

    Parameters
    ----------
    Y_super, Y_sub : Wave admittance of the semi-infinite media.
    Y_layers : Wave admittance of each layer, top to bottom.
    phis : Phase thickness of each layer, same order.

    Returns
    -------
    A : Operator of shape ``(*sweep, N + 1, N + 1)``.
    b : Right-hand side of shape ``(*sweep, N + 1, 1)``.
    """
    # p = exp(i phi); 2i p sin(phi) = p^2 - 1 and 2 p cos(phi) = p^2 + 1.
    # ``diff`` vanishes exactly at a Dirichlet resonance, where the naive
    # kernel is infinite, so it carries the same zero the sine did.
    p = [_detune_resonance(nd.exp(1j * phi)) for phi in phis]
    diff = [pk * pk - 1.0 for pk in p]
    total = [pk * pk + 1.0 for pk in p]
    n_layers = len(phis)
    n_interfaces = n_layers + 1

    zero = nd.zeros_like(1j * Y_layers[0] * diff[0])
    rows = [[zero] * n_interfaces for _ in range(n_interfaces)]
    sources = [zero] * n_interfaces

    # Interface 0, scaled by 2i p: u_0 = 1 + r and v_0 = Y_super (1 - r)
    # eliminate r, and the incident wave lands here as the only source.
    rows[0][0] = Y_super * diff[0] - Y_layers[0] * total[0]
    rows[0][1] = 2 * p[0] * Y_layers[0]
    sources[0] = 2 * Y_super * diff[0]

    # Interior interfaces, scaled by 4 p_above p_below: two layers meet,
    # and the row carries both their phase factors -- which is what keeps
    # a half-wave layer finite.
    for j in range(1, n_layers):
        above, below = j - 1, j
        rows[j][j - 1] = 2 * p[above] * Y_layers[above] * diff[below]
        rows[j][j] = -(
            Y_layers[above] * total[above] * diff[below]
            + Y_layers[below] * total[below] * diff[above]
        )
        rows[j][j + 1] = 2 * p[below] * Y_layers[below] * diff[above]

    # Interface N, scaled by 2i p: the substrate holds a single outgoing
    # wave, v = Y_sub u.
    last = n_layers - 1
    rows[n_layers][n_layers - 1] = -2 * p[last] * Y_layers[last]
    rows[n_layers][n_layers] = Y_layers[last] * total[last] - Y_sub * diff[last]

    A = nd.stack([nd.stack(row, axis=-1) for row in rows], axis=-2)
    b = nd.stack(sources, axis=-1)[..., None]

    # Equilibrate: divide each row by its absolute row sum.  A row touching
    # two layers carries two sines and a row touching one carries one, so
    # near a resonance the rows differ in magnitude by many orders even
    # though each is individually well formed.  An LU factorisation
    # compares candidate pivots by absolute value and cannot see past that
    # spread; balancing the rows first is what lets it pick the pivot the
    # arithmetic actually wants.  Scaling rows of ``A`` and ``b`` together
    # leaves the solution exactly unchanged, so this is free of any effect
    # on the answer or its gradient.
    scale = nd.sum(nd.abs(A), axis=-1, keepdims=True)
    return A / scale, b / scale


def _dtn_solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    thicknesses: nd.ndarray | None = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    """Compute R and T by assembling and solving the stack's DtN system.

    Parameters
    ----------
    stack : Multilayer stack (superstrate, layers, substrate).
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : ``TE`` or ``TM``.
    thicknesses : Optional 1-D ndarray overriding the stack's layer
        thicknesses.  Must have length ``len(stack.layers)``.  When
        provided, ``stack.layers[i].thickness`` is ignored in favour of
        ``thicknesses[i]``, enabling autodiff w.r.t. thickness.

    Returns
    -------
    R : Power reflectance.
    T : Power transmittance.
    intermediates : Dict carrying the resolved thicknesses.
    """
    _omega, _k0, kzs, _, _, denom_vals, no_flux = _medium_params(
        stack, wavelength, kx, polarization
    )

    resolved = _resolve_thicknesses(stack, thicknesses)
    admittances = _wave_admittances(kzs, denom_vals)
    phis = _layer_phases(kzs, resolved)

    if phis:
        A, b = _assemble(admittances[0], admittances[1:-1], admittances[-1], phis)
        fields = nd.linalg.solve(A, b)[..., 0]
        r_total = fields[..., 0] - 1.0
        t_total = fields[..., -1]
    else:
        # No layers means no DtN operator to assemble: the two faces the
        # kernel relates have collapsed onto one interface, and the single
        # Fresnel interface is what the assembly degenerates to.
        r_total, _, t_total, _ = _interface_coeffs(admittances[0], admittances[-1])

    R = _safe_R(r_total, no_flux)
    T = _safe_T(t_total, kzs[0], denom_vals[0], kzs[-1], denom_vals[-1], no_flux)

    return R, T, {"thicknesses": resolved}
