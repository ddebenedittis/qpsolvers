#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2024 Davide De Benedittis

"""Solver interface for `ReLUQP <https://github.com/RoboticExplorationLab/ReLUQP-py.git>`__

A GPU Accelerated Quadratic Programming Solver for Model-Predictive Control.
"""

from typing import Optional

import numpy as np
from numpy import hstack, vstack
import reluqp.reluqpth as reluqp

from torch import Tensor

from ..conversions import combine_linear_box_inequalities
from ..problem import Problem
from ..solution import Solution


def reluqp_solve_problem(
    problem: Problem,
    initvals: Optional[dict] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using ReLUQP.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector (not used).
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Raises
    ------
    ProblemError :
        If the cost matrix of the quadratic program if not positive definite,
        or if the problem is ill-formed in some way, for instance if some
        matrices are not dense.

    Note
    ----
    

    Notes
    -----
    All other keyword arguments are forwarded to the ReLUQP solver. For
    instance, you can call ``reluqp_solve_qp(P, q, G, h, device=device)``.
    See the solver documentation for details.
    """
    
    P, q, G, h, A, b, lb, ub = problem.unpack()
    n: int = q.shape[0]
    
    if lb is not None or ub is not None:
        qp_C, qp_u, qp_l = combine_linear_box_inequalities(G, h, lb, ub, n, use_csc=False)
    elif G is not None:
        qp_C = G
        qp_u = h
        qp_l = - np.inf * np.ones(h.shape)
    else:
        # The constraints cannot be empty.
        qp_C = np.zeros((1, n))
        qp_u =   np.inf * np.ones(1)
        qp_l = - np.inf * np.ones(1)
        
    if A is not None and b is not None:
        qp_C = vstack([qp_C, A])
        qp_u = hstack([qp_u, b])
        qp_l = hstack([qp_l, b])
    
    solution = Solution(problem)
    
    model = reluqp.ReLU_QP()
    model.setup(
        P, q, qp_C, qp_l, qp_u,
        verbose = verbose,
        **kwargs,
    )
    
    if initvals is not None:
        if not isinstance(initvals, dict):
            raise ValueError("initvals must be a dictionary containing the "
                             "keys 'primal', 'dual', and 'lam'")
        if set(initvals.keys()) != {'primal', 'dual', 'lam'}:
            raise ValueError("initvals must be a dictionary containing the "
                             "keys 'primal', 'dual', and 'lam'")
        for key in initvals.keys():
            if not isinstance(initvals[key], Tensor):
                raise TypeError(f"initvals['{key}'] must be a torch.Tensor")
        
        model.warm_start(
            x = initvals["primal"],
            z = initvals["dual"],
            lam = initvals["lam"],
        )
    
    results = model.solve()
        
    solution.found = results.info.status == 'solved'
    solution.x = results.x.detach().cpu().numpy() if solution.found else None
    solution.obj = results.info.obj_val
    
    n_ie = G.shape[0] if G is not None else 0
    n_eq = A.shape[0] if A is not None else 0
    n_box = lb.shape[0] if lb is not None else ub.shape[0] if ub is not None else 0
    
    solution.z = results.lam[:n_ie].detach().cpu().numpy() if G is not None else np.empty((0,))
    solution.z_box = (
        results.lam[n_ie:n_ie+n_box].detach().cpu().numpy()
        if lb is not None or ub is not None
        else np.empty((0,))
    )
    solution.y = results.lam[-n_eq:].detach().cpu().numpy() if A is not None else np.empty((0,))
    
    solution.extras = {
        "rho": results.info.rho_estimate.detach().cpu().numpy()
    }

    return solution


def reluqp_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using ReLUQP.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
        \underset{\mbox{minimize}}{x} &
            \frac{1}{2} x^T P x + q^T x \\
        \mbox{subject to}
            & G x \leq h                \\
            & A x = b                   \\
            & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `ReLUQP <https://github.com/RoboticExplorationLab/ReLUQP-py.git>`__.

    Parameters
    ----------
    P :
        Positive semidefinite cost matrix.
    q :
        Cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = reluqp_solve_problem(
        problem, initvals, verbose, **kwargs
    )
    return solution.x if solution.found else None
