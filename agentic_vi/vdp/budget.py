"""Trusted target boundary. The probe receives only scoped primitive methods."""
from collections import Counter
from threading import Lock

import numpy as np


class Denied(Exception):
    pass


class Budget:
    def __init__(self, limit, dimension, backend, allocations):
        if type(limit) is not int or limit < 0 or dimension < 1:
            raise ValueError("invalid budget or dimension")
        if any(type(v) is not int or v < 0 for v in allocations.values()) or sum(allocations.values()) > limit:
            raise ValueError("invalid stage allocations")
        self.limit = limit
        self.allocations = dict(allocations)
        self._backend = backend
        self.dimension = dimension
        self._lock = Lock()
        self._ledger = {key: Counter(attempts=0, completed=0, failed=0, denied=0,
                                     f=0, g=0, h=0, used=0) for key in allocations}

    def scope(self, key):
        if key not in self.allocations:
            raise KeyError(key)
        return Primitives(lambda order, x, v: self._call(key, order, x, v))

    def _call(self, key, order, x, v):
        x = np.asarray(x)
        if x.ndim not in (1, 2) or x.shape[-1] != self.dimension or not x.size:
            raise ValueError("expected a point or nonempty batch")
        if x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError("expected finite float64 points")
        if v is not None and (v.shape != x.shape or v.dtype != x.dtype or not np.isfinite(v).all()):
            raise ValueError("invalid directions")
        n = x.size // self.dimension
        charge = n * order
        row = self._ledger[key]
        with self._lock:
            row['attempts'] += 1
            if charge > self.allocations[key] - row['used']:
                row['denied'] += 1
                raise Denied(key)
            row['used'] += charge
            for name in ('f', 'g', 'h')[:order]:
                row[name] += n
        try:
            result = self._backend(order, x, v)
            shapes = (x.shape[:-1], x.shape, x.shape)[:order]
            if len(result) != order:
                raise ValueError("wrong output count")
            for i, (a, shape) in enumerate(zip(result, shapes)):
                if np.shape(a) != shape or np.asarray(a).dtype != np.float64:
                    raise ValueError("invalid primitive output")
                valid = np.isfinite(a)
                if order == 1:
                    valid |= np.isneginf(a)
                if not np.all(valid):
                    raise FloatingPointError("invalid target value or derivative")
        except Exception:
            with self._lock:
                row['failed'] += 1
            raise
        with self._lock:
            row['completed'] += 1
        return result

    def report(self):
        with self._lock:
            total = Counter(dict.fromkeys(('used','f','g','h','attempts','completed','failed','denied'), 0))
            for row in self._ledger.values():
                total.update(row)
            return dict(B=self.limit, remaining=self.limit-total['used'], **total,
                        charge_schedule=[1, 2, 3], allocations=self.allocations.copy(),
                        ledger={k: dict(v) for k, v in self._ledger.items()})


class Primitives:
    __slots__ = ('__call_backend',)

    def __init__(self, call):
        self.__call_backend = call

    def value(self, x):
        return self.__call_backend(1, x, None)[0]

    def value_grad(self, x):
        return self.__call_backend(2, x, None)

    def value_grad_hvp(self, x, direction):
        """Third output is the Hessian of negative log density times direction."""
        return self.__call_backend(3, x, direction)
