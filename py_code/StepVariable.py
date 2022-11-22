from typing import TypeVar
Self = TypeVar('Self')

class StepVariable():
    def __init__(self, f:float, _gamma:float, _step:int = 1, _max:float = float('inf'), _min:float = -float('inf')):
        self.initValue = float(f)
        self.f = self.initValue
        self.stepCount = 0
        self._step = _step
        self._gamma = float(_gamma)
        self._max = float(_max)
        self._min = float(_min)
        
    def step(self):
        self.stepCount += 1
        if self.stepCount % self._step == 0 and self.f < self._max and self.f > self._min:
            self.f *= self._gamma
            self.f = min(max(self.f, self._min), self._max)

    def reset(self):
        self.stepCount = 0
        self.f = self.initValue

    def __repr__(self): return str(self.f)
    def __add__(self, __x: float) -> float: return self.f + __x
    def __sub__(self, __x: float) -> float: return self.f - __x
    def __mul__(self, __x: float) -> float: return self.f * __x
    def __truediv__(self, __x: float) -> float: return self.f / __x
    def __floordiv__(self, __x: float) -> float: return self.f // __x
    def __mod__(self, __x: float) -> float: return self.f % __x
    
    def __eq__(self, __x: object) -> bool: return self.f == __x
    def __ne__(self, __x: object) -> bool: return self.f != __x
    def __lt__(self, __x: float) -> bool: return self.f < __x
    def __le__(self, __x: float) -> bool: return self.f <= __x
    def __gt__(self, __x: float) -> bool: return self.f > __x
    def __ge__(self, __x: float) -> bool: return self.f >= __x

    def __iadd__(self, __x: float) -> Self:
        self.f += __x
        return self
    def __isub__(self, __x: float) -> Self:
        self.f -= __x
        return self
    def __imul__(self, __x: float) -> Self:
        self.f *= __x
        return self
    def __itruediv__(self, __x: float) -> Self:
        self.f /= __x
        return self
    def __imod__(self, __x: float) -> Self:
        self.f %= __x
        return self
    def __float__(self) -> float: return self.f
    def __int__(self) -> int: return int(self.f)