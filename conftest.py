"""Minimal Slicer boundary doubles; real Slicer tests run in a separate process."""
import sys
import types
import numpy as np

for name in ('vtk', 'vtk.util', 'vtk.util.numpy_support', 'slicer', 'slicer.util',
             'qt', 'ctk', 'SimpleITK', 'sitkUtils', 'slicer.ScriptedLoadableModule'):
    sys.modules.setdefault(name, types.ModuleType(name))

class Matrix:
    def __init__(self):
        self._m = np.eye(4)
    def GetElement(self, r, c):
        return float(self._m[r, c])
    def MultiplyDoublePoint(self, p):
        return self._m @ p
    def Invert(self):
        self._m = np.linalg.inv(self._m)

sys.modules['vtk'].vtkMatrix4x4 = Matrix
sys.modules['slicer'].modules = types.SimpleNamespace(NeedleFinderWidget=None)
