"""A small deterministic physics-only volume exercises real native tracking."""
import numpy as np
import pytest
from NeedleFinderLib import TrackingAlgorithms as tracking
from NeedleFinderLib.RustCore import is_rust_available

class Volume:
    def __init__(self, data, matrix):self.data=data;self.matrix=matrix
    def GetImageData(self):return self.data
    def GetSpacing(self):return np.linalg.norm(self.matrix[:3,:3],axis=0)
    def GetIJKToRASMatrix(self, target):target._m=self.matrix.copy()
    def GetID(self):return 'synthetic'

class Logic(tracking.TrackingAlgorithmsMixin):
    def findAxialSegmentationLimitFromMarker(self,bForceFallback=False):return 0,0.0


def test_synthetic_two_shafts_invariant_under_axis_permutation(monkeypatch):
    assert is_rust_available(), 'Synthetic integration requires the native core (Rust or C++ fallback)'
    monkeypatch.setattr(tracking,'get_volume_f64_buffer',lambda img:(img,img.shape[::-1]))
    monkeypatch.setattr(tracking,'run_nn_3d_segmentation',lambda *a,**kw:None)
    monkeypatch.setattr(tracking,'run_mpnn_and_graph_transformer',lambda candidates,*a,**kw:candidates)
    data=np.full((64,80,80),200.0)
    yy,xx=np.mgrid[:80,:80]
    for x,y in [(28,32),(51,47)]:
        data[5:55, (xx-x)**2+(yy-y)**2<=2.2**2]=15
    matrix=np.diag([0.8,0.8,1.5,1.0])
    logic=Logic()
    first=logic.autoDetectAllNeedles(Volume(data,matrix),minConfidence=10,kLim=4,renderNeedles=False,script=False)
    assert len(first)==2
    for trajectory in first:
        points=np.asarray(trajectory['ras'])
        center=points[:,:2].mean(axis=0)
        assert min(np.linalg.norm(center-[28*.8,32*.8]),np.linalg.norm(center-[51*.8,47*.8]))<1.0
        assert np.ptp(points[:,2])>50
    swapped=logic.autoDetectAllNeedles(Volume(np.ascontiguousarray(data.swapaxes(0,1)),matrix[:,[0,2,1,3]]),minConfidence=10,kLim=4,renderNeedles=False,script=False)
    assert len(swapped)==len(first)
    for a,b in zip(first,swapped):np.testing.assert_allclose(a['ras'],b['ras'],rtol=0,atol=1e-8)
