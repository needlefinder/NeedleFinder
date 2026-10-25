"""Run in a dedicated Slicer process, not through pytest."""
import json
import os
import sys
import traceback
from pathlib import Path
import numpy as np
import slicer
import vtk

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'NeedleFinder'))
sys.path.insert(0,str(ROOT))
from tools.release import source_fingerprint, sha256

output=Path(os.environ.get('NEEDLEFINDER_EVIDENCE_DIR',ROOT/'reproducibility/runs/evidence'))
output.mkdir(parents=True,exist_ok=True)
report=output/'slicer-report.json'
status='failed'
try:
    from NeedleFinderLib.GeometryUtils import ijk_to_ras,ras_to_ijk,make_fast_scalar_getter
    from NeedleFinderLib.RustCore import is_rust_available
    node=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    values=np.arange(12*16*20,dtype=np.float64).reshape(12,16,20)
    slicer.util.updateVolumeFromArray(node,values)
    matrix=vtk.vtkMatrix4x4()
    transform=np.array([[-0.7,0,0,10],[0,0,2.4,-20],[0,1.2,0,30],[0,0,0,1]])
    for r in range(4):
        for c in range(4):matrix.SetElement(r,c,transform[r,c])
    node.SetIJKToRASMatrix(matrix)
    point=[4.5,8.25,2]
    np.testing.assert_allclose(ijk_to_ras(point,node),(transform@[*point,1])[:3])
    np.testing.assert_allclose(ras_to_ijk(ijk_to_ras(point,node),node),point)
    sample=make_fast_scalar_getter(node.GetImageData())
    assert sample(3,2,1)==values[1,2,3]
    assert sample(-1,2,1)==0
    assert is_rust_available(), 'Native library did not load or build'
    slicer.mrmlScene.RemoveNode(node)
    status='passed'
    result={'status':status,'checks':['real MRML RAS/IJK transforms','VTK buffer order and boundaries','native library availability'],
            'slicer_version':slicer.app.applicationVersion}
except Exception:
    result={'status':'failed','traceback':traceback.format_exc()}
    traceback.print_exc()
report.write_text(json.dumps(result,indent=2)+'\n')
(output/'slicer.json').write_text(json.dumps(dict(status=status,source_fingerprint=source_fingerprint(),skipped=0,
                                    report=report.name,report_sha256=sha256(report)),indent=2)+'\n')
slicer.app.exit(0 if status=='passed' else 1)
