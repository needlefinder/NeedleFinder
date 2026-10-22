"""Needle detection, tip finding, mechanical deflection models, and curve builders."""

import math
import os
import random
import time
import numpy as np
import vtk
import qt
import ctk
import slicer

from .DeflectionModelData import (
    MAT_ARC_LEN_MM as matArcLen_mm,
    MAT_END_SEG_ANGLES_RAD as matEndSegAngles_rad,
    MAT_FS_MN as matFs_mN,
    MAT_Y_DEFL_MM as matYDefl_mm,
)
from .GeometryUtils import (
    build_cspline_polydata,
    build_polyline_polydata,
    build_tube_polydata,
    compute_bezier_curve,
    extract_model_id_number,
    make_fast_scalar_getter,
    trilinear_interp,
    unique_points,
)
from .RustCore import (
    _DOUBLE_PTR,
    _FLOAT_PTR,
    get_volume_f64_buffer,
    rust_compute_3d_sym_response,
    rust_find_2d_nms_peaks,
    rust_find_tip,
    rust_search_cone_13,
    rust_search_cone_15_1,
    rust_search_cone_current_dev,
    rust_trace_seed_3d,
    rust_trim_trajectory_inside_out,
)
from .NeuralNeedleNet import (
    create_or_update_slicer_labelmap,
    extract_nn_labelmap_tubes,
    run_mpnn_and_graph_transformer,
    run_nn_3d_segmentation,
)
from .LegacyEditorCompat import (
    EditorLib,
    create_fiducial_node,
    get_all_fiducial_nodes,
    get_fiducial_coords,
    set_fiducial_coords,
)
from .HybridNeedleNetWeights import score_trajectory_hybrid_nn
from .ProfilingUtils import (
    MAXCOL,
    MAXNEEDLES,
    assertbox,
    breakbox,
    cmp,
    conesColor,
    frequent,
    getClassName,
    getNode,
    lineno,
    loggingInfos,
    logprint,
    msgbox,
    outlierThresh_mm,
    pause,
    profbox,
    profiling,
    profprint,
    whoami,
    whosdaddy,
    whosgranny,
)


def _ensure_scalar_getter(img_or_getter):
  """Return a fast scalar getter callable from either a vtkImageData or an existing getter."""
  if callable(img_or_getter):
    return img_or_getter
  return make_fast_scalar_getter(img_or_getter)


def _get_ijk_to_ras_16():
  """Return the 16-element row-major IJK-to-RAS matrix of the active volume node."""
  try:
    vol_node = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
  except Exception:
    vol_node = None
  if vol_node is None:
    return None
  m = vtk.vtkMatrix4x4()
  vol_node.GetIJKToRASMatrix(m)
  return [m.GetElement(r, c) for r in range(4) for c in range(4)]


class TrackingAlgorithmsMixin:
  """Mixin providing needle tracking, tip optimization, rationality checks, and 3D curve building."""

  def needleDetection(self):
    """
    This solution is optional but not used anymore in the workflow.
    Use the label map of the needle tips
    Apply the island effect
    Extract the coordinates of the islands (self.array2)
    Start a detection for each island (self.needleDetectionThread)
    """
    profbox(whoami())
    editUtil = EditorLib.EditUtil.EditUtil()
    parameterNode = editUtil.getParameterNode()
    sliceLogic = editUtil.getSliceLogic()
    islandsEffect = EditorLib.IdentifyIslandsEffectOptions()
    islandsEffect.setMRMLDefaults()
    islandsEffect.__del__()
    islandTool = EditorLib.IdentifyIslandsEffectLogic(sliceLogic)
    parameterNode.SetParameter("IslandEffect,minimumSize", '0')
    islandTool.removeIslands()
    m = vtk.vtkMatrix4x4()
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    volumeNode.GetIJKToRASMatrix(m)
    imageData = volumeNode.GetImageData()
    spacing = volumeNode.GetSpacing()
    self.t0 = time.perf_counter()
    label = self.array2()
    for I in range(len(label)):
      A = label[I]
      colorVar = I
      self.needleDetectionThread(A, imageData, colorVar, spacing)
      if slicer.modules.NeedleFinderWidget.autoStopTip.isChecked():
        self.needleDetectionUPThread(A, imageData, colorVar, spacing)

  def findNextStepNumber(self, needleNumber):
    stepValue = 0
    for modelNode in get_all_fiducial_nodes():
      if modelNode.GetAttribute("ValidationNeedle") == "1" and int(modelNode.GetAttribute("NeedleNumber")) == needleNumber:
        if int(modelNode.GetAttribute("NeedleStep")) > stepValue:
          stepValue = int(modelNode.GetAttribute("NeedleStep"))
    return stepValue + 1

  def lockControlPoints(self, needleNumber):
    """Lock every control point except the ones belonging to needleNumber."""
    profprint()
    for modelNode in get_all_fiducial_nodes():
      if modelNode.GetAttribute("ValidationNeedle") == "1" and int(modelNode.GetAttribute("NeedleNumber")) != needleNumber:
        modelNode.SetLocked(1)

  def unlockControlPoints(self, needleNumber):
    """Unlock control points from this needle."""
    profprint()
    for modelNode in get_all_fiducial_nodes():
      if modelNode.GetAttribute("ValidationNeedle") == "1" and int(modelNode.GetAttribute("NeedleNumber")) == needleNumber:
        modelNode.SetLocked(0)

  def deleteDuplicateMarkers(self):
    """Remove duplicate legacy annotation hierarchy markers if present."""
    profprint()
    scene = slicer.mrmlScene
    nodes = scene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode', 'Fiducials List')
    if nodes and nodes.GetNumberOfItems() > 1:
      for i in range(1, nodes.GetNumberOfItems()):
        hnode = nodes.GetItemAsObject(i)
        if hnode:
          children = vtk.vtkCollection()
          hnode.GetAssociatedChildrenNodes(children, "vtkMRMLAnnotationFiducialNode")
          for j in range(children.GetNumberOfItems()):
            annotNode = children.GetItemAsObject(j)
            oneToOneHierarchyNode = hnode.GetAssociatedHierarchyNode(annotNode.GetScene(), annotNode.GetID())
            pointDisplayNode = annotNode.GetAnnotationPointDisplayNode()
            textDisplayNode = annotNode.GetAnnotationTextDisplayNode()
            storageNode = annotNode.GetStorageNode()
            for aux_node in (oneToOneHierarchyNode, pointDisplayNode, textDisplayNode, storageNode, annotNode):
              if aux_node:
                try:
                  scene.RemoveNode(aux_node)
                except Exception:
                  pass

    for list_name in ('Fiducials List', 'All Annotations'):
      nodes = scene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode', list_name)
      if nodes and nodes.GetNumberOfItems() > 1:
        for i in range(1, nodes.GetNumberOfItems()):
          try:
            scene.RemoveNode(nodes.GetItemAsObject(i))
          except Exception:
            pass

  def deleteDuplicateNeedles(self):
    """Remove duplicate needles by filtering names."""
    profprint()
    nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    for i in range(nodes.GetNumberOfItems()):
      node = nodes.GetItemAsObject(i)
      if node is None:
        continue
      name = node.GetName()
      if len(name.split('manual-seg')) > 1 and len(name.split('_')) > 2:
        slicer.mrmlScene.RemoveNode(node)

  def placeNeedleShaftEvalMarker(self, A, needleNumber, stepValue, type='ijk', createPoint=1):
    """
    Add a fiducial point to the vtkMRMLScene where the mouse click was triggered.
    Represents a control point for a manually segmented needle (validation needle).
    """
    profprint()
    pointName = '.' + str(needleNumber) + "-" + str(stepValue)
    pointName2 = '.' + str(needleNumber) + "-" + str(stepValue) + '\r'
    if getNode(pointName) is None and getNode(pointName2) is None and createPoint == 1:
      print('creating fiducial point...')
      ras_coords = self.ijk2ras(A) if type == 'ijk' else A
      nth = int(needleNumber) % MAXCOL
      create_fiducial_node(
          name=pointName,
          ras=ras_coords,
          color=self.color[nth],
          glyph_scale=2.0,
          text_scale=4.0,
          attributes={
              'ValidationNeedle': '1',
              'NeedleNumber': str(needleNumber),
              'NeedleStep': str(stepValue),
          },
      )

    self.addPointToTable(pointName, needleNumber, stepValue, None, sortTable=createPoint)

  def removeNeedleShaftEvalMarker(self, listArgs):
    """Remove a control point from the table and from the scene."""
    profprint()
    print(listArgs)
    ID = listArgs[0]
    pointToRemove = getNode(ID)
    if pointToRemove is not None:
      print(pointToRemove.GetID())
      slicer.mrmlScene.RemoveNode(pointToRemove)
    self.drawValidationNeedles()
    self.deletePointFromTable(ID)
    return 0

  def obturatorNeedle(self, A, imageData, colorVar, spacing):
    """Use the mouse click coordinates to draw obturator needles."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    if widget.obtuNeedle <= 1:
      needleNumber = 0
    else:
      needleNumber = widget.obtuNeedle - 1
      needleStep = 0
    if widget.obtuNeedle == 1:
      needleStep = 1
    elif widget.obtuNeedle == 0:
      needleStep = 0

    nth = int(needleNumber) % MAXCOL
    ras_coords = self.ijk2ras(A)
    create_fiducial_node(
        name='.obtu-' + str(needleNumber),
        ras=ras_coords,
        color=self.color[nth],
        glyph_scale=2.0,
        text_scale=4.0,
        attributes={
            'ObturatorNeedle': '1',
            'NeedleNumber': str(needleNumber),
            'NeedleStep': str(needleStep),
        },
    )

    widget.obtuNeedleValueCtrPt[needleNumber][needleStep] = ras_coords

    if needleNumber >= 1:
      Vx = widget.obtuNeedleValueCtrPt[0][1][0] - widget.obtuNeedleValueCtrPt[0][0][0]
      Vy = widget.obtuNeedleValueCtrPt[0][1][1] - widget.obtuNeedleValueCtrPt[0][0][1]
      Vz = widget.obtuNeedleValueCtrPt[0][1][2] - widget.obtuNeedleValueCtrPt[0][0][2]

      L = float(Vx ** 2 + Vy ** 2 + Vz ** 2) ** 0.5

      E = ras_coords
      Ex = E[0] + 100 * Vx / L
      Ey = E[1] + 100 * Vy / L
      Ez = E[2] + 100 * Vz / L
      widget.obtuNeedleValueCtrPt[needleNumber][1] = [Ex, Ey, Ez]

    self.drawObturatorNeedles()
    widget.stop()
    widget.fiducialObturatorButton.checked = 1
    widget.start(3)

  def objectiveFunction(self, imageData, ijk, radiusNeedleParameter, spacing, gradientPonderation):
    """Objective function used by needleDetectionUPThread."""
    if frequent:
      profprint()
    get_scalar = _ensure_scalar_getter(imageData)
    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))
    ix = int(round(ijk[0]))
    iy = int(round(ijk[1]))
    iz = int(round(ijk[2]))
    ijk[0], ijk[1], ijk[2] = ix, iy, iz
    center = (
        get_scalar(ix, iy, iz)
        + get_scalar(ix + 1, iy, iz)
        + get_scalar(ix, iy + 1, iz)
        + get_scalar(ix - 1, iy, iz)
        + get_scalar(ix, iy - 1, iz)
    )

    if gradientPonderation != 0:
      g1 = get_scalar(ix + radiusNeedle, iy, iz)
      g2 = get_scalar(ix - radiusNeedle, iy, iz)
      g3 = get_scalar(ix, iy + radiusNeedle, iz)
      g4 = get_scalar(ix, iy - radiusNeedle, iz)
      g5 = get_scalar(ix + radiusNeedleCorner, iy + radiusNeedleCorner, iz)
      g6 = get_scalar(ix - radiusNeedleCorner, iy - radiusNeedleCorner, iz)
      g7 = get_scalar(ix - radiusNeedleCorner, iy + radiusNeedleCorner, iz)
      g8 = get_scalar(ix + radiusNeedleCorner, iy - radiusNeedleCorner, iz)
      total = center / 5.0 - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * gradientPonderation
    else:
      total = center / 5.0

    return total

  def objectiveFunctionLOG(self, imageData, ijk, radiusNeedleParameter, spacing, gradientPonderation):
    """Logarithmic objective function (experimental)."""
    print("objectiveFunctionLOG")
    msgbox(whoami())
    get_scalar = _ensure_scalar_getter(imageData)
    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))
    ix, iy, iz = int(ijk[0]), int(ijk[1]), int(ijk[2])

    center = (
        get_scalar(ix, iy, iz)
        + get_scalar(ix + 1, iy, iz)
        + get_scalar(ix, iy + 1, iz)
        + get_scalar(ix - 1, iy, iz)
        + get_scalar(ix, iy - 1, iz)
    )

    if gradientPonderation != 0:
      g1 = get_scalar(ix + radiusNeedle, iy, iz)
      g2 = get_scalar(ix - radiusNeedle, iy, iz)
      g3 = get_scalar(ix, iy + radiusNeedle, iz)
      g4 = get_scalar(ix, iy - radiusNeedle, iz)
      g5 = get_scalar(ix + radiusNeedleCorner, iy + radiusNeedleCorner, iz)
      g6 = get_scalar(ix - radiusNeedleCorner, iy - radiusNeedleCorner, iz)
      g7 = get_scalar(ix - radiusNeedleCorner, iy + radiusNeedleCorner, iz)
      g8 = get_scalar(ix + radiusNeedleCorner, iy - radiusNeedleCorner, iz)
      total = center / 5.0 - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0)
    else:
      total = center / 5.0

    return math.log(total / 30.0 + 0.1)

  def findTip(self, A, imageData, radiusNeedle, coeff, sigmaValue, gradientPonderation, X, Y, Z):
    """Find optimal needle tip in a neighborhood around the initial click A."""
    profprint()
    radiusNeedle = int(radiusNeedle)
    A = [int(A[0]), int(A[1]), int(A[2])]
    print(A)
    X = int(X)
    Y = int(Y)
    Z = int(Z)
    rust_res = rust_find_tip(imageData, A, radiusNeedle, float(coeff), float(sigmaValue), float(gradientPonderation), X, Y)
    if rust_res is not None:
      return rust_res

    get_scalar = _ensure_scalar_getter(imageData)
    minTotalTip = 0
    IBest, JBest, KBest = A[0], A[1], A[2]
    for I in range(-X, X):
      i = int(I / float(coeff))
      for J in range(-Y, Y):
        j = int(J / float(coeff))
        for k in range(1):
          totalTip = 0
          ax = A[0] + i
          ay = A[1] + j
          for l in range(-3, 1):
            az = A[2] + k + l
            v0 = 8 * get_scalar(ax, ay, az)
            v1 = get_scalar(ax + radiusNeedle, ay, az)
            v2 = get_scalar(ax - radiusNeedle, ay, az)
            v3 = get_scalar(ax, ay + radiusNeedle, az)
            v4 = get_scalar(ax, ay - radiusNeedle, az)
            totalTip += v0 - ((v1 + v2 + v3 + v4) / 4.0) * gradientPonderation

          rgauss = (i ** 2 + j ** 2 + k ** 2) ** 0.5
          gaussianAttenuation = math.exp(-(rgauss / 10.0) ** 2 / float(2 * (sigmaValue / 10.0) ** 2))
          totalTip = gaussianAttenuation * totalTip

          if totalTip < minTotalTip or minTotalTip == 0:
            minTotalTip = totalTip
            IBest = ax
            JBest = ay
            KBest = A[2] + k

    return [IBest, JBest, KBest]

  def findAxialSegmentationLimitFromMarker(self, bForceFallback=False):
    """Find the limit marker in the scene and return its z-coord."""
    profprint()
    asl = [0.0, 0.0, -float("inf")]
    coord = [0.0, 0.0, 0.0]
    nodes = slicer.util.getNodes('template slice position*')
    found = False
    node = None
    for node in list(nodes.values()):
      get_fiducial_coords(node, coord)
      aslNew = list(coord)
      if aslNew[2] > asl[2]:
        asl = aslNew
        if found:
          print("higher limit marker found in scene, z-limit [ras]: ", coord[2])
        else:
          print("first limit marker found in scene, z-limit [ras]: ", coord[2])
      if found:
        print("/!\\ there should be only one axial limit marker!")
      found = True
    if not found or bForceFallback:
      print("/!\\ z-limit marker in scene required! --> fallback")
      bases, names = self.returnBasesFromNeedleModels()
      auto_attr = {}
      if len(bases) > 0:
        bases2 = np.array(bases)[:, 2]
        i = bases2.argmin()
        asl = self.ijk2ras(bases[i])
        asl = [coord[0], coord[1], asl[2]]
      else:
        asl = [0.0, 0.0, 0.0]
        try:
          vnode = (
              slicer.app.layoutManager()
              .sliceWidget("Red")
              .sliceLogic()
              .GetBackgroundLayer()
              .GetVolumeNode()
          )
          if vnode is None:
            vnode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
          if vnode is not None and vnode.GetImageData() is not None:
            vol_buf, _ = get_volume_f64_buffer(vnode.GetImageData())
            sp_z = float(vnode.GetSpacing()[2])
            k_est = self.estimateAxialLimitSlice(vol_buf, sp_z)
            asl = list(self.ijk2ras([0.0, 0.0, float(k_est)], vnode))
            auto_attr = {"AutoEstimated": "1"}
        except Exception:
          asl = [0.0, 0.0, 0.0]
      if not node:
        self.fiducialNode = create_fiducial_node(
            name='template slice position',
            ras=asl,
            color=(0.0, 1.0, 0.0),
            visibility=True,
            attributes=auto_attr if auto_attr else None,
        )
      else:
        set_fiducial_coords(node, asl)
    if asl[2] == -float("inf"):
      asl = [0.0, 0.0, 0.0]
    k_out = int(round(self.ras2ijk(asl)[2]))
    widget = getattr(slicer.modules, "NeedleFinderWidget", None)
    if widget is not None:
      widget.axialSegmentationLimit = k_out
      widget.axialSegmentationLimitRAS = float(asl[2])
    return k_out, asl[2]

  def interp3(self, V, x, y, z, interpswitch):
    """Trilinear or nearest-neighbor interpolation in 3D volume V."""
    return trilinear_interp(V, x, y, z, interp_switch=interpswitch)
      
  def needleDetectionThread(self, tips = [], imageData = None, spacing=[1,1,1], script=False, names=""):
    """ Ruibin: comment...
    Switches between the versions of the algorithm. For comparison tests.
    """
    # productive #onbutton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.axialSegmentationLimit, widget.axialSegmentationLimitRAS = self.findAxialSegmentationLimitFromMarker()
    if widget.labelMapNode:
      labelData=widget.labelMapNode.GetImageData()
    else:
      labelData=None
    # select algo version
    NeedleIndex0=0
    if not script:
      NeedleIndex0 = len(self.returnTipsFromNeedleModels('Detection')[0])+1
    if widget.algoVersParameter.value in range(0,4):
      #print 'tips', tips
      for NeedleIndex in range(len(tips)):
          A = tips[NeedleIndex]
          #print 'tip: ',A
          colorVar = NeedleIndex0+NeedleIndex
          if names == "":
            strName = 'auto-seg_%d' % colorVar
          else:
            strName = names[NeedleIndex]
          print('strName: ',strName)
          if widget.algoVersParameter.value == 0:
              self.needleDetectionThreadCurrentDev(A, imageData, colorVar, spacing, script, labelData, manualName=strName)
          elif widget.algoVersParameter.value == 1:
              self.needleDetectionThread13_1(A, imageData, colorVar, spacing, script, labelData, manName=strName,bDrawNeedle=True)
          elif widget.algoVersParameter.value == 2:
              self.needleDetectionThread13_2(A, imageData, colorVar, spacing, script, labelData,manualName=strName)
          elif widget.algoVersParameter.value == 3:
              self.needleDetectionThread15_1(A, imageData, labelData, widget.tempPointList, colorVar, spacing, bUp=False, bScript=script, strManualName=strName,bDrawNeedle=True)
          NeedleIndex+=1

    if widget.algoVersParameter.value == 4: # 1 with Ruibins post-processing
        widget.algoVersParameter.value = 1
        self.needleDetectionThread_RM(tips, imageData, labelData, widget.tempPointList, spacing, bUp=False, bScript=script,  names=names)
        widget.algoVersParameter.value = 4
    elif widget.algoVersParameter.value == 5: # 3 with Ruibins post-processing
        widget.algoVersParameter.value = 3
        self.needleDetectionThread_RM(tips, imageData, labelData, widget.tempPointList, spacing, bUp=False, bScript=script,  names=names)
        widget.algoVersParameter.value = 5
    if not widget.algoVersParameter.value in range(0,6):
      msgbox ("/!\\ needleDetectionThread %d not defined"%widget.algoVersParameter.value)

  def needleDetectionThread_RM(self, tips, imgData, imgLabelData, lrasTempPoints, fvSpacing, bUp=False, bScript=False, names="", tipOnly=False):
    print("RM15____________________________Thread")
    profprint()
    t0 = time.perf_counter()
    bDraw=True
    #msgbox("Detour: \!/ This site is under heavy construction. /!\ ")
    widget = slicer.modules.NeedleFinderWidget
    print("\n"*10) #<<< fast forward in console to create visual break
    # ## load parameters from GUI
    iRadiusMax_mm = widget.radiusMax.value
    iGradientPonderation = widget.gradientPonderation.value
    iSigmaValue = widget.sigmaValue.value
    bGaussianAttenuation = widget.gaussianAttenuationButton.isChecked()
    bGradient = widget.gradient.isChecked()
    nPointsPerNeedle = widget.numberOfPointsPerNeedle.value
    nRotatingIts = widget.nbRotatingIterations.value
    iRadiusNeedle_mm = widget.radiusNeedleParameter.value
    iAxialSegmentationLimit = widget.axialSegmentationLimit
    bAutoStopTip = widget.autoStopTip.isChecked()
    global conesColor
    if conesColor: conesColor=(conesColor+1)%308;
    if conesColor==0: conesColor=300
    print("conesColor= ", conesColor)
    ### initialisation of some parameters
    fModelNeedleLength_mm=187
    fModelSegmentLength_mm=fModelNeedleLength_mm/20.
    fK=2*np.pi*2050/(1000.) # 18 gauge brachy needle
    
    ControlPointsPackage=[]
    ControlPointsPackageIJK=[]
    NumberOfNeedles = len(tips)
    
    for NeedleIndex in range(NumberOfNeedles):
        print("The ",NeedleIndex+1," Needle")
        ijkA = tips[NeedleIndex]
        strManualName = names[NeedleIndex]
        colorVar=NeedleIndex

        #call single needle detection >>>>>>>AM>>>>>>>>>
        if widget.algoVersParameter.value == 1:
          lvControlPointsRAS,lvControlPointsIJK=self.needleDetectionThread13_1(ijkA, imgData, colorVar, fvSpacing, bScript, imgLabelData, strManualName, bDrawNeedle=False)
        elif widget.algoVersParameter.value == 3:
          lvControlPointsRAS,lvControlPointsIJK=self.needleDetectionThread15_1( ijkA, imgData, imgLabelData, lrasTempPoints,0, fvSpacing, bUp=False, bScript=False, strManualName=strManualName, tipOnly=False, bDrawNeedle=False)
        else:
          msgbox ("/!\\ single needleDetectionThread %d not defined"%widget.algoVersParameter.value)
        # end call single needle detection <<<<<<AM<<<<<<<<<
        
        ControlPointsPackage.append(lvControlPointsRAS)
        ControlPointsPackageIJK.append(lvControlPointsIJK)
        print("The", NeedleIndex+1,"needle finished")
    OriginalControlPointsPackage = []
    OriginalControlPointsPackageIJK = []
    for i in range(len(ControlPointsPackage)):
        OriginalControlPointsPackage.append(ControlPointsPackage[i])
        OriginalControlPointsPackageIJK.append(ControlPointsPackageIJK[i])

    OriginalWrongPositions = []
     
    NumberOfFeedbacks = 2
    LastWrongPositionsToSend = []
    for feedbackindex in range(NumberOfFeedbacks):
        if feedbackindex == 0:
            WrongPosition,WrongReason,GlobalDirection = self.needleRationalityCheck(ControlPointsPackageIJK, ControlPointsPackage, imgData, fvSpacing)
            NumberOfWrongNeedles = len(WrongPosition)
            Counting = [0]*NumberOfWrongNeedles
            WrongPositionsToSend = []
            for i in range(NumberOfWrongNeedles):
                if Counting[i]==0:
                    wrongpositiontosend = []
                    wrongpositiontosend.append(WrongPosition[i][0])
                    wrongpositiontosend.append(WrongPosition[i][1])
                    for k in range(NumberOfWrongNeedles):
                        if k!=i and WrongPosition[k][0] == WrongPosition[i][0]:
                            wrongpositiontosend.append(WrongPosition[k][1])
                            Counting[k]=1
                    WrongPositionsToSend.append(wrongpositiontosend)
            NumberOfWrongPositionsToSend = len(WrongPositionsToSend)
            print("The",feedbackindex+1," Loop")
            for i in range(NumberOfWrongPositionsToSend):
                Punish = 0.85
                
                #call Ruibins needle detection feedback >>>>>>>AM>>>>>>>>>
                if widget.algoVersParameter.value == 1:
                  colorVar=NeedleIndex
                  CorrectedControlPoints, CorrectedControlPointsIJK = self.needleDetectionThread13_1(ijkA, imgData, colorVar, fvSpacing, bScript, imgLabelData, strManualName, bDrawNeedle=False,whetherfeedback=True,ControlPointsPackage=ControlPointsPackage,ControlPointsPackageIJK=ControlPointsPackageIJK,WrongPosition=WrongPositionsToSend[i],GlobalDirection=GlobalDirection,Punish=Punish)
                elif widget.algoVersParameter.value == 3:
                  CorrectedControlPoints, CorrectedControlPointsIJK = self.needleDetectionThread15_1( ijkA, imgData, imgLabelData, lrasTempPoints,0, fvSpacing, bUp=False, bScript=False, strManualName=strManualName, tipOnly=False, bDrawNeedle=False,whetherfeedback=True,ControlPointsPackage=ControlPointsPackage,ControlPointsPackageIJK=ControlPointsPackageIJK,WrongPosition=WrongPositionsToSend[i],GlobalDirection=GlobalDirection,Punish=Punish)
                else:
                  msgbox ("/!\\ single needleDetectionThread %d not defined"%widget.algoVersParameter.value)
                #end call Ruibins needle detection feedback <<<<<<AM<<<<<<<<<
                
                #CorrectedControlPoints, CorrectedControlPointsIJK = self.needleDetectionThread15_RM_Feedback(ControlPointsPackage, ControlPointsPackageIJK, WrongPositionsToSend[i], imgData, fvSpacing, imgLabelData, GlobalDirection, Punish,bScript,bUp)
                ControlPointsPackage[WrongPositionsToSend[i][0]] = CorrectedControlPoints
                ControlPointsPackageIJK[WrongPositionsToSend[i][0]] = CorrectedControlPointsIJK
                print("The ",WrongPositionsToSend[i],"(+1 each) Needle corrected")
            LastWrongPositionsToSend = WrongPositionsToSend
            OriginalWrongPositions = WrongPositionsToSend
        else:
            WrongPosition,WrongReason,GlobalDirection = self.needleRationalityCheck(ControlPointsPackageIJK, ControlPointsPackage, imgData, fvSpacing)
            NumberOfWrongNeedles = len(WrongPosition)
            if NumberOfWrongNeedles == 0:
                print("Cannot detect wrong needles")
            else:
                NumberOfLastWrongNeedles = len(LastWrongPositionsToSend)
                
                ## this needle is wrong in last diagnosis and wrong now again, so there should be great probability that it is still wrong. This is risky
                for i in range(NumberOfWrongNeedles):
                    for k in range(NumberOfLastWrongNeedles):
                        if WrongPosition[i][1]==LastWrongPositionsToSend[k][0]:
                            temp = WrongPosition[i][0]
                            WrongPosition[i][0]=WrongPosition[i][1]
                            WrongPosition[i][1]=temp
                            WrongReason[i] = 10 
                #
                
                Counting = [0]*NumberOfWrongNeedles
                WrongPositionsToSend = []
                Punish = 0.85
                
                for i in range(NumberOfWrongNeedles):
                    if Counting[i]==0:
                        wrongpositiontosend = []
                        wrongpositiontosend.append(WrongPosition[i][0])
                        wrongpositiontosend.append(WrongPosition[i][1])
                        for k in range(NumberOfWrongNeedles):
                            if k!=i and WrongPosition[k][0] == WrongPosition[i][0]:
                                wrongpositiontosend.append(WrongPosition[k][1])
                                Counting[k]=1
                        for k in range(NumberOfLastWrongNeedles):
                            if LastWrongPositionsToSend[k][0] == WrongPosition[i][0]:
                                NumberOfRelated = len(LastWrongPositionsToSend[k])-1
                                for w in range(NumberOfRelated):
                                    wrongpositiontosend.append(LastWrongPositionsToSend[k][w+1])
                                    if LastWrongPositionsToSend[k][w+1]==WrongPosition[i][1] and WrongReason[i]==1:# this means they stil cross each other, Punish!
                                        Punish = feedbackindex+1
                        WrongPositionsToSend.append(wrongpositiontosend)
                NumberOfWrongPositionsToSend = len(WrongPositionsToSend)
                print("The",feedbackindex+1," Loop")
                for i in range(NumberOfWrongPositionsToSend):
                    CorrectedControlPoints, CorrectedControlPointsIJK = self.needleDetectionThread15_1( ijkA, imgData, imgLabelData, lrasTempPoints,0, fvSpacing, bUp=False, bScript=False, strManualName=strManualName, tipOnly=False, bDrawNeedle=False,whetherfeedback=True,ControlPointsPackage=ControlPointsPackage,ControlPointsPackageIJK=ControlPointsPackageIJK,WrongPosition=WrongPositionsToSend[i],GlobalDirection=GlobalDirection,Punish=Punish)
                    #CorrectedControlPoints, CorrectedControlPointsIJK = self.needleDetectionThread15_RM_Feedback(ControlPointsPackage, ControlPointsPackageIJK, WrongPositionsToSend[i], imgData, fvSpacing, imgLabelData, GlobalDirection,Punish,bScript,bUp)
                    ControlPointsPackage[WrongPositionsToSend[i][0]] = CorrectedControlPoints
                    ControlPointsPackageIJK[WrongPositionsToSend[i][0]] = CorrectedControlPointsIJK
                    print("The ",WrongPositionsToSend[i],"(+1 each) Needle corrected")
                LastWrongPositionsToSend = WrongPositionsToSend
        
    FinalControlPointsPackage, FinalControlPointsPackageIJK = self.needleRationalityProtect(OriginalControlPointsPackage, OriginalControlPointsPackageIJK, ControlPointsPackage, ControlPointsPackageIJK, OriginalWrongPositions,imgData,fvSpacing)    
   # FinalControlPointsPackage = ControlPointsPackage  
        
    for i in range(NumberOfNeedles):
        colorVar = i
        strManualName = names[i]
        if not bAutoStopTip:
          self.addNeedleToScene(FinalControlPointsPackage[i], (205)% MAXCOL, 'Detection', bScript,0, manualName=strManualName)
        if bAutoStopTip and bUp:
          self.addNeedleToScene(FinalControlPointsPackage[i], (205)% MAXCOL, 'Detection', bScript,0, manualName=strManualName)
    print("The Time of 15_RM:_________________________",time.perf_counter()-t0)


  def needleRationalityProtect(self, OriginalControlPointsPackage, OriginalControlPointsPackageIJK, ControlPointsPackage, ControlPointsPackageIJK, OriginalWrongPositions, imageData, spacing):
    '''Ruibin: comment... '''
    print("Protection Start")
    flag_slope = False
    flag_combine = False
    WrongPosition = []
    WrongReason = []
    widget = slicer.modules.NeedleFinderWidget
    radiusNeedleParameter = widget.radiusNeedleParameter.value
    gradient = widget.gradient.isChecked()
    gradientPonderation = widget.gradientPonderation.value
    FinalControlPointsPackage = ControlPointsPackage
    FinalControlPointsPackageIJK = ControlPointsPackageIJK
    
    # The check of the similarity with global direction
    GlobalDirection = [0,0,0]
    SingleDirection = []
    SlopeMetric_1 = []
    NumberOfNeedles = len(ControlPointsPackage)
    for i in range(NumberOfNeedles):
        NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
        direction = [ControlPointsPackage[i][0][0]-ControlPointsPackage[i][NumberOfPointsOnThisNeedle-1][0],
                     ControlPointsPackage[i][0][1]-ControlPointsPackage[i][NumberOfPointsOnThisNeedle-1][1],
                     ControlPointsPackage[i][0][2]-ControlPointsPackage[i][NumberOfPointsOnThisNeedle-1][2]]
        l = (direction[0]**2+direction[1]**2+direction[2]**2)**0.5
        direction = [direction[0]/l,direction[1]/l,direction[2]/l]
        SingleDirection.append(direction)
        print("The ",i+1," direction:[",direction[0],",",direction[1],",",direction[2],"]")
        GlobalDirection[0] += direction[0]
        GlobalDirection[1] += direction[1]
        GlobalDirection[2] += direction[2]
    GlobalDirection = [GlobalDirection[0]/NumberOfNeedles,GlobalDirection[1]/NumberOfNeedles,GlobalDirection[2]/NumberOfNeedles]
    for i in range(NumberOfNeedles):
        tempmetric1 = 1-(SingleDirection[i][0]*GlobalDirection[0]+SingleDirection[i][1]*GlobalDirection[1]+SingleDirection[i][2]*GlobalDirection[2])
        print("The ",i+1," slope_metric_1:",tempmetric1)
        SlopeMetric_1.append(tempmetric1)
    print("The global direction is:[",GlobalDirection[0],",",GlobalDirection[1],",",GlobalDirection[2],"]")
    # The check of curvature
    CurvatureMetric = []
    for i in range(NumberOfNeedles):
        NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
        tempmetric2 = 0;
        if NumberOfPointsOnThisNeedle>2:
            for j in range(NumberOfPointsOnThisNeedle - 2):
                direction1 = [ControlPointsPackage[i][j][0]-ControlPointsPackage[i][j+1][0],
                              ControlPointsPackage[i][j][1]-ControlPointsPackage[i][j+1][1],
                              ControlPointsPackage[i][j][2]-ControlPointsPackage[i][j+1][2]]
                l1 = (direction1[0]**2+direction1[1]**2+direction1[2]**2)**0.5                   
                direction2 = [ControlPointsPackage[i][j+1][0]-ControlPointsPackage[i][j+2][0],
                              ControlPointsPackage[i][j+1][1]-ControlPointsPackage[i][j+2][1],
                              ControlPointsPackage[i][j+1][2]-ControlPointsPackage[i][j+2][2]]
                l2 = (direction2[0]**2+direction2[1]**2+direction2[2]**2)**0.5
                if l1!=0 and l2!=0:
                    direction1 = [direction1[0]/l1,direction1[1]/l1,direction1[2]/l1]
                    direction2 = [direction2[0]/l2,direction2[1]/l2,direction2[2]/l2]
                    temp = 1-(direction1[0]*direction2[0]+direction1[1]*direction2[1]+direction1[2]*direction2[2])
                    tempmetric2 += temp
        print("The ",i+1," curvature_metric_2:",tempmetric2)
        CurvatureMetric.append(tempmetric2)   

    #distance metric
    DistanceMetric = []
    NumberOfNeedles = len(ControlPointsPackage)
    
    for i in range(NumberOfNeedles):
        NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
        currentDistances = [([0]*NumberOfNeedles) for k in range(NumberOfPointsOnThisNeedle)]
        for j in range(NumberOfPointsOnThisNeedle):
            
            currentPoint = ControlPointsPackage[i][j]
            
            for p in range(NumberOfNeedles):
                if p!=i:
                    NumberOfPointsOnThatNeedle = len(ControlPointsPackage[p])
                    distance = 9999
                    if currentPoint[2]>=ControlPointsPackage[p][0][2]:
                        distance = ((currentPoint[0]-ControlPointsPackage[p][0][0])**2+(currentPoint[1]-ControlPointsPackage[p][0][1])**2+(currentPoint[2]-ControlPointsPackage[p][0][2])**2)**0.5
                    elif currentPoint[2]<=ControlPointsPackage[p][NumberOfPointsOnThatNeedle-1][2]:
                        distance = ((currentPoint[0]-ControlPointsPackage[p][NumberOfPointsOnThatNeedle-1][0])**2+(currentPoint[1]-ControlPointsPackage[p][NumberOfPointsOnThatNeedle-1][1])**2+(currentPoint[2]-ControlPointsPackage[p][NumberOfPointsOnThatNeedle-1][2])**2)**0.5
                    else:
                        for q in range(NumberOfPointsOnThatNeedle-1):  
                            pointabove = ControlPointsPackage[p][q]
                            pointbelow = ControlPointsPackage[p][q+1]
                            if pointabove[2]<=currentPoint[2]:
                                h = ((currentPoint[0]-pointabove[0])**2+(currentPoint[1]-pointabove[1])**2+(currentPoint[2]-pointabove[2])**2)**0.5
                            elif pointbelow[2]>=currentPoint[2]:
                                h = ((currentPoint[0]-pointbelow[0])**2+(currentPoint[1]-pointbelow[1])**2+(currentPoint[2]-pointbelow[2])**2)**0.5
                            else:
                                a = ((pointabove[0]-currentPoint[0])**2+(pointabove[1]-currentPoint[1])**2+(pointabove[2]-currentPoint[2])**2)**0.5
                                b = ((pointbelow[0]-currentPoint[0])**2+(pointbelow[1]-currentPoint[1])**2+(pointbelow[2]-currentPoint[2])**2)**0.5
                                c = ((pointabove[0]-pointbelow[0])**2+(pointabove[1]-pointbelow[1])**2+(pointabove[2]-pointbelow[2])**2)**0.5
                                hc= (a+b+c)/2
                                h = (hc*(hc-a)*(hc-b)*(hc-c))**0.5*2/c
                            if h< distance:
                                distance = h
                    currentDistances[j][p] = distance
        DistanceMetric.append(currentDistances)
    DistanceBetweenEachOther = [([0]*NumberOfNeedles) for k in range(NumberOfNeedles)]
    DistanceThreshold = radiusNeedleParameter*0.85
    OverlapCounter = [([0]*NumberOfNeedles) for k in range(NumberOfNeedles)]
    NearestPoint = [([0]*NumberOfNeedles) for k in range(NumberOfNeedles)]
    NearestCoordinate = [([([0]*3) for k in range(NumberOfNeedles)]) for n in range(NumberOfNeedles)]
    print("DistanceMetric")
    print("Distance Between Each Other:")
    for i in range(NumberOfNeedles):
        NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
        if i<9:
            print("Needle  ",i+1,":", end=' ')
        else:
            print("Needle ",i+1,":", end=' ')
        for j in range(NumberOfNeedles):
            if i!=j:
                distancebetween = 9999
                NumberOfPointsOnThatNeedle = len(ControlPointsPackage[j])
                for k in range(NumberOfPointsOnThisNeedle):
                    if DistanceMetric[i][k][j]<distancebetween:
                        distancebetween = DistanceMetric[i][k][j]
                        nearestPointIndex = k
                NearestPoint[i][j] = nearestPointIndex
                res = 10
                neighbour = []
                if nearestPointIndex==0:
                    for t in range(res):
                        temp = [float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][0]-ControlPointsPackage[i][nearestPointIndex][0])+ControlPointsPackage[i][nearestPointIndex][0],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][1]-ControlPointsPackage[i][nearestPointIndex][1])+ControlPointsPackage[i][nearestPointIndex][1],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][2]-ControlPointsPackage[i][nearestPointIndex][2])+ControlPointsPackage[i][nearestPointIndex][2]]
                        neighbour.append(temp)
                elif nearestPointIndex==NumberOfPointsOnThisNeedle-1:
                    for t in range(res):
                        temp = [float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][0]-ControlPointsPackage[i][nearestPointIndex][0])+ControlPointsPackage[i][nearestPointIndex][0],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][1]-ControlPointsPackage[i][nearestPointIndex][1])+ControlPointsPackage[i][nearestPointIndex][1],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][2]-ControlPointsPackage[i][nearestPointIndex][2])+ControlPointsPackage[i][nearestPointIndex][2]]
                        neighbour.append(temp)
                else:  
                    for t in range(res):
                        temp = [float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][0]-ControlPointsPackage[i][nearestPointIndex][0])+ControlPointsPackage[i][nearestPointIndex][0],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][1]-ControlPointsPackage[i][nearestPointIndex][1])+ControlPointsPackage[i][nearestPointIndex][1],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][2]-ControlPointsPackage[i][nearestPointIndex][2])+ControlPointsPackage[i][nearestPointIndex][2]]
                        neighbour.append(temp)
                        temp = [float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][0]-ControlPointsPackage[i][nearestPointIndex][0])+ControlPointsPackage[i][nearestPointIndex][0],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][1]-ControlPointsPackage[i][nearestPointIndex][1])+ControlPointsPackage[i][nearestPointIndex][1],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][2]-ControlPointsPackage[i][nearestPointIndex][2])+ControlPointsPackage[i][nearestPointIndex][2]]
                        neighbour.append(temp)
                NumberOfNeighbour = len(neighbour)
                distance = 9999
                for n in range(NumberOfNeighbour):
                    currentPoint = neighbour[n]
                    if currentPoint[2]>=ControlPointsPackage[j][0][2]:
                        hmin = ((currentPoint[0]-ControlPointsPackage[j][0][0])**2+(currentPoint[1]-ControlPointsPackage[j][0][1])**2+(currentPoint[2]-ControlPointsPackage[j][0][2])**2)**0.5
                    elif currentPoint[2]<=ControlPointsPackage[j][NumberOfPointsOnThatNeedle-1][2]:
                        hmin = ((currentPoint[0]-ControlPointsPackage[j][NumberOfPointsOnThatNeedle-1][0])**2+(currentPoint[1]-ControlPointsPackage[j][NumberOfPointsOnThatNeedle-1][1])**2+(currentPoint[2]-ControlPointsPackage[j][NumberOfPointsOnThatNeedle-1][2])**2)**0.5
                    else:
                        hmin = 9999
                        for q in range(NumberOfPointsOnThatNeedle-1):  
                            pointabove = ControlPointsPackage[j][q]
                            pointbelow = ControlPointsPackage[j][q+1]
                            if pointabove[2]<=currentPoint[2]:
                                h = ((currentPoint[0]-pointabove[0])**2+(currentPoint[1]-pointabove[1])**2+(currentPoint[2]-pointabove[2])**2)**0.5
                            elif pointbelow[2]>=currentPoint[2]:
                                h = ((currentPoint[0]-pointbelow[0])**2+(currentPoint[1]-pointbelow[1])**2+(currentPoint[2]-pointbelow[2])**2)**0.5
                            else:
                                a = ((pointabove[0]-currentPoint[0])**2+(pointabove[1]-currentPoint[1])**2+(pointabove[2]-currentPoint[2])**2)**0.5
                                b = ((pointbelow[0]-currentPoint[0])**2+(pointbelow[1]-currentPoint[1])**2+(pointbelow[2]-currentPoint[2])**2)**0.5
                                c = ((pointabove[0]-pointbelow[0])**2+(pointabove[1]-pointbelow[1])**2+(pointabove[2]-pointbelow[2])**2)**0.5
                                hc= (a+b+c)/2
                                h = (hc*(hc-a)*(hc-b)*(hc-c))**0.5*2/c
                            if h<hmin:
                                hmin = h
                    if hmin< distance:
                        distance = hmin
                NearestCoordinate[i][j] = currentPoint
                DistanceBetweenEachOther[i][j] = distance
                
                
                
                DistanceThreshold = radiusNeedleParameter*0.85
                judge = SlopeMetric_1[i]
                if SlopeMetric_1[j]>SlopeMetric_1[i]:
                    judge = SlopeMetric_1[j]
                DistanceThreshold += judge
                if distance<DistanceThreshold:
                    OverlapCounter[i][j] += 1  
            print(" (%d"%NearestPoint[i][j],")%.2f"%DistanceBetweenEachOther[i][j], end=' ')
        print(" ")
    print("The Overlap Counter")    
    for i in range(NumberOfNeedles):
        if i<9:
            print("Needle  ",i+1,":", end=' ')
        else:
            print("Needle ",i+1,":", end=' ')
        for j in range(NumberOfNeedles):
            print(" %d"%OverlapCounter[i][j], end=' ')
        print(" ")        
    print("Check End")
    print("Diagnosis Start")
    InteractPairs = []
    InteractReasons = []
    for i in range(NumberOfNeedles):
        for j in range(i+1,NumberOfNeedles):
            if OverlapCounter[i][j] >0 or OverlapCounter[j][i] >0:
                
                NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
                NumberOfPointsOnThatNeedle = len(ControlPointsPackage[j])
                joinflag1 = 1
                joinflag2 = 1
                for m in range(NumberOfPointsOnThisNeedle):
                    if DistanceMetric[i][m][j]>2:
                        joinflag1 = 0
                for n in range(NumberOfPointsOnThatNeedle):
                    if DistanceMetric[j][n][i]>2:
                        joinflag2 = 0
                if joinflag1==1:
                    interactpair = [j,i]
                    interactreason = 2
                elif joinflag2==1:
                    interactpair = [i,j]
                    interactreason = 2
                
                
                elif   SlopeMetric_1[i] > SlopeMetric_1[j]+0.02 or CurvatureMetric[i] > CurvatureMetric[j]+0.015:
                    interactpair = [i,j]
                    interactreason = 1
                elif SlopeMetric_1[i]+0.02 < SlopeMetric_1[j] or CurvatureMetric[i]+0.015 < CurvatureMetric[j]:
                    interactpair = [j,i]
                    interactreason = 1
                elif SlopeMetric_1[i] > SlopeMetric_1[j]+0.01 and CurvatureMetric[i] > CurvatureMetric[j]+0.005:
                    interactpair = [i,j]
                    interactreason = 1
                elif SlopeMetric_1[i]+0.01 < SlopeMetric_1[j] and CurvatureMetric[i]+0.005 < CurvatureMetric[j]: #one of the needles has a much higher slope or curvature
                    interactpair = [j,i]
                    interactreason = 1
                else:
                    
                    direction1 = [ControlPointsPackage[i][0][0]-NearestCoordinate[i][j][0],ControlPointsPackage[i][0][1]-NearestCoordinate[i][j][1],ControlPointsPackage[i][0][2]-NearestCoordinate[i][j][2]]
                    direction2 = [ControlPointsPackage[j][0][0]-NearestCoordinate[i][j][0],ControlPointsPackage[j][0][1]-NearestCoordinate[i][j][1],ControlPointsPackage[j][0][2]-NearestCoordinate[i][j][2]]
                    l1 = (direction1[0]**2+direction1[1]**2+direction1[2]**2)**0.5   
                    l2 = (direction2[0]**2+direction2[1]**2+direction2[2]**2)**0.5
                    
                    
                    
                    if l1<1 and ControlPointsPackage[j][0][2]>ControlPointsPackage[i][0][2] or l1==0:
                        interactpair = [j,i]
                        interactreason = 2
                    elif l2<1 and ControlPointsPackage[i][0][2]>ControlPointsPackage[j][0][2] or l2==0:
                        interactpair = [i,j]
                        interactreason = 2
                    elif direction1[2]/l1 > direction2[2]/l2:
                        interactpair = [i,j]
                        interactreason = 3
                    elif direction2[2]/l2 > direction1[2]/l1:
                        interactpair = [j,i]
                        interactreason = 3
                          
                    
                InteractPairs.append(interactpair)  
                InteractReasons.append(interactreason)
    WrongPosition = InteractPairs
    WrongReason   = InteractReasons
    print("Wrong Positions __ Interact Pairs (Interact Reason):")
    NumberOfInteractPairs = len(InteractPairs)
    for i in range(NumberOfInteractPairs):
        pairtoshow = [InteractPairs[i][0]+1,InteractPairs[i][1]+1]
        print(pairtoshow,"(%d"%InteractReasons[i],")")
    print("Diagnosis end")
    
    
    
    NumberOfOriginalWrongPositions = len(OriginalWrongPositions)
    for i in range(NumberOfOriginalWrongPositions):
        if CurvatureMetric[OriginalWrongPositions[i][0]]>0.4 or SlopeMetric_1[OriginalWrongPositions[i][0]]>0.03:
            print("The ",OriginalWrongPositions[i][0]+1,"needle is protected")
            FinalControlPointsPackage[OriginalWrongPositions[i][0]] = OriginalControlPointsPackage[OriginalWrongPositions[i][0]]
            FinalControlPointsPackageIJK[OriginalWrongPositions[i][0]] = OriginalControlPointsPackageIJK[OriginalWrongPositions[i][0]]
        #else:
            #for j in range(NumberOfInteractPairs):
            #    if InteractPairs[j][0]==OriginalWrongPositions[i][0]:
            #        FinalControlPointsPackage[OriginalWrongPositions[i][0]] = OriginalControlPointsPackage[OriginalWrongPositions[i][0]]
            #        FinalControlPointsPackageIJK[OriginalWrongPositions[i][0]] = OriginalControlPointsPackageIJK[OriginalWrongPositions[i][0]]
    
    return FinalControlPointsPackage, FinalControlPointsPackageIJK

  def needleRationalityCheck(self, ControlPointsPackageIJK, ControlPointsPackage, imageData, spacing):
    """
    Ruibin:
    To check which needle is wrong according to their slope and their relative location
    More criteria can be added here.
    return: There will be two outputs: 
        1. A list of [the index of the wrong needle, the index of the first wrong point on each needle]
        
        2. The reason why they are wrong, in flags like [true, false] or [true, true]...
        
    """ 
    print("Check Start")
    flag_slope = False
    flag_combine = False
    WrongPosition = []
    WrongReason = []
    widget = slicer.modules.NeedleFinderWidget
    radiusNeedleParameter = widget.radiusNeedleParameter.value
    gradient = widget.gradient.isChecked()
    gradientPonderation = widget.gradientPonderation.value
    
    # The check of the similarity with global direction
    GlobalDirection = [0,0,0]
    SingleDirection = []
    SlopeMetric_1 = []
    NumberOfNeedles = len(ControlPointsPackage)
    for i in range(NumberOfNeedles):
        NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
        direction = [ControlPointsPackage[i][0][0]-ControlPointsPackage[i][NumberOfPointsOnThisNeedle-1][0],
                     ControlPointsPackage[i][0][1]-ControlPointsPackage[i][NumberOfPointsOnThisNeedle-1][1],
                     ControlPointsPackage[i][0][2]-ControlPointsPackage[i][NumberOfPointsOnThisNeedle-1][2]]
        l = (direction[0]**2+direction[1]**2+direction[2]**2)**0.5
        direction = [direction[0]/l,direction[1]/l,direction[2]/l]
        SingleDirection.append(direction)
        print("The ",i+1," direction:[",direction[0],",",direction[1],",",direction[2],"]")
        GlobalDirection[0] += direction[0]
        GlobalDirection[1] += direction[1]
        GlobalDirection[2] += direction[2]
    GlobalDirection = [GlobalDirection[0]/NumberOfNeedles,GlobalDirection[1]/NumberOfNeedles,GlobalDirection[2]/NumberOfNeedles]
    for i in range(NumberOfNeedles):
        tempmetric1 = 1-(SingleDirection[i][0]*GlobalDirection[0]+SingleDirection[i][1]*GlobalDirection[1]+SingleDirection[i][2]*GlobalDirection[2])
        print("The ",i+1," slope_metric_1:",tempmetric1)
        SlopeMetric_1.append(tempmetric1)
    print("The global direction is:[",GlobalDirection[0],",",GlobalDirection[1],",",GlobalDirection[2],"]")
    # The check of curvature
    CurvatureMetric = []
    for i in range(NumberOfNeedles):
        NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
        tempmetric2 = 0;
        if NumberOfPointsOnThisNeedle>2:
            for j in range(NumberOfPointsOnThisNeedle - 2):
                direction1 = [ControlPointsPackage[i][j][0]-ControlPointsPackage[i][j+1][0],
                              ControlPointsPackage[i][j][1]-ControlPointsPackage[i][j+1][1],
                              ControlPointsPackage[i][j][2]-ControlPointsPackage[i][j+1][2]]
                l1 = (direction1[0]**2+direction1[1]**2+direction1[2]**2)**0.5                   
                direction2 = [ControlPointsPackage[i][j+1][0]-ControlPointsPackage[i][j+2][0],
                              ControlPointsPackage[i][j+1][1]-ControlPointsPackage[i][j+2][1],
                              ControlPointsPackage[i][j+1][2]-ControlPointsPackage[i][j+2][2]]
                l2 = (direction2[0]**2+direction2[1]**2+direction2[2]**2)**0.5
                if l1!=0 and l2!=0:
                    direction1 = [direction1[0]/l1,direction1[1]/l1,direction1[2]/l1]
                    direction2 = [direction2[0]/l2,direction2[1]/l2,direction2[2]/l2]
                    temp = 1-(direction1[0]*direction2[0]+direction1[1]*direction2[1]+direction1[2]*direction2[2])
                    tempmetric2 += temp
        print("The ",i+1," curvature_metric_2:",tempmetric2)
        CurvatureMetric.append(tempmetric2)
    # The check of Variation
    """
    print "Variation Metric:"
    
    VariationMetric = [0]*NumberOfNeedles
    RangeMetric     = [0]*NumberOfNeedles
    for i in range(NumberOfNeedles):
        RecordOfThisNeedle = []
        NumberOfPointsOnThisNeedle = len(ControlPointsPackageIJK[i])
        for j in range(NumberOfPointsOnThisNeedle-1):
            currentPoint = ControlPointsPackageIJK[i][j]
            nextPoint    = ControlPointsPackageIJK[i][j+1]
            L = ((currentPoint[0]-nextPoint[0])**2+(currentPoint[0]-nextPoint[0])**2+(currentPoint[0]-nextPoint[0])**2)**0.5
            tIter = int(round(L))
            if tIter!=0:
                for t in xrange(int(tIter) + 1):
                    
                    tt = t / float(tIter)
                    ijk = [0,0,0]
                    for n in range(3):
                        M = (1 - tt) * currentPoint[n] + tt * nextPoint[n]
                        ijk[n] = int(round(M))
                    if gradient == 1 :
                        center = _get_scalar_safe(imageData, ijk[0], ijk[1], ijk[2], 0)
    
                        radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
                        radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))
    
                        g1 = _get_scalar_safe(imageData, ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
                        g2 = _get_scalar_safe(imageData, ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
                        g3 = _get_scalar_safe(imageData, ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
                        g4 = _get_scalar_safe(imageData, ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)
                        g5 = _get_scalar_safe(imageData, ijk[0] + radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                        g6 = _get_scalar_safe(imageData, ijk[0] - radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
                        g7 = _get_scalar_safe(imageData, ijk[0] - radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                        g8 = _get_scalar_safe(imageData, ijk[0] + radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
        
                        value = 8 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8) * gradientPonderation
                    else:
                        value = _get_scalar_safe(imageData, ijk[0], ijk[1], ijk[2], 0)
                    RecordOfThisNeedle.append(value)
        LengthOfThisRecord = len(RecordOfThisNeedle)
        average = 0
        variation = 0
        
        for k in range(LengthOfThisRecord):
            average = average+RecordOfThisNeedle[k]
        if LengthOfThisRecord!= 0:
            average /=  LengthOfThisRecord
        Sum = 0
        for k in range(LengthOfThisRecord-1):
            Sum += abs(RecordOfThisNeedle[k]-RecordOfThisNeedle[k+1])
        if LengthOfThisRecord!=0:
            Sum /= LengthOfThisRecord
        for k in range(LengthOfThisRecord):
            variation = variation+(RecordOfThisNeedle[k]-average)**2
        if LengthOfThisRecord!=0:
            variation /=LengthOfThisRecord
        Range = 0
        if LengthOfThisRecord!=0:
            Range = max(RecordOfThisNeedle) - min(RecordOfThisNeedle)
        
        VariationMetric[i] = variation
        RangeMetric[i] = Range
        if i<9:
            print "Needle  ",i+1,":",variation," ",Range," ",Sum
        else:
            print "Needle ",i+1,":",variation," ",Range," ",Sum
    """      
    # The check of location
    DistanceMetric = []
    NumberOfNeedles = len(ControlPointsPackage)
    
    for i in range(NumberOfNeedles):
        NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
        currentDistances = [([0]*NumberOfNeedles) for k in range(NumberOfPointsOnThisNeedle)]
        for j in range(NumberOfPointsOnThisNeedle):
            
            currentPoint = ControlPointsPackage[i][j]
            
            for p in range(NumberOfNeedles):
                if p!=i:
                    NumberOfPointsOnThatNeedle = len(ControlPointsPackage[p])
                    distance = 9999
                    if currentPoint[2]>=ControlPointsPackage[p][0][2]:
                        distance = ((currentPoint[0]-ControlPointsPackage[p][0][0])**2+(currentPoint[1]-ControlPointsPackage[p][0][1])**2+(currentPoint[2]-ControlPointsPackage[p][0][2])**2)**0.5
                    elif currentPoint[2]<=ControlPointsPackage[p][NumberOfPointsOnThatNeedle-1][2]:
                        distance = ((currentPoint[0]-ControlPointsPackage[p][NumberOfPointsOnThatNeedle-1][0])**2+(currentPoint[1]-ControlPointsPackage[p][NumberOfPointsOnThatNeedle-1][1])**2+(currentPoint[2]-ControlPointsPackage[p][NumberOfPointsOnThatNeedle-1][2])**2)**0.5
                    else:
                        for q in range(NumberOfPointsOnThatNeedle-1):  
                            pointabove = ControlPointsPackage[p][q]
                            pointbelow = ControlPointsPackage[p][q+1]
                            if pointabove[2]<=currentPoint[2]:
                                h = ((currentPoint[0]-pointabove[0])**2+(currentPoint[1]-pointabove[1])**2+(currentPoint[2]-pointabove[2])**2)**0.5
                            elif pointbelow[2]>=currentPoint[2]:
                                h = ((currentPoint[0]-pointbelow[0])**2+(currentPoint[1]-pointbelow[1])**2+(currentPoint[2]-pointbelow[2])**2)**0.5
                            else:
                                a = ((pointabove[0]-currentPoint[0])**2+(pointabove[1]-currentPoint[1])**2+(pointabove[2]-currentPoint[2])**2)**0.5
                                b = ((pointbelow[0]-currentPoint[0])**2+(pointbelow[1]-currentPoint[1])**2+(pointbelow[2]-currentPoint[2])**2)**0.5
                                c = ((pointabove[0]-pointbelow[0])**2+(pointabove[1]-pointbelow[1])**2+(pointabove[2]-pointbelow[2])**2)**0.5
                                hc= (a+b+c)/2
                                h = (hc*(hc-a)*(hc-b)*(hc-c))**0.5*2/c
                            if h< distance:
                                distance = h
                    currentDistances[j][p] = distance
        DistanceMetric.append(currentDistances)
    DistanceBetweenEachOther = [([0]*NumberOfNeedles) for k in range(NumberOfNeedles)]
    DistanceThreshold = radiusNeedleParameter*0.85
    OverlapCounter = [([0]*NumberOfNeedles) for k in range(NumberOfNeedles)]
    NearestPoint = [([0]*NumberOfNeedles) for k in range(NumberOfNeedles)]
    NearestCoordinate = [([([0]*3) for k in range(NumberOfNeedles)]) for n in range(NumberOfNeedles)]
    print("DistanceMetric")
    print("Distance Between Each Other:")
    for i in range(NumberOfNeedles):
        NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
        if i<9:
            print("Needle  ",i+1,":", end=' ')
        else:
            print("Needle ",i+1,":", end=' ')
        for j in range(NumberOfNeedles):
            if i!=j:
                distancebetween = 9999
                NumberOfPointsOnThatNeedle = len(ControlPointsPackage[j])
                for k in range(NumberOfPointsOnThisNeedle):
                    if DistanceMetric[i][k][j]<distancebetween:
                        distancebetween = DistanceMetric[i][k][j]
                        nearestPointIndex = k
                NearestPoint[i][j] = nearestPointIndex
                res = 10
                neighbour = []
                if nearestPointIndex==0:
                    for t in range(res):
                        temp = [float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][0]-ControlPointsPackage[i][nearestPointIndex][0])+ControlPointsPackage[i][nearestPointIndex][0],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][1]-ControlPointsPackage[i][nearestPointIndex][1])+ControlPointsPackage[i][nearestPointIndex][1],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][2]-ControlPointsPackage[i][nearestPointIndex][2])+ControlPointsPackage[i][nearestPointIndex][2]]
                        neighbour.append(temp)
                elif nearestPointIndex==NumberOfPointsOnThisNeedle-1:
                    for t in range(res):
                        temp = [float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][0]-ControlPointsPackage[i][nearestPointIndex][0])+ControlPointsPackage[i][nearestPointIndex][0],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][1]-ControlPointsPackage[i][nearestPointIndex][1])+ControlPointsPackage[i][nearestPointIndex][1],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][2]-ControlPointsPackage[i][nearestPointIndex][2])+ControlPointsPackage[i][nearestPointIndex][2]]
                        neighbour.append(temp)
                else:  
                    for t in range(res):
                        temp = [float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][0]-ControlPointsPackage[i][nearestPointIndex][0])+ControlPointsPackage[i][nearestPointIndex][0],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][1]-ControlPointsPackage[i][nearestPointIndex][1])+ControlPointsPackage[i][nearestPointIndex][1],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex+1][2]-ControlPointsPackage[i][nearestPointIndex][2])+ControlPointsPackage[i][nearestPointIndex][2]]
                        neighbour.append(temp)
                        temp = [float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][0]-ControlPointsPackage[i][nearestPointIndex][0])+ControlPointsPackage[i][nearestPointIndex][0],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][1]-ControlPointsPackage[i][nearestPointIndex][1])+ControlPointsPackage[i][nearestPointIndex][1],
                                float(t)/float(res)*(ControlPointsPackage[i][nearestPointIndex-1][2]-ControlPointsPackage[i][nearestPointIndex][2])+ControlPointsPackage[i][nearestPointIndex][2]]
                        neighbour.append(temp)
                NumberOfNeighbour = len(neighbour)
                distance = 9999
                for n in range(NumberOfNeighbour):
                    currentPoint = neighbour[n]
                    if currentPoint[2]>=ControlPointsPackage[j][0][2]:
                        hmin = ((currentPoint[0]-ControlPointsPackage[j][0][0])**2+(currentPoint[1]-ControlPointsPackage[j][0][1])**2+(currentPoint[2]-ControlPointsPackage[j][0][2])**2)**0.5
                    elif currentPoint[2]<=ControlPointsPackage[j][NumberOfPointsOnThatNeedle-1][2]:
                        hmin = ((currentPoint[0]-ControlPointsPackage[j][NumberOfPointsOnThatNeedle-1][0])**2+(currentPoint[1]-ControlPointsPackage[j][NumberOfPointsOnThatNeedle-1][1])**2+(currentPoint[2]-ControlPointsPackage[j][NumberOfPointsOnThatNeedle-1][2])**2)**0.5
                    else:
                        hmin = 9999
                        for q in range(NumberOfPointsOnThatNeedle-1):  
                            pointabove = ControlPointsPackage[j][q]
                            pointbelow = ControlPointsPackage[j][q+1]
                            if pointabove[2]<=currentPoint[2]:
                                h = ((currentPoint[0]-pointabove[0])**2+(currentPoint[1]-pointabove[1])**2+(currentPoint[2]-pointabove[2])**2)**0.5
                            elif pointbelow[2]>=currentPoint[2]:
                                h = ((currentPoint[0]-pointbelow[0])**2+(currentPoint[1]-pointbelow[1])**2+(currentPoint[2]-pointbelow[2])**2)**0.5
                            else:
                                a = ((pointabove[0]-currentPoint[0])**2+(pointabove[1]-currentPoint[1])**2+(pointabove[2]-currentPoint[2])**2)**0.5
                                b = ((pointbelow[0]-currentPoint[0])**2+(pointbelow[1]-currentPoint[1])**2+(pointbelow[2]-currentPoint[2])**2)**0.5
                                c = ((pointabove[0]-pointbelow[0])**2+(pointabove[1]-pointbelow[1])**2+(pointabove[2]-pointbelow[2])**2)**0.5
                                hc= (a+b+c)/2
                                h = (hc*(hc-a)*(hc-b)*(hc-c))**0.5*2/c
                            if h<hmin:
                                hmin = h
                    if hmin< distance:
                        distance = hmin
                NearestCoordinate[i][j] = currentPoint
                DistanceBetweenEachOther[i][j] = distance
                
                
                
                DistanceThreshold = radiusNeedleParameter*0.85
                judge = SlopeMetric_1[i]
                if SlopeMetric_1[j]>SlopeMetric_1[i]:
                    judge = SlopeMetric_1[j]
                DistanceThreshold += judge
                if distance<DistanceThreshold:
                    OverlapCounter[i][j] += 1  
            print(" (%d"%NearestPoint[i][j],")%.2f"%DistanceBetweenEachOther[i][j], end=' ')
        print(" ")
    print("The Overlap Counter")    
    for i in range(NumberOfNeedles):
        if i<9:
            print("Needle  ",i+1,":", end=' ')
        else:
            print("Needle ",i+1,":", end=' ')
        for j in range(NumberOfNeedles):
            print(" %d"%OverlapCounter[i][j], end=' ')
        print(" ")        
    print("Check End")
    print("Diagnosis Start")
    InteractPairs = []
    InteractReasons = []
    for i in range(NumberOfNeedles):
        for j in range(i+1,NumberOfNeedles):
            if OverlapCounter[i][j] >0 or OverlapCounter[j][i] >0:
                
                NumberOfPointsOnThisNeedle = len(ControlPointsPackage[i])
                NumberOfPointsOnThatNeedle = len(ControlPointsPackage[j])
                joinflag1 = 1
                joinflag2 = 1
                for m in range(NumberOfPointsOnThisNeedle):
                    if DistanceMetric[i][m][j]>2:
                        joinflag1 = 0
                for n in range(NumberOfPointsOnThatNeedle):
                    if DistanceMetric[j][n][i]>2:
                        joinflag2 = 0
                if joinflag1==1:
                    interactpair = [j,i]
                    interactreason = 2
                elif joinflag2==1:
                    interactpair = [i,j]
                    interactreason = 2
                
                
                elif   SlopeMetric_1[i] > SlopeMetric_1[j]+0.02 or CurvatureMetric[i] > CurvatureMetric[j]+0.015:
                    interactpair = [i,j]
                    interactreason = 1
                elif SlopeMetric_1[i]+0.02 < SlopeMetric_1[j] or CurvatureMetric[i]+0.015 < CurvatureMetric[j]:
                    interactpair = [j,i]
                    interactreason = 1
                elif SlopeMetric_1[i] > SlopeMetric_1[j]+0.01 and CurvatureMetric[i] > CurvatureMetric[j]+0.005:
                    interactpair = [i,j]
                    interactreason = 1
                elif SlopeMetric_1[i]+0.01 < SlopeMetric_1[j] and CurvatureMetric[i]+0.005 < CurvatureMetric[j]: #one of the needles has a much higher slope or curvature
                    interactpair = [j,i]
                    interactreason = 1
                else:
                    
                    direction1 = [ControlPointsPackage[i][0][0]-NearestCoordinate[i][j][0],ControlPointsPackage[i][0][1]-NearestCoordinate[i][j][1],ControlPointsPackage[i][0][2]-NearestCoordinate[i][j][2]]
                    direction2 = [ControlPointsPackage[j][0][0]-NearestCoordinate[i][j][0],ControlPointsPackage[j][0][1]-NearestCoordinate[i][j][1],ControlPointsPackage[j][0][2]-NearestCoordinate[i][j][2]]
                    l1 = (direction1[0]**2+direction1[1]**2+direction1[2]**2)**0.5   
                    l2 = (direction2[0]**2+direction2[1]**2+direction2[2]**2)**0.5
                    
                    
                    
                    if l1<1 and ControlPointsPackage[j][0][2]>ControlPointsPackage[i][0][2] or l1==0:
                        interactpair = [j,i]
                        interactreason = 2
                    elif l2<1 and ControlPointsPackage[i][0][2]>ControlPointsPackage[j][0][2] or l2==0:
                        interactpair = [i,j]
                        interactreason = 2
                    elif direction1[2]/l1 > direction2[2]/l2:
                        interactpair = [i,j]
                        interactreason = 3
                    elif direction2[2]/l2 > direction1[2]/l1:
                        interactpair = [j,i]
                        interactreason = 3
                          
                    
                InteractPairs.append(interactpair)  
                InteractReasons.append(interactreason)
    WrongPosition = InteractPairs
    WrongReason   = InteractReasons
    print("Wrong Positions __ Interact Pairs (Interact Reason):")
    NumberOfInteractPairs = len(InteractPairs)
    for i in range(NumberOfInteractPairs):
        pairtoshow = [InteractPairs[i][0]+1,InteractPairs[i][1]+1]
        print(pairtoshow,"(%d"%InteractReasons[i],")")
    print("Diagnosis End")

    return WrongPosition, WrongReason, GlobalDirection

  def needleDetectionThreadCurrentDev(self, A, imageData, colorVar, spacing, script=False, imgLabelData=None, manualName=""):
    """
    From the needle tip, the algorithm looks for a direction maximizing the "needle likelihood" of a small segment in a conic region.
    The second extremity of this segment is saved as a control point (in controlPoints), used later.
    Then, this step is iterated, replacing the needle tip by the latest control point.
    The height of the new conic region (stepsize) is increased as well as its base diameter (rMax) and its normal is collinear to the previous computed segment. (cf. C0)
    NbStepsNeedle iterations give NbStepsNeedle-1 control points, the last one being used as an extremity as well as the needle tip.
    From these NbStepsNeedle-1 control points and 2 extremities a Bezier curve is computed, approximating the needle path.
    """
    profprint()
    get_scalar = _ensure_scalar_getter(imageData)
    global conesColor
    if conesColor:
      conesColor = (conesColor + 1) % 308
    if conesColor == 0:
      conesColor = 300
    ijk = [0, 0, 0]
    bestPoint = [0, 0, 0]
    mode = "circle"

    widget = slicer.modules.NeedleFinderWidget

    distanceMax = widget.radiusMax.value
    gradientPonderation = widget.gradientPonderation.value
    sigmaValue = widget.sigmaValue.value
    gaussianAttenuationChecked = widget.gaussianAttenuationButton.isChecked()
    gradient = widget.gradient.isChecked()
    numberOfPointsPerNeedle = max(1, widget.numberOfPointsPerNeedle.value - 1)
    nbRotatingIterations = widget.nbRotatingIterations.value
    radiusNeedleParameter = widget.radiusNeedleParameter.value
    axialSegmentationLimit = widget.axialSegmentationLimit
    lenghtNeedleParameter = widget.lenghtNeedleParameter.value / (spacing[2])
    autoCorrectTip = widget.autoCorrectTip.isChecked()
    exponent = widget.exponent.value
    drawFiducialPoints = widget.drawFiducialPoints.isChecked()
    autoStopTip = widget.autoStopTip.isChecked()

    A = [int(A[0]), int(A[1]), int(A[2])]
    if axialSegmentationLimit is not None:
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    elif axialSegmentationLimit is None and widget.maxLength.isChecked():
      axialSegmentationLimit = 0
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    else:
      lenghtNeedle = lenghtNeedleParameter

    rMax = distanceMax / float(spacing[0])
    NbStepsNeedle = numberOfPointsPerNeedle - 1
    nbRotatingStep = nbRotatingIterations

    dims = [0, 0, 0]
    imageData.GetDimensions(dims)

    A0 = A

    self.controlPoints = []
    controlPointsIJK = []
    bestControlPoints = []

    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))

    coeff = 1
    radiusSphere = 5
    X = coeff * radiusSphere / float(spacing[0])
    Y = coeff * radiusSphere / float(spacing[1])
    Z = 5 / float(spacing[2])
    if autoCorrectTip:
      A = self.findTip(A, get_scalar, radiusNeedle, coeff, sigmaValue, gradientPonderation, X, Y, Z)
    A0 = A

    self.controlPoints.append(self.ijk2ras(A))
    controlPointsIJK.append(A)
    bestControlPoints.append(self.ijk2ras(A))
    if drawFiducialPoints:
      create_fiducial_node(name='.c0_' + str(colorVar), ras=self.ijk2ras(A), color=(0.0, 0.0, 1.0))

    for step in range(0, NbStepsNeedle + 1):
      if step == 0:
        stepSize = self.stepSize(step, NbStepsNeedle) * lenghtNeedle
        Vx = 0
        Vy = 0
        Vz = -stepSize
        rMax = distanceMax / float(spacing[0])
        rIter = int(round(rMax))
        tIter = max(1, int(round(stepSize)))
        tot = stepSize
      else:
        stepSize = self.stepSize(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        rMax = max(stepSize, distanceMax / float(spacing[0]))
        rIter = max(15, min(20, int(rMax / float(spacing[0]))))
        tIter = max(1, int(round(stepSize)))
        Vx = A[0] - tip0[0]
        Vy = A[1] - tip0[1]
        Vz = A[2] - tip0[2]

      coeffSize = abs(stepSize) / spacing[2]
      if Vz != 0:
        K = coeffSize / float(abs(Vz))
      else:
        break

      P0 = A
      C0 = [
          P0[0] + K * Vx,
          P0[1] + K * Vy,
          P0[2] + K * Vz,
      ]

      if drawFiducialPoints:
        create_fiducial_node(name='.b' + str(step + 1) + '_' + str(colorVar), ras=self.ijk2ras(C0), color=(0.0, 0.0, 1.0))

      estimator = 0
      minEstimator = 0
      step_sigma = 100 if (gaussianAttenuationChecked == 1 and step >= 2 and step == NbStepsNeedle) else sigmaValue
      rust_cone_res = (
          rust_search_cone_current_dev(
              imageData,
              imgLabelData,
              A,
              C0,
              rMax,
              int(rIter),
              int(nbRotatingStep),
              int(tIter),
              float(exponent),
              int(gradient),
              mode,
              int(radiusNeedle),
              int(radiusNeedleCorner),
              float(gradientPonderation),
              1 if (gaussianAttenuationChecked == 1 and step >= 2) else 0,
              bool(step >= 2 and Vz != 0),
              float(step_sigma),
              bestPoint,
          )
          if not conesColor
          else None
      )
      if rust_cone_res is not None:
        bestPoint, minEstimator = rust_cone_res
        if step == NbStepsNeedle and gaussianAttenuationChecked == 1 and step >= 2:
          sigmaValue = 100
      else:
        for R in range(rIter + 1):
          r = R * (rMax / float(rIter))
          for thetaStep in range(nbRotatingStep):
            angleInDegree = (thetaStep * 360) / float(nbRotatingStep)
            theta = math.radians(angleInDegree)

            C = [
                C0[0] + r * (math.cos(theta)),
                C0[1] + r * (math.sin(theta)),
                C0[2],
            ]

            total = 0
            M = [[0, 0, 0] for i in range(int(tIter) + 1)]

            for t in range(tIter + 1):
              tt = t / float(tIter)
              for i in range(3):
                M[t][i] = (1 - tt) * A[i] + tt * C[i]
                ijk[i] = int(round(M[t][i]))

              ix, iy, iz = ijk[0], ijk[1], ijk[2]
              if 0 < ix < dims[0] and 0 < iy < dims[1] and 0 < iz < dims[2]:
                center = (
                    get_scalar(ix, iy, iz)
                    + get_scalar(ix + 1, iy, iz)
                    + get_scalar(ix - 1, iy, iz)
                    + get_scalar(ix, iy + 1, iz)
                    + get_scalar(ix, iy - 1, iz)
                    + get_scalar(ix + 1, iy + 1, iz)
                    + get_scalar(ix + 1, iy - 1, iz)
                    + get_scalar(ix - 1, iy + 1, iz)
                    + get_scalar(ix - 1, iy - 1, iz)
                )

                total += center ** exponent
                if gradient == 1 and mode == "circle":
                  g1 = get_scalar(ix + radiusNeedle, iy, iz)
                  g2 = get_scalar(ix - radiusNeedle, iy, iz)
                  g3 = get_scalar(ix, iy + radiusNeedle, iz)
                  g4 = get_scalar(ix, iy - radiusNeedle, iz)
                  g5 = get_scalar(ix + radiusNeedleCorner, iy + radiusNeedleCorner, iz)
                  g6 = get_scalar(ix - radiusNeedleCorner, iy - radiusNeedleCorner, iz)
                  g7 = get_scalar(ix - radiusNeedleCorner, iy + radiusNeedleCorner, iz)
                  g8 = get_scalar(ix + radiusNeedleCorner, iy - radiusNeedleCorner, iz)

                  total += ((center - g1) ** gradientPonderation) / float(tIter)
                  total += ((center - g2) ** gradientPonderation) / float(tIter)
                  total += ((center - g3) ** gradientPonderation) / float(tIter)
                  total += ((center - g4) ** gradientPonderation) / float(tIter)
                  total += ((center - g5) ** gradientPonderation) / float(tIter)
                  total += ((center - g6) ** gradientPonderation) / float(tIter)
                  total += ((center - g7) ** gradientPonderation) / float(tIter)
                  total += ((center - g8) ** gradientPonderation) / float(tIter)

                if gradient == 1 and mode == "square":
                  g1 = get_scalar(ix + radiusNeedle, iy, iz)
                  g2 = get_scalar(ix - radiusNeedle, iy, iz)
                  g3 = get_scalar(ix, iy + radiusNeedle, iz)
                  g4 = get_scalar(ix, iy - radiusNeedle, iz)
                  total += (8 * center - ((g1 + g2 + g3 + g4) / 4.0) * gradientPonderation) / float(tIter)

                if imgLabelData:
                  fLabel = imgLabelData.GetScalarComponentAsFloat(ix, iy, iz, 0)
                  if fLabel and fLabel < 300:
                    total -= 10000
                  if conesColor:
                    imgLabelData.SetScalarComponentFromFloat(ix, iy, iz, 0, conesColor)
            if R == 0:
              initialIntensity = total
              estimator = total

            if gaussianAttenuationChecked == 1 and step >= 2:
              if step == NbStepsNeedle:
                sigmaValue = 100

              if Vz != 0:
                rgauss = ((C[0] - C0[0]) ** 2 + (C[1] - C0[1]) ** 2 + (C[2] - C0[2]) ** 2) ** 0.5
                gaussianAttenuation = math.exp(-(rgauss / float(rMax)) ** 2 / float((2 * (sigmaValue / 10.0)) ** 2))
                estimator = total * gaussianAttenuation
              else:
                estimator = total
            else:
              estimator = total

            if estimator < initialIntensity:
              if estimator < minEstimator or minEstimator == 0:
                minEstimator = estimator
                if minEstimator != 0:
                  bestPoint = C

      tip0 = A
      if bestPoint == [0, 0, 0]:
        A = C0
      elif bestPoint != tip0:
        A = bestPoint

      if step == 1:
        self.estimatorReference = minEstimator

      if A[2] < axialSegmentationLimit and A != A0 and (tip0[2] - A[2]) != 0:
        asl = axialSegmentationLimit
        l = (A[2] - asl) / float(tip0[2] - A[2])
        A = [
            A[0] - l * (tip0[0] - A[0]),
            A[1] - l * (tip0[1] - A[1]),
            A[2] - l * (tip0[2] - A[2]),
        ]

      self.controlPoints.append(self.ijk2ras(A))
      controlPointsIJK.append(A)
      if drawFiducialPoints:
        create_fiducial_node(name='.c' + str(step) + '_' + str(colorVar), ras=self.controlPoints[step + 1])

      if A[2] <= axialSegmentationLimit and A != A0:
        break

    if not autoStopTip:
      self.addNeedleToScene(self.controlPoints, colorVar, 'Detection', script, 0, manualName=manualName)

  def needleDetectionThread13_1(self, Aori, imageData, colorVar, spacing, script=False, imgLabelData=None, manName="", bDrawNeedle=False, whetherfeedback=False, ControlPointsPackage=None, ControlPointsPackageIJK=None, WrongPosition=None, GlobalDirection=None, Punish=None):
    """MICCAI2013 version 13_1."""
    profprint()
    get_scalar = _ensure_scalar_getter(imageData)
    global conesColor
    if conesColor:
      conesColor = (conesColor + 1) % 308
    if conesColor == 0:
      conesColor = 300
    ijk = [0, 0, 0]
    bestPoint = [0, 0, 0]
    widget = slicer.modules.NeedleFinderWidget

    distanceMax = widget.radiusMax.value
    gradientPonderation = widget.gradientPonderation.value
    sigmaValue = widget.sigmaValue.value
    stepsize = widget.stepsize.value
    gaussianAttenuationChecked = widget.gaussianAttenuationButton.isChecked()
    gradient = widget.gradient.isChecked()
    numberOfPointsPerNeedle = widget.numberOfPointsPerNeedle.value
    nbRotatingIterations = widget.nbRotatingIterations.value
    radiusNeedleParameter = widget.radiusNeedleParameter.value
    axialSegmentationLimit = widget.axialSegmentationLimit
    autoStopTip = widget.autoStopTip.isChecked()

    if whetherfeedback:
      WrongIndex = WrongPosition[0]
      NumberOfRelatedIndexs = len(WrongPosition) - 1
      RelatedIndex = [WrongPosition[k + 1] for k in range(NumberOfRelatedIndexs)]
      A = ControlPointsPackageIJK[WrongIndex][0]
      NeighbourAttenuation = 0
    else:
      A = Aori

    if axialSegmentationLimit is not None:
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    else:
      lenghtNeedle = A[2] * 0.9 * spacing[2]

    rMax = distanceMax / float(spacing[0])
    NbStepsNeedle = numberOfPointsPerNeedle - 1
    nbRotatingStep = nbRotatingIterations

    dims = [0, 0, 0]
    imageData.GetDimensions(dims)

    A0 = A
    print(A0)
    if widget.drawFiducialPoints.isChecked():
      create_fiducial_node(name='.c0_' + str(colorVar), ras=self.ijk2ras(A), color=(0.0, 0.0, 1.0))

    controlPoints = []
    controlPointsIJK = []
    bestControlPoints = []

    controlPoints.append(self.ijk2ras(A))
    controlPointsIJK.append(A)
    bestControlPoints.append(self.ijk2ras(A))

    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))

    for step in range(0, NbStepsNeedle + 2):
      print("length", lenghtNeedle)
      if step == 0:
        L = self.stepSize13(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        C0 = [A[0], A[1], A[2] - L]
        rMax = distanceMax / float(spacing[0])
        rIter = rMax
        tIter = int(round(L))
      else:
        stepSize = self.stepSize13(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        print(stepSize)
        C0 = [
            2 * A[0] - tip0[0],
            2 * A[1] - tip0[1],
            A[2] - stepSize,
        ]
        rMax = max(stepSize, distanceMax / float(spacing[0]))
        rIter = max(15, min(20, int(rMax / float(spacing[0]))))
        tIter = stepSize

      if not script and widget.drawFiducialPoints.isChecked():
        create_fiducial_node(name='.b' + str(step + 1) + '_' + str(colorVar), ras=self.ijk2ras(C0), color=(0.0, 0.0, 1.0))

      estimator = 0
      minEstimator = 0
      rust_cone_res = (
          rust_search_cone_13(
              imageData,
              imgLabelData,
              A,
              C0,
              tip0 if step >= 2 else None,
              float(rMax),
              float(rIter),
              int(nbRotatingStep),
              float(tIter),
              int(gradient),
              int(radiusNeedle),
              int(radiusNeedleCorner),
              float(gradientPonderation),
              1 if (gaussianAttenuationChecked == 1 and step >= 2) else 0,
              float(sigmaValue),
              bestPoint,
              whether_feedback=bool(whetherfeedback),
              control_points_package=ControlPointsPackage if whetherfeedback else None,
              related_indices=RelatedIndex if whetherfeedback else None,
              ijk_to_ras_mat=_get_ijk_to_ras_16() if whetherfeedback else None,
              radius_needle_mm=float(radiusNeedleParameter),
              punish=float(Punish) if Punish is not None else 0.0,
          )
          if not conesColor
          else None
      )
      if rust_cone_res is not None:
        bestPoint, minEstimator = rust_cone_res
      else:
        for R in range(int(rIter) + 1):
          r = R * (rMax / float(rIter))
          for thetaStep in range(int(nbRotatingStep)):
            angleInDegree = (thetaStep * 360) / float(nbRotatingStep)
            theta = math.radians(angleInDegree)

            C = [
                C0[0] + r * (math.cos(theta)),
                C0[1] + r * (math.sin(theta)),
                C0[2],
            ]

            total = 0
            M = [[0, 0, 0] for i in range(int(tIter) + 1)]

            for t in range(int(tIter) + 1):
              tt = t / float(tIter)
              for i in range(3):
                M[t][i] = (1 - tt) * A[i] + tt * C[i]
                ijk[i] = int(round(M[t][i]))

              ix, iy, iz = ijk[0], ijk[1], ijk[2]
              if 0 < ix < dims[0] and 0 < iy < dims[1] and 0 < iz < dims[2]:
                center = get_scalar(ix, iy, iz)
                total += center

                if whetherfeedback:
                  NeighbourAttenuation = self.Feedback(ControlPointsPackage, Punish, radiusNeedleParameter, NumberOfRelatedIndexs, len, RelatedIndex, range, M[t])
                  total += NeighbourAttenuation

                if gradient == 1:
                  g1 = get_scalar(ix + radiusNeedle, iy, iz)
                  g2 = get_scalar(ix - radiusNeedle, iy, iz)
                  g3 = get_scalar(ix, iy + radiusNeedle, iz)
                  g4 = get_scalar(ix, iy - radiusNeedle, iz)
                  g5 = get_scalar(ix + radiusNeedleCorner, iy + radiusNeedleCorner, iz)
                  g6 = get_scalar(ix - radiusNeedleCorner, iy - radiusNeedleCorner, iz)
                  g7 = get_scalar(ix - radiusNeedleCorner, iy + radiusNeedleCorner, iz)
                  g8 = get_scalar(ix + radiusNeedleCorner, iy - radiusNeedleCorner, iz)
                  total += 8 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * gradientPonderation

                if imgLabelData:
                  fLabel = imgLabelData.GetScalarComponentAsFloat(ix, iy, iz, 0)
                  if fLabel and fLabel < 300:
                    total -= 10000
                  if conesColor:
                    imgLabelData.SetScalarComponentFromFloat(ix, iy, iz, 0, conesColor)
            if R == 0:
              initialIntensity = total
              estimator = total

            if gaussianAttenuationChecked == 1 and step >= 2:
              if tip0[2] - A[2] != 0:
                stepSize = A[2] - C0[2]
                K = stepSize / float(tip0[2] - A[2])
                X = [
                    A[0] + K * (A[0] - tip0[0]),
                    A[1] + K * (A[1] - tip0[1]),
                    A[2] + K * (A[2] - tip0[2]),
                ]
                rgauss = ((C[0] - X[0]) ** 2 + (C[1] - X[1]) ** 2 + (C[2] - X[2]) ** 2) ** 0.5
                gaussianAttenuation = math.exp(-(rgauss / float(rMax)) ** 2 / float((2 * (sigmaValue / 10.0)) ** 2))
                estimator = total * gaussianAttenuation
              else:
                estimator = total
            else:
              estimator = total

            if estimator < initialIntensity:
              if estimator < minEstimator or minEstimator == 0:
                minEstimator = estimator
                if minEstimator != 0:
                  bestPoint = C

      tip0 = A
      if bestPoint == [0, 0, 0]:
        A = C0
      elif bestPoint != tip0:
        A = bestPoint

      if A[2] < axialSegmentationLimit and A != A0 and (tip0[2] - A[2]) != 0:
        asl = axialSegmentationLimit
        l = (A[2] - asl) / float(tip0[2] - A[2])
        A = [
            A[0] - l * (tip0[0] - A[0]),
            A[1] - l * (tip0[1] - A[1]),
            A[2] - l * (tip0[2] - A[2]),
        ]

      controlPoints.append(self.ijk2ras(A))
      controlPointsIJK.append(A)

      if not script and widget.drawFiducialPoints.isChecked():
        create_fiducial_node(name='.c_' + str(colorVar), ras=controlPoints[step + 1])

      if A[2] <= axialSegmentationLimit and A != A0:
        break

    if not autoStopTip:
      if bDrawNeedle:
        self.addNeedleToScene(controlPoints, colorVar, 'Detection', script, 0, manualName=manName)

    return controlPoints, controlPointsIJK

  def needleDetectionThread13_2(self, A, imageData, colorVar, spacing, script=False, imgLabelData=None, manualName=""):
    """MICCAI2013 version 13_2 (with stepSize/spacing[2] bugfixes)."""
    profprint()
    t0 = time.perf_counter()
    get_scalar = _ensure_scalar_getter(imageData)
    global conesColor
    if conesColor:
      conesColor = (conesColor + 1) % 308
    if conesColor == 0:
      conesColor = 300
    ijk = [0, 0, 0]
    bestPoint = [0, 0, 0]
    widget = slicer.modules.NeedleFinderWidget

    distanceMax = widget.radiusMax.value
    gradientPonderation = widget.gradientPonderation.value
    sigmaValue = widget.sigmaValue.value
    stepsize = widget.stepsize.value
    gaussianAttenuationChecked = widget.gaussianAttenuationButton.isChecked()
    gradient = widget.gradient.isChecked()
    numberOfPointsPerNeedle = widget.numberOfPointsPerNeedle.value
    nbRotatingIterations = widget.nbRotatingIterations.value
    radiusNeedleParameter = widget.radiusNeedleParameter.value
    axialSegmentationLimit = widget.axialSegmentationLimit
    autoStopTip = widget.autoStopTip.isChecked()

    if axialSegmentationLimit is not None:
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    else:
      lenghtNeedle = A[2] * 0.9 * spacing[2]

    rMax = distanceMax / float(spacing[0])
    NbStepsNeedle = numberOfPointsPerNeedle - 1
    nbRotatingStep = nbRotatingIterations

    dims = [0, 0, 0]
    imageData.GetDimensions(dims)

    A0 = A
    print(A0)
    if widget.drawFiducialPoints.isChecked():
      create_fiducial_node(name='.c0_' + str(colorVar), ras=self.ijk2ras(A), color=(0.0, 0.0, 1.0))

    controlPoints = []
    controlPointsIJK = []
    bestControlPoints = []

    controlPoints.append(self.ijk2ras(A))
    controlPointsIJK.append(A)
    bestControlPoints.append(self.ijk2ras(A))

    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))

    for step in range(0, NbStepsNeedle + 2):
      print("length", lenghtNeedle)
      if step == 0:
        L = self.stepSize13(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        C0 = [A[0], A[1], A[2] - L / spacing[2]]
        rMax = distanceMax / float(spacing[0])
        rIter = rMax
        tIter = int(round(L))
      else:
        stepSize = self.stepSize13(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        print(stepSize)
        C0 = [
            2 * A[0] - tip0[0],
            2 * A[1] - tip0[1],
            A[2] - stepSize / spacing[2],
        ]
        rMax = max(stepSize, distanceMax / float(spacing[0]))
        rIter = max(15, min(20, int(rMax / float(spacing[0]))))
        tIter = stepSize

      if not script and widget.drawFiducialPoints.isChecked():
        create_fiducial_node(name='.b' + str(step + 1) + '_' + str(colorVar), ras=self.ijk2ras(C0), color=(0.0, 0.0, 1.0))

      estimator = 0
      minEstimator = 0
      rust_cone_res = (
          rust_search_cone_13(
              imageData,
              imgLabelData,
              A,
              C0,
              tip0 if step >= 2 else None,
              float(rMax),
              float(rIter),
              int(nbRotatingStep),
              float(tIter),
              int(gradient),
              int(radiusNeedle),
              int(radiusNeedleCorner),
              float(gradientPonderation),
              1 if (gaussianAttenuationChecked == 1 and step >= 2) else 0,
              float(sigmaValue),
              bestPoint,
          )
          if not conesColor
          else None
      )
      if rust_cone_res is not None:
        bestPoint, minEstimator = rust_cone_res
      else:
        for R in range(int(rIter) + 1):
          r = R * (rMax / float(rIter))
          for thetaStep in range(int(nbRotatingStep)):
            angleInDegree = (thetaStep * 360) / float(nbRotatingStep)
            theta = math.radians(angleInDegree)

            C = [
                C0[0] + r * (math.cos(theta)),
                C0[1] + r * (math.sin(theta)),
                C0[2],
            ]

            total = 0
            M = [[0, 0, 0] for i in range(int(tIter) + 1)]

            for t in range(int(tIter) + 1):
              tt = t / float(tIter)
              for i in range(3):
                M[t][i] = (1 - tt) * A[i] + tt * C[i]
                ijk[i] = int(round(M[t][i]))

              ix, iy, iz = ijk[0], ijk[1], ijk[2]
              if 0 < ix < dims[0] and 0 < iy < dims[1] and 0 < iz < dims[2]:
                center = get_scalar(ix, iy, iz)
                total += center
                if gradient == 1:
                  g1 = get_scalar(ix + radiusNeedle, iy, iz)
                  g2 = get_scalar(ix - radiusNeedle, iy, iz)
                  g3 = get_scalar(ix, iy + radiusNeedle, iz)
                  g4 = get_scalar(ix, iy - radiusNeedle, iz)
                  g5 = get_scalar(ix + radiusNeedleCorner, iy + radiusNeedleCorner, iz)
                  g6 = get_scalar(ix - radiusNeedleCorner, iy - radiusNeedleCorner, iz)
                  g7 = get_scalar(ix - radiusNeedleCorner, iy + radiusNeedleCorner, iz)
                  g8 = get_scalar(ix + radiusNeedleCorner, iy - radiusNeedleCorner, iz)
                  total += 8 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * gradientPonderation
                if imgLabelData:
                  fLabel = imgLabelData.GetScalarComponentAsFloat(ix, iy, iz, 0)
                  if fLabel and fLabel < 300:
                    total -= 10000
                  if conesColor:
                    imgLabelData.SetScalarComponentFromFloat(ix, iy, iz, 0, conesColor)
            if R == 0:
              initialIntensity = total
              estimator = total

            if gaussianAttenuationChecked == 1 and step >= 2:
              if tip0[2] - A[2] != 0:
                stepSize = A[2] - C0[2]
                K = stepSize / float(tip0[2] - A[2])
                X = [
                    A[0] + K * (A[0] - tip0[0]),
                    A[1] + K * (A[1] - tip0[1]),
                    A[2] + K * (A[2] - tip0[2]),
                ]
                rgauss = ((C[0] - X[0]) ** 2 + (C[1] - X[1]) ** 2 + (C[2] - X[2]) ** 2) ** 0.5
                gaussianAttenuation = math.exp(-(rgauss / float(rMax)) ** 2 / float((2 * (sigmaValue / 10.0)) ** 2))
                estimator = total * gaussianAttenuation
              else:
                estimator = total
            else:
              estimator = total

            if estimator < initialIntensity:
              if estimator < minEstimator or minEstimator == 0:
                minEstimator = estimator
                if minEstimator != 0:
                  bestPoint = C

      tip0 = A
      if bestPoint == [0, 0, 0]:
        A = C0
      elif bestPoint != tip0:
        A = bestPoint

      if A[2] < axialSegmentationLimit and A != A0 and (tip0[2] - A[2]) != 0:
        asl = axialSegmentationLimit
        l = (A[2] - asl) / float(tip0[2] - A[2])
        A = [
            A[0] - l * (tip0[0] - A[0]),
            A[1] - l * (tip0[1] - A[1]),
            A[2] - l * (tip0[2] - A[2]),
        ]

      controlPoints.append(self.ijk2ras(A))
      controlPointsIJK.append(A)

      if not script and widget.drawFiducialPoints.isChecked():
        create_fiducial_node(name='.c_' + str(colorVar), ras=controlPoints[step + 1])

      if A[2] <= axialSegmentationLimit and A != A0:
        break

    if not autoStopTip:
      self.addNeedleToScene(controlPoints, colorVar, 'Detection', script, manualName=manualName)
    print(time.perf_counter() - t0, "seconds process time")

  def addPlaneToScene(self, foot, x, y):
    """Add a plane to the scene."""
    profprint()
    scene = slicer.mrmlScene
    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetName(scene.GenerateUniqueName(".ObturatorPlane"))

    planeSource = vtk.vtkPlaneSource()
    foot -= 25 * (x + y)
    planeSource.SetOrigin(list(foot))
    planeSource.SetPoint1(np.array(foot) + 50 * x)
    planeSource.SetPoint2(np.array(foot) + 50 * y)
    planeSource.Update()
    model.SetAndObservePolyData(planeSource.GetOutput())

    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetColor(1, 1, 0)
    modelDisplay.SetBackfaceCulling(0)
    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    scene.AddNode(model)

  def addFiducialToScene(self, ijk, name, rgbColor=(1, 1, 0), glyphScale=2, textScale=2):
    """Add fiducial marker to scene."""
    create_fiducial_node(
        name=name,
        ras=self.ijk2ras(ijk),
        color=rgbColor,
        glyph_scale=glyphScale,
        text_scale=textScale,
    )

  def needleDetectionThread15_1(self, ijkAori, imgData, imgLabelData, lrasTempPoints, iColorVar, fvSpacing, bUp=False, bScript=False, strManualName="", tipOnly=False, bDrawNeedle=False, whetherfeedback=False, ControlPointsPackage=None, ControlPointsPackageIJK=None, WrongPosition=None, GlobalDirection=None, Punish=None):
    """MICCAI2015 version 15_1 with mechanical deflection model."""
    profprint()
    t0 = time.perf_counter()
    get_scalar = _ensure_scalar_getter(imgData)
    bDraw = True
    widget = slicer.modules.NeedleFinderWidget
    print("\n" * 10)
    iRadiusMax_mm = widget.radiusMax.value
    iGradientPonderation = widget.gradientPonderation.value
    iSigmaValue = widget.sigmaValue.value
    bGaussianAttenuation = widget.gaussianAttenuationButton.isChecked()
    bGradient = widget.gradient.isChecked()
    nPointsPerNeedle = widget.numberOfPointsPerNeedle.value
    nRotatingIts = widget.nbRotatingIterations.value
    iRadiusNeedle_mm = widget.radiusNeedleParameter.value
    iAxialSegmentationLimit = widget.axialSegmentationLimit
    bAutoStopTip = widget.autoStopTip.isChecked()
    global conesColor
    if conesColor:
      conesColor = (conesColor + 1) % 308
    if conesColor == 0:
      conesColor = 300
    print("conesColor= ", conesColor)
    fModelNeedleLength_mm = 187
    fModelSegmentLength_mm = fModelNeedleLength_mm / 20.0
    fK = 2 * np.pi * 2050 / (1000.0)

    if whetherfeedback:
      WrongIndex = WrongPosition[0]
      NumberOfRelatedIndexs = len(WrongPosition) - 1
      RelatedIndex = [WrongPosition[k + 1] for k in range(NumberOfRelatedIndexs)]
      ijkA = ControlPointsPackageIJK[WrongIndex][0]
      NeighbourAttenuation = 0
    else:
      ijkA = ijkAori

    def norm(fv):
      return np.sqrt(np.dot(fv, fv))

    def normalized(fv):
      fLen = np.sqrt(np.dot(fv, fv))
      return fv / fLen, fLen

    def defLocalNeedleCooSys(fvX, fvY, fvZ, iStep, rasA, rasDeflectionDirection, rasNeedlePlaneNormal, bDraw=False):
      rasSegmentVector = rasA - rasAPrev
      fLenSV = norm(rasSegmentVector)
      rasSegmentVector1 = rasSegmentVector / fLenSV
      if not rasNeedlePlaneNormal.any():
        rasNeedlePlaneNormal = np.cross(rasSegmentVector1, fvZ)
      if np.arccos(np.dot(rasNeedlePlaneNormal, fvX)) > np.pi / 2:
        rasNeedlePlaneNormal *= -1
      if not sum(rasNeedlePlaneNormal):
        rasNeedlePlaneNormal = np.cross(fvY, fvZ)
      fLenNPN = norm(rasNeedlePlaneNormal)
      rasNeedlePlaneNormal1 = rasNeedlePlaneNormal / fLenNPN
      if not rasDeflectionDirection.any():
        rasDeflectionDirection = +np.cross(fvZ, rasNeedlePlaneNormal1)
      if np.arccos(np.dot(rasDeflectionDirection, fvY)) > np.pi / 2:
        rasDeflectionDirection *= -1
      fLenDD = norm(rasDeflectionDirection)
      rasDeflectionDirection1 = rasDeflectionDirection / fLenDD
      fvZ = +np.cross(rasNeedlePlaneNormal1, rasDeflectionDirection1)
      fvZ /= norm(fvZ)
      return rasDeflectionDirection1, rasNeedlePlaneNormal1, fvZ

    def initNeedleModel(fAngleRefNeedle_rad, fEstNeedleLength_mm, fvZ, rasA, bDraw=False):
      rasAPrev2A = rasA - rasAPrev
      rasAPrev2A /= norm(rasAPrev2A)
      fAngleLongSegVZ_rad = np.arccos(np.dot(fvZ, rasAPrev2A))
      fEstDeflection_mm = fEstNeedleLength_mm * 1 / np.cos(fAngleRefNeedle_rad) * np.sin(fAngleLongSegVZ_rad)
      faArcLen_mm = list(zip(*matArcLen_mm))[0]
      ixArcLenColMin = find_nearest(faArcLen_mm, fEstNeedleLength_mm)[0]
      faYDefl_mm = matYDefl_mm[ixArcLenColMin]
      ixYDeflRowMin = find_nearest(faYDefl_mm, fEstDeflection_mm)[0]
      fEstF0_mmN = matFs_mN[ixArcLenColMin][ixYDeflRowMin] / 1000
      fEstEndSegAngle_rad = matEndSegAngles_rad[ixArcLenColMin][ixYDeflRowMin]
      fSumAngle_rad = fEstEndSegAngle_rad
      fF_mmN = fEstF0_mmN
      return fF_mmN, fSumAngle_rad

    def runNeedleModel(fStepAngle_rad, fF_mmN, fStepSize_mm, fvZ, iModelIt, rasDeflectionDirection1, fSumAngle_rad, rasB, bDraw=False):
      rasModelStepVector = rasDeflectionDirection1 * np.sin(fSumAngle_rad) + fvZ * np.cos(fSumAngle_rad)
      rasEstStepVector1 = rasModelStepVector / norm(rasModelStepVector)
      rasBDef = rasB
      rasB = rasA + rasEstStepVector1 * fStepSize_mm
      fDistBDef2BMod = norm(rasBDef - rasB)
      fMaxDistBDef2BMod_mm = 0
      if fDistBDef2BMod > fMaxDistBDef2BMod_mm:
        rasBDef2BMod = rasB - rasBDef
        rasBDef2BMod1, fDistBDef2BMod_mm = normalized(rasBDef2BMod)
        rasB = rasBDef + rasBDef2BMod1 * min(fMaxDistBDef2BMod_mm, fDistBDef2BMod_mm / 2.0)
      fAngleMultiSegments_rad = 0
      if fStepSize_mm / fModelSegmentLength_mm < 1:
        iFineStep = 5
        print("# fine steps ")
      else:
        iFineStep = 1
        print("normal steps")
      for j in range(0, int(round(fStepSize_mm / (fModelSegmentLength_mm / iFineStep)))):
        fF_mmN = fF_mmN / np.cos(fSumAngle_rad)
        fStepAngle_rad = fF_mmN / (fK * iFineStep)
        fSumAngle_rad -= fStepAngle_rad
        fSumAngle_deg = np.degrees(fSumAngle_rad)
        if fSumAngle_rad < 0:
          fStepAngle_rad = fSumAngle_rad = 0
        fAngleMultiSegments_rad += fStepAngle_rad
        iModelIt += 1
      fAngleMultiSegments_deg = np.degrees(fAngleMultiSegments_rad)
      return fF_mmN, fAngleMultiSegments_deg, rasEstStepVector1, fSumAngle_deg, rasB

    def find_nearest(t, x):
      from bisect import bisect_left
      i = bisect_left(t, x)
      if i == len(t):
        i -= 1
      if t[i] - x > 0.5:
        i -= 1
      return i, t[i]

    ijkAPrev = rasCBestPrev = ijkCBest = ijkCBestPrev = None
    rasA2CBestInit = []
    rasA2CBest = rasCBestPrev2CBest = rasCBestPrev2CBestPrev1 = np.array([])
    ijk = [0, 0, 0]
    fLabel = None
    if not bScript:
      widget.createAddOrSelectLabelMapNode(bScript)

    if iAxialSegmentationLimit is not None:
      fEstNeedleLength_mm = abs(ijkA[2] - iAxialSegmentationLimit) * 1.15 * fvSpacing[2]
    else:
      fEstNeedleLength_mm = ijkA[2] * 0.9 * fvSpacing[2]

    fAngleRefNeedle_rad = np.deg2rad(22.5)
    fvX = np.array([1, 0, 0])
    fvY = -np.array([0, np.cos(fAngleRefNeedle_rad), np.sin(fAngleRefNeedle_rad)])
    fvZ = np.array([0, np.sin(fAngleRefNeedle_rad), -np.cos(fAngleRefNeedle_rad)])
    fSumAngle_rad = fStepAngle_rad = fAngleA2CBestVZ_rad = None
    rasA = np.array(self.ijk2ras(ijkA))
    rasA0 = rasA
    ijkA0 = ijkA
    nStepsNeedle = nPointsPerNeedle - 1
    iZDirectionSign = 1 if bUp else -1

    ivDims = [0, 0, 0]
    imgData.GetDimensions(ivDims)

    if not bUp:
      self.controlPoints = []
    lvControlPointsRAS = []
    lvControlPointsIJK = []
    lvBestControlPoints = []
    rasNeedlePlaneNormal = rasDeflectionDirection = np.zeros(3)

    lvControlPointsRAS.append(self.ijk2ras(ijkA))
    lvControlPointsIJK.append(ijkA)
    lvBestControlPoints.append(self.ijk2ras(ijkA))
    print("#-----------------------------------------------------------")
    fStepsSum_mm = 0
    iModelIt = 0
    iStart = -1
    iRadiusNeedle = int(round(iRadiusNeedle_mm / float(fvSpacing[0])))
    iRadiusNeedleCorner = int(round((iRadiusNeedle_mm / float(fvSpacing[0]) / 1.414)))

    for iAbsStep, iStep in enumerate(range(iStart, nStepsNeedle)):
      print("#--------------------------------- %s: iStart: %d / iStep: %d / b,c%d----------------" % (strManualName, iStart, iStep, iStep + 1))
      fStepSize_mm = fEstNeedleLength_mm / (nStepsNeedle)
      if iStep > -1:
        fStepsSum_mm += fStepSize_mm
      print("fStepSize_mm, fStepsSum_mm, fEstNeedleLength_mm= %f, %f, %f" % (fStepSize_mm, fStepsSum_mm, fEstNeedleLength_mm))

      if iStep < 0:
        fFac = 1.0 / (2 * abs(iStart)) * (iAbsStep + 1)
        fStepSizeInit_mm = fFac * fEstNeedleLength_mm
        rasEstStepVector1 = np.array([fvZ[0], fvZ[1], fvZ[2]])
        rasB = rasA + rasEstStepVector1 * fStepSizeInit_mm
        ijkB = self.ras2ijk(rasB)
        iRMax = max(fStepSizeInit_mm, iRadiusMax_mm / float(fvSpacing[0]))
        nRIter = max(15, min(20, int(iRMax / fvSpacing[0])))
        nTIter = fStepSizeInit_mm
        print("iRMax, nRIter, nTIter= %d, %d, %d" % (iRMax, nRIter, nTIter))
      elif iStep == 0:
        ijkB = [ijkA[0], ijkA[1], ijkA[2] + iZDirectionSign * fStepSize_mm / fvSpacing[2]]
        rasB = self.ijk2ras(ijkB)
        ijkEstStepVector = np.array([0, 0, iZDirectionSign * fStepSize_mm / fvSpacing[2]])

        if len(rasA2CBestInit) == 0:
          rasA2CBestInit.append(fvZ)
        rasInitStepVector = sum(rasA2CBestInit) / len(rasA2CBestInit)
        rasInitStepVector1, dum = normalized(rasInitStepVector)
        print("rasInitStepVector1= ", rasInitStepVector1)
        rasB = rasA + rasInitStepVector1 * fStepSize_mm
        ijkB = self.ras2ijk(rasB)
        rasEstStepVector = self.ijk2ras(ijkEstStepVector)
        rasEstStepVector1, dum = normalized(rasEstStepVector)
        print("ijkB= ", ijkB, "# line: ", lineno())
        iRMax = iRadiusMax_mm / fvSpacing[0]
        nRIter = iRMax
        nTIter = int(round(fStepSize_mm))
        print("iRMax, nRIter, nTIter= %d, %d, %d" % (iRMax, nRIter, nTIter))
        if iStart < 0 and iStep == 0:
          fF_mmN, fSumAngle_rad = initNeedleModel(fAngleRefNeedle_rad, fEstNeedleLength_mm, fvZ, rasA)
      else:
        ijkB = [
            2 * ijkA[0] - ijkAPrev[0],
            2 * ijkA[1] - ijkAPrev[1],
            ijkA[2] + iZDirectionSign * fStepSize_mm / fvSpacing[2],
        ]

        rasB = self.ijk2ras(ijkB)
        print("default_ijkB= ", ijkB)
        iRMax = max(fStepSize_mm, iRadiusMax_mm / fvSpacing[0])
        nRIter = max(15, min(20, int(iRMax / fvSpacing[0])))
        nTIter = fStepSize_mm
        print("iRMax, nRIter, nTIter= %d, %d, %d" % (iRMax, nRIter, nTIter))

        rasDeflectionDirection1, rasNeedlePlaneNormal1, fvZ = defLocalNeedleCooSys(fvX, fvY, fvZ, iStep, rasA, rasDeflectionDirection, rasNeedlePlaneNormal, bDraw=True)
        if iStart == 0 and iStep == 1:
          fF_mmN, fSumAngle_rad = initNeedleModel(fAngleRefNeedle_rad, fEstNeedleLength_mm, fvZ, rasA, bDraw=True)
        if (iStart == 0 and iStep >= 1) or (iStart < 0 and iStep >= 0):
          fF_mmN, fAngleMultiSegments_deg, rasEstStepVector1, fSumAngle_deg, rasB = runNeedleModel(fStepAngle_rad, fF_mmN, fStepSize_mm, fvZ, iModelIt, rasDeflectionDirection1, fSumAngle_rad, rasB, bDraw=True)
        ijkB = self.ras2ijk(rasB)

      if ijkCBest:
        ijkCBestPrev = ijkCBest
      ijkCBest = [np.nan, np.nan, np.nan]
      print("# searching cone:")
      fMinTotal = np.inf
      rust_cone_res = (
          rust_search_cone_15_1(
              imgData,
              imgLabelData,
              ijkA,
              ijkB,
              ijkAPrev if iStep >= 2 else None,
              float(iRMax),
              float(nRIter),
              int(nRotatingIts),
              float(nTIter),
              int(bGradient),
              int(iRadiusNeedle),
              int(iRadiusNeedleCorner),
              float(iGradientPonderation),
              1 if (bGaussianAttenuation == 1 and iStep >= 2) else 0,
              float(iSigmaValue),
              whether_feedback=bool(whetherfeedback),
              control_points_package=ControlPointsPackage if whetherfeedback else None,
              related_indices=RelatedIndex if whetherfeedback else None,
              ijk_to_ras_mat=_get_ijk_to_ras_16() if whetherfeedback else None,
              radius_needle_mm=float(iRadiusNeedle_mm),
              punish=float(Punish) if Punish is not None else 0.0,
          )
          if not conesColor
          else None
      )
      if rust_cone_res is not None:
        ijkCBest, fMinTotal, fTotal = rust_cone_res
        if not np.isnan(ijkCBest[0]):
          rasCBest = self.ijk2ras(ijkCBest)
      else:
        for iR in range(int(nRIter) + 1):
          fR = iR * (iRMax / float(nRIter))
          for iThetaStep in range(int(nRotatingIts)):
            fAngle_deg = (iThetaStep * 360.0) / float(nRotatingIts)
            fThetaStep_rad = math.radians(fAngle_deg)

            ijkC = [
                ijkB[0] + fR * (math.cos(fThetaStep_rad)),
                ijkB[1] + fR * (math.sin(fThetaStep_rad)),
                ijkB[2],
            ]

            fTotal = 0
            lijkM = [[0, 0, 0] for i in range(int(nTIter) + 1)]

            fLabel = 0
            for iTStep in range(int(nTIter) + 1):
              fTStep = iTStep / float(nTIter)
              for i in range(3):
                lijkM[iTStep][i] = (1 - fTStep) * ijkA[i] + fTStep * ijkC[i]
                ijk[i] = int(round(lijkM[iTStep][i]))

              ix, iy, iz = ijk[0], ijk[1], ijk[2]
              if 0 < ix < ivDims[0] and 0 < iy < ivDims[1] and 0 < iz < ivDims[2]:
                dCenter = get_scalar(ix, iy, iz)
                fTotal += dCenter

                if whetherfeedback:
                  NeighbourAttenuation = self.Feedback(ControlPointsPackage, Punish, iRadiusNeedle_mm, NumberOfRelatedIndexs, len, RelatedIndex, range, lijkM[iTStep])
                  fTotal += NeighbourAttenuation

                if bGradient == 1:
                  g1 = get_scalar(ix + iRadiusNeedle, iy, iz)
                  g2 = get_scalar(ix - iRadiusNeedle, iy, iz)
                  g3 = get_scalar(ix, iy + iRadiusNeedle, iz)
                  g4 = get_scalar(ix, iy - iRadiusNeedle, iz)
                  g5 = get_scalar(ix + iRadiusNeedleCorner, iy + iRadiusNeedleCorner, iz)
                  g6 = get_scalar(ix - iRadiusNeedleCorner, iy - iRadiusNeedleCorner, iz)
                  g7 = get_scalar(ix - iRadiusNeedleCorner, iy + iRadiusNeedleCorner, iz)
                  g8 = get_scalar(ix + iRadiusNeedleCorner, iy - iRadiusNeedleCorner, iz)

                  fTotal += 8 * dCenter - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * iGradientPonderation

                if imgLabelData:
                  fLabel = imgLabelData.GetScalarComponentAsFloat(ix, iy, iz, 0)
                  if fLabel and fLabel < 300:
                    fTotal -= 10000
                  if conesColor:
                    imgLabelData.SetScalarComponentFromFloat(ix, iy, iz, 0, conesColor)

            if bGaussianAttenuation == 1 and iStep >= 2:
              if ijkAPrev[2] - ijkA[2] != 0:
                fK_att = (ijkA[2] - ijkB[2]) / float(ijkAPrev[2] - ijkA[2])
                ijkX = [
                    ijkA[0] + fK_att * (ijkA[0] - ijkAPrev[0]),
                    ijkA[1] + fK_att * (ijkA[1] - ijkAPrev[1]),
                    ijkA[2] + fK_att * (ijkA[2] - ijkAPrev[2]),
                ]
                fRgauss = ((ijkC[0] - ijkX[0]) ** 2 + (ijkC[1] - ijkX[1]) ** 2 + (ijkC[2] - ijkX[2]) ** 2) ** 0.5
                fGaussianAttenuation = math.exp(-(fRgauss / float(iRMax)) ** 2 / float((2 * (iSigmaValue / 10.0)) ** 2))
                if fGaussianAttenuation < 0 or fGaussianAttenuation > 1:
                  print("fGaussianAttenuation= ", fGaussianAttenuation, " # /!\\ ")
                fTotal = fTotal * fGaussianAttenuation

            if fTotal < fMinTotal:
              fMinTotal = fTotal
              ijkCBest = ijkC
              rasCBest = self.ijk2ras(ijkCBest)

      if ijkCBest == [np.nan, np.nan, np.nan]:
        breakbox("/!!\\ no ijkCBest found")

      ijkAPrev = ijkA
      rasAPrev = np.array(self.ijk2ras(ijkAPrev))

      print("ijkCBest= ", ijkCBest, '# line: ', lineno())
      rasB2CBest = rasCBest - np.array(rasB)
      fDistB2CBest_mm = norm(rasB2CBest)
      print("fDistB2CBest_mm= ", fDistB2CBest_mm)
      if ijkCBestPrev:
        rasCBestPrev = np.array(self.ijk2ras(ijkCBestPrev))
        if rasCBestPrev2CBest.any():
          rasCBestPrev2CBestPrev = rasCBestPrev2CBest
          rasCBestPrev2CBestPrev1 = rasCBestPrev2CBestPrev / norm(rasCBestPrev2CBestPrev)
        rasCBestPrev2CBest = rasCBest - rasCBestPrev
      rasA = np.array(self.ijk2ras(ijkA))
      rasA2CBest = rasCBest - rasA
      if (iStart < 0 and iStep < 0) or (iStart == 0 and iStep == 0):
        fDistB2CBest_mm = -np.inf
        fLimitCrit4 = fMaxDistB2CBest_mm = np.inf
      if ((iStart == 0 and iStep >= 1) or (iStart < 0 and iStep >= 0)) and ijkCBest != [np.nan, np.nan, np.nan]:
        fMaxDistB2CBest_mm = 1
        fLimitCrit4 = fMaxDistB2CBest_mm
      bReject = not fDistB2CBest_mm < fLimitCrit4
      if bReject:
        print("#^^^^^^^^^^^^^ crit. rejects best cone point")
      if ijkCBest == [np.nan, np.nan, np.nan] or bReject:
        if rasB2CBest.any():
          rasB2CBest1, dum = normalized(rasB2CBest)
        else:
          rasB2CBest1 = np.array([0, 0, 0])
        rasA = rasB + rasB2CBest1 * min(fMaxDistB2CBest_mm, fDistB2CBest_mm / 2.0)
        ijkA = self.ras2ijk(rasA)
        print('#middle vector, but within constraint: rasA= ', rasA)
        print("using_point_near_ijkB= ", ijkA)
        if iStep == 0 - abs(iStart):
          breakbox("/!!\\ not using CBest in first it.??")
      elif ijkCBest != ijkAPrev:
        if iStep <= -1:
          print("# iStep -1")
          if iStep == -1:
            rasA2CBestInit.append(rasA2CBest)
            rasAPrev = rasA0 - rasA2CBest
            ijkAPrev = self.ras2ijk(rasAPrev)
            ijkA = ijkA0
            rasA = self.ijk2ras(ijkA)
          elif iStep <= -2:
            rasA2CBestInit.append(rasA2CBest)
          continue
        ijkA = ijkCBest
        rasA = self.ijk2ras(ijkA)
        rasNeedlePlaneNormal = rasDeflectionDirection = np.zeros(3)
        print("using_ijkC= ", ijkCBest)
        print("best_fEstimator_value= ", fTotal)
      else:
        breakbox("/!!| uncaught ijkCBest==ijkAPrev")

      if ijkA[2] < iAxialSegmentationLimit and ijkA != ijkA0 and (ijkAPrev[2] - ijkA[2]) != 0:
        asl = iAxialSegmentationLimit
        l = (ijkA[2] - asl) / float(ijkAPrev[2] - ijkA[2])
        ijkA = [
            ijkA[0] - l * (ijkAPrev[0] - ijkA[0]),
            ijkA[1] - l * (ijkAPrev[1] - ijkA[1]),
            ijkA[2] - l * (ijkAPrev[2] - ijkA[2]),
        ]

      if iStep >= 0:
        lvControlPointsRAS.append(self.ijk2ras(ijkA))
        lvControlPointsIJK.append(ijkA)
        if ijkA[2] <= iAxialSegmentationLimit and ijkA != ijkA0:
          print("# /!\\ needle too long, break")
          break

    for i in range(len(lvControlPointsRAS)):
      self.controlPoints.append(lvControlPointsRAS[i])
    if not bAutoStopTip:
      if bDrawNeedle:
        self.addNeedleToScene(lvControlPointsRAS, (205) % MAXCOL, 'Detection', bScript, 0, manualName=strManualName)
      self.controlPoints = []
    if bAutoStopTip and bUp:
      if bDrawNeedle:
        self.addNeedleToScene(self.controlPoints, (205) % MAXCOL, 'Detection', bScript, 0, manualName=strManualName)
      self.controlPoints = []
    print(time.perf_counter() - t0, "seconds process time")
    if bUp:
      return lvControlPointsIJK[-1]
    return lvControlPointsRAS, lvControlPointsIJK

  def Feedback(self, ControlPointsPackage, Punish, iRadiusNeedle_mm, NumberOfRelatedIndexs, len, RelatedIndex, range, ijk):
    """Feedback attenuation term for rationality-checked needles."""
    ras = self.ijk2ras(ijk)
    NeighbourAttenuation = 0
    for w in range(NumberOfRelatedIndexs):
      distance = 9999
      NumberOfPointsOnThatNeedle = len(ControlPointsPackage[RelatedIndex[w]])
      if ras[2] >= ControlPointsPackage[RelatedIndex[w]][0][2]:
        dz = ras[2] - ControlPointsPackage[RelatedIndex[w]][0][2]
        distance = (3 * (dz ** 2)) ** 0.5
      elif ras[2] <= ControlPointsPackage[RelatedIndex[w]][NumberOfPointsOnThatNeedle - 1][2]:
        dz = ras[2] - ControlPointsPackage[RelatedIndex[w]][NumberOfPointsOnThatNeedle - 1][2]
        distance = (3 * (dz ** 2)) ** 0.5
      else:
        for q in range(NumberOfPointsOnThatNeedle - 1):
          pointabove = ControlPointsPackage[RelatedIndex[w]][q]
          pointbelow = ControlPointsPackage[RelatedIndex[w]][q + 1]
          if pointabove[2] <= ras[2]:
            h = ((ras[0] - pointabove[0]) ** 2 + (ras[1] - pointabove[1]) ** 2 + (ras[2] - pointabove[2]) ** 2) ** 0.5
          elif pointbelow[2] >= ras[2]:
            h = ((ras[0] - pointbelow[0]) ** 2 + (ras[1] - pointbelow[1]) ** 2 + (ras[2] - pointbelow[2]) ** 2) ** 0.5
          else:
            a = ((pointabove[0] - ras[0]) ** 2 + (pointabove[1] - ras[1]) ** 2 + (pointabove[2] - ras[2]) ** 2) ** 0.5
            b = ((pointbelow[0] - ras[0]) ** 2 + (pointbelow[1] - ras[1]) ** 2 + (pointbelow[2] - ras[2]) ** 2) ** 0.5
            c = ((pointabove[0] - pointbelow[0]) ** 2 + (pointabove[1] - pointbelow[1]) ** 2 + (pointabove[2] - pointbelow[2]) ** 2) ** 0.5
            hc = (a + b + c) / 2
            h = (hc * (hc - a) * (hc - b) * (hc - c)) ** 0.5 * 2 / c
          if h < distance:
            distance = h

      if distance == 0:
        NeighbourAttenuation += 3000
      elif distance >= iRadiusNeedle_mm * Punish:
        NeighbourAttenuation += 0
      else:
        NeighbourAttenuation += 1000 * (iRadiusNeedle_mm / distance)

    return NeighbourAttenuation

  def needleDetectionUPThread(self, A, imageData, colorVar, spacing, script=False, strName="", tipOnly=False):
    """Call different versions of needle detection up thread."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.axialSegmentationLimit, widget.axialSegmentationLimitRAS = self.findAxialSegmentationLimitFromMarker()
    if widget.algoVersParameter.value == 0:
      return self.needleDetectionUPThreadCurrentDev(A, imageData, colorVar, spacing, script=script, tipOnly=tipOnly, manualName=strName)
    if widget.algoVersParameter.value == 1:
      msgbox("/!\\ needleDetectionUPThread N/A for algoVersion 1")
    if widget.algoVersParameter.value == 2:
      msgbox("/!\\ needleDetectionUPThread N/A for algoVersion 2")
    if widget.algoVersParameter.value in (3, 4):
      label_img = widget.labelMapNode.GetImageData() if widget.labelMapNode else None
      return self.needleDetectionThread15_1(A, imageData, label_img, widget.tempPointList, colorVar, spacing, bUp=True, bScript=script, strManualName=strName)
    msgbox("/!\\ needleDetectionUPThread not defined")

  def needleDetectionUPThreadCurrentDev(self, A, imageData, colorVar, spacing, script=False, tipOnly=False, manualName=""):
    """Track needle in +Z direction using dichotomy to locate the needle tip."""
    profprint()
    get_scalar = _ensure_scalar_getter(imageData)
    widget = slicer.modules.NeedleFinderWidget

    ijk = [0, 0, 0]
    bestPoint = [0, 0, 0]

    distanceMax = widget.radiusMax.value
    gradientPonderation = widget.gradientPonderation.value
    sigmaValue = widget.sigmaValue.value
    stepsize = widget.stepsize.value
    gaussianAttenuationChecked = widget.gaussianAttenuationButton.isChecked()
    lookNeighborhood = widget.gradient.isChecked()
    numberOfPointsPerNeedle = max(1, widget.numberOfPointsPerNeedle.value - 1)
    nbRotatingIterations = widget.nbRotatingIterations.value
    radiusNeedleParameter = widget.radiusNeedleParameter.value
    axialSegmentationLimit = widget.axialSegmentationLimit
    lenghtNeedleParameter = widget.lenghtNeedleParameter.value / (spacing[2])

    if axialSegmentationLimit is not None:
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    elif axialSegmentationLimit is None and widget.maxLength.isChecked():
      axialSegmentationLimit = 0
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    else:
      lenghtNeedle = lenghtNeedleParameter

    rMax = distanceMax / float(spacing[0])
    NbStepsNeedle = numberOfPointsPerNeedle - 1
    nbRotatingStep = nbRotatingIterations

    dims = [0, 0, 0]
    imageData.GetDimensions(dims)

    A0 = A
    controlPointsUP = [self.ijk2ras(A)]
    controlPointsIJK = [A]
    bestControlPoints = [self.ijk2ras(A)]
    minEstimator0 = None
    stopTracking = 0

    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))

    for step in range(0, NbStepsNeedle + 1):
      if step == 0:
        stepSize = self.stepSize(step, NbStepsNeedle) * lenghtNeedle
        Vx = 0
        Vy = 0
        Vz = stepSize
        C0 = [A[0], A[1], A[2] + stepSize]
        rMax = stepSize / float(spacing[0])
        rIter = int(round(rMax))
        tIter = max(1, int(round(stepSize)))
      else:
        stepSize = self.stepSize(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        Vx = A[0] - tip0[0]
        Vy = A[1] - tip0[1]
        Vz = A[2] - tip0[2]

      coeffSize = abs(stepSize) / spacing[2]
      if Vz != 0:
        K = coeffSize / float(abs(Vz))
      else:
        break

      P0 = A
      P1 = [
          P0[0] + K * Vx,
          P0[1] + K * Vy,
          P0[2] + K * Vz,
      ]

      dichoStep = 0
      continueDichotomy = 1
      testSuccess = None
      while continueDichotomy == 1 and dichoStep <= 7 and stopTracking == 0:
        estimator = 0
        minEstimator = 0
        dichoStep += 1
        if step != 100:
          if dichoStep > 1:
            K /= 2.0
            stepSize /= 2.0
            if testSuccess == 1:
              P0 = P1
            P1 = [
                P0[0] + K * Vx,
                P0[1] + K * Vy,
                P0[2] + K * Vz,
            ]
          C0 = P1
          rMax = stepSize / float(spacing[0])
          rIter = max(15, min(20, int(rMax / float(spacing[0]))))
          tIter = max(3, int(round(stepSize)))

        for R in range(rIter + 1):
          r = R * (rMax / float(rIter))
          for thetaStep in range(nbRotatingStep):
            angleInDegree = (thetaStep * 360) / float(nbRotatingStep)
            theta = math.radians(angleInDegree)

            C = [
                C0[0] + r * (math.cos(theta)),
                C0[1] + r * (math.sin(theta)),
                C0[2],
            ]

            total = 0
            M = [[0, 0, 0] for i in range(int(tIter) + 1)]

            for t in range(tIter + 1):
              tt = t / float(tIter)
              for i in range(3):
                M[t][i] = (1 - tt) * A[i] + tt * C[i]
                ijk[i] = int(round(M[t][i]))

              ix, iy, iz = ijk[0], ijk[1], ijk[2]
              if 0 < ix < dims[0] and 0 < iy < dims[1] and 0 < iz < dims[2]:
                center = get_scalar(ix, iy, iz)
                total += center / float(tIter)
                if lookNeighborhood == 1:
                  g1 = get_scalar(ix + radiusNeedle, iy, iz)
                  g2 = get_scalar(ix - radiusNeedle, iy, iz)
                  g3 = get_scalar(ix, iy + radiusNeedle, iz)
                  g4 = get_scalar(ix, iy - radiusNeedle, iz)
                  g5 = get_scalar(ix + radiusNeedleCorner, iy + radiusNeedleCorner, iz)
                  g6 = get_scalar(ix - radiusNeedleCorner, iy - radiusNeedleCorner, iz)
                  g7 = get_scalar(ix - radiusNeedleCorner, iy + radiusNeedleCorner, iz)
                  g8 = get_scalar(ix + radiusNeedleCorner, iy - radiusNeedleCorner, iz)

                  total += (8 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8.0) * gradientPonderation) / float(tIter)

            if R == 0:
              initialIntensity = total
              estimator = total

            if gaussianAttenuationChecked == 1 and step >= 1:
              if Vz != 0:
                rgauss = ((C[0] - C0[0]) ** 2 + (C[1] - C0[1]) ** 2 + (C[2] - C0[2]) ** 2) ** 0.5
                gaussianAttenuation = math.exp(-(rgauss / float(rMax)) ** 2 / float((2 * (sigmaValue / 10.0)) ** 2))
                estimator = total * gaussianAttenuation
              else:
                estimator = total
            else:
              estimator = total

            if estimator < initialIntensity:
              if estimator < minEstimator or minEstimator == 0:
                minEstimator = estimator
                if minEstimator != 0:
                  bestPoint = C

        if step == 0:
          minEstimator0 = self.estimatorReference

        if minEstimator0 is not None and minEstimator0 != 0:
          RelMinEstimator = abs((minEstimator - minEstimator0) / float(minEstimator0))
          valCtrPt = self.objectiveFunction(get_scalar, A, 3, spacing, 1)

          if dichoStep >= 8:
            stopTracking = 1
            bestPoint = A
            break
          if ((RelMinEstimator < 0.6) and valCtrPt < 0 and minEstimator0 < 0) or (minEstimator < -80 and valCtrPt < 0):
            testSuccess = 1
            if dichoStep > 1:
              continueDichotomy = 0
              stopTracking = 1
              break
            elif dichoStep == 1:
              continueDichotomy = 0
          else:
            testSuccess = 0

      minEstimator0 = minEstimator
      tip0 = A
      if bestPoint == [0, 0, 0]:
        A = C0
      elif bestPoint != tip0:
        A = bestPoint

      controlPointsUP.append(self.ijk2ras(A))
      controlPointsIJK.append(A)

      if widget.drawFiducialPoints.isChecked():
        create_fiducial_node(
            name='..' + str(step) + '.' + str(self.objectiveFunction(get_scalar, A, 3, spacing, 1)) + '_' + str(colorVar),
            ras=controlPointsUP[step + 1],
            color=(1.0, 1.0, 0.0),
        )

      if A[2] <= axialSegmentationLimit and A != A0:
        break

    for i in range(len(controlPointsUP)):
      self.controlPoints.append(controlPointsUP[i])
    if not tipOnly:
      self.addNeedleToScene(self.controlPoints, colorVar, 'Detection', script, manualName=manualName)
    return controlPointsIJK[-1]

  def drawObturatorNeedles(self):
    """Draw straight obturator needles of length realNeedleLength."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    realNeedleLength = widget.realNeedleLength.value

    self.table = None
    self.row = 0
    widget.initTableView()
    while slicer.util.getNodes('obturator-seg*') != {}:
      nodes = slicer.util.getNodes('obturator-seg*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)

    if widget.obtuNeedleValueCtrPt == [[[]]]:
      widget.obtuNeedleValueCtrPt = [[[999, 999, 999] for i in range(10)] for j in range(10)]
    if widget.obtuNeedlePt == [[[]]]:
      widget.obtuNeedlePt = [[[999, 999, 999] for i in range(10)] for j in range(10)]

    for modelNode in get_all_fiducial_nodes():
      if modelNode.GetAttribute("ObturatorNeedle") == "1":
        needleNumber = int(modelNode.GetAttribute("NeedleNumber"))
        needleStep = int(modelNode.GetAttribute("NeedleStep"))
        coord = [0.0, 0.0, 0.0]
        get_fiducial_coords(modelNode, coord)
        widget.obtuNeedleValueCtrPt[needleNumber][needleStep] = coord

    for i in range(0, len(widget.obtuNeedleValueCtrPt)):
      sign = cmp(widget.obtuNeedleValueCtrPt[0][0][2], widget.obtuNeedleValueCtrPt[0][1][2])
      if i == 0 and sign <= -1:
        AA = widget.obtuNeedleValueCtrPt[0][0][2]
        BB = widget.obtuNeedleValueCtrPt[0][1][2]
        widget.obtuNeedleValueCtrPt[0][0][2] = BB
        widget.obtuNeedleValueCtrPt[0][1][2] = AA

      Vx = widget.obtuNeedleValueCtrPt[0][1][0] - widget.obtuNeedleValueCtrPt[0][0][0]
      Vy = widget.obtuNeedleValueCtrPt[0][1][1] - widget.obtuNeedleValueCtrPt[0][0][1]
      Vz = widget.obtuNeedleValueCtrPt[0][1][2] - widget.obtuNeedleValueCtrPt[0][0][2]

      L = float(Vx ** 2 + Vy ** 2 + Vz ** 2) ** 0.5
      if L:
        E = widget.obtuNeedleValueCtrPt[i][0]
        Ex = E[0] + realNeedleLength * Vx / L
        Ey = E[1] + realNeedleLength * Vy / L
        Ez = E[2] + realNeedleLength * Vz / L
        widget.obtuNeedlePt[i][1] = [Ex, Ey, Ez]
        widget.obtuNeedlePt[i][0] = [E[0], E[1], E[2]]

    for i in range(len(widget.obtuNeedlePt)):
      if widget.obtuNeedlePt[i][0][0] != 999:
        controlPoints = [val for val in widget.obtuNeedlePt[i] if val != [999, 999, 999]]
        if (i == 0 and len(controlPoints) >= 1) or i >= 1:
          self.addNeedleToScene(controlPoints, i, 'Obturator')

  def drawValidationNeedles(self, nb=None):
    """Fit a Bezier curve to each manually segmented validation needle's control points and render it as a tube."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    if nb:
      widget.editNeedleTxtBox.value = nb
    widget.initTableView()
    self.deleteEvaluationNeedlesFromTable()
    while slicer.util.getNodes('manual-seg_' + str(widget.editNeedleTxtBox.value)) != {}:
      nodes = slicer.util.getNodes('manual-seg_' + str(widget.editNeedleTxtBox.value))
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)

    tableValueCtrPt = [[[999, 999, 999] for i in range(100)] for j in range(100)]
    for modelNode in get_all_fiducial_nodes():
      modelNodeName = modelNode.GetName().strip('.').strip('\r')
      parts = modelNodeName.split('-')
      if modelNode.GetName().startswith('.') and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        needleNumber = int(parts[0])
        needleStep = int(parts[1])
        if needleNumber == widget.editNeedleTxtBox.value:
          coord = [0.0, 0.0, 0.0]
          get_fiducial_coords(modelNode, coord)
          tableValueCtrPt[needleNumber][needleStep] = coord
          print(needleNumber, needleStep, coord)

    for i in range(len(tableValueCtrPt)):
      if not all(e == [999, 999, 999] for e in tableValueCtrPt[i]):
        controlPointsUnsorted = [val for val in tableValueCtrPt[i] if val != [999, 999, 999]]
        controlPoints = self.sortTable(controlPointsUnsorted, (2, 1, 0))
        print("Control points", controlPoints)
        self.addNeedleToScene(controlPoints, i, 'Validation')
        self.observeManualNeedles()
    self.findAxialSegmentationLimitFromMarker(bForceFallback=True)

  def addCSplineToScene(self, controlPoint, colorVar, needleType='Detection', endMarker=False, name="^", script=False, manualName=""):
    """Adds visual needle representation as interpolating cardinal spline to the scene."""
    widget = slicer.modules.NeedleFinderWidget
    realNeedleLength = widget.realNeedleLength.value
    extendNeedle = widget.extendNeedle.isChecked()

    controlPointListSorted = list(controlPoint)
    lenghtTotal = 0
    for i in range(len(controlPoint) - 1):
      lenghtTotal += self.distanceTwoPoints(controlPointListSorted[i + 1], controlPointListSorted[i])

    if lenghtTotal < realNeedleLength and extendNeedle:
      lastPoint = [controlPointListSorted[-1][0], controlPointListSorted[-1][1], controlPointListSorted[-1][2] - (realNeedleLength - lenghtTotal)]
      controlPointListSorted.append(lastPoint)

    scene = slicer.mrmlScene
    tubePolyData = build_cspline_polydata(controlPointListSorted, num_output_points=50, radius=1.0, sides=50)

    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetAndObservePolyData(tubePolyData)
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    scene.AddNode(model)
    model.GetDisplayNode().Visibility2DOn()
    if needleType == 'Validation':
      model.SetName('.manual-seg_' + str(colorVar))
    elif needleType == 'Obturator':
      model.SetName('.obturator-seg_' + str(colorVar))
    else:
      model.SetName('.auto-seg_' + str(self.round) + '-ID-' + str(model.GetID()) + '-' + manualName)
    model.SetAttribute('type', needleType)

    if needleType == 'Validation':
      nth = int(colorVar) % MAXCOL
    else:
      nth = extract_model_id_number(model.GetID()) % MAXCOL
    modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
    model.SetAttribute("nth", str(nth))

    if endMarker:
      create_fiducial_node(
          name='.' + name + '_' + str(colorVar),
          ras=controlPointListSorted[-1],
          color=(0.0, 0.0, 0.0),
          glyph_scale=2.1,
          text_scale=3.0,
          locked=True,
          attributes={'TemporaryFiducial': '1'},
      )

  def addPolyLineToScene(self, controlPoint, colorVar, needleType='Detection', script=False, endMarker=False, name="^", trans=0, manualName="", radius=1, bPause=False):
    """Adds visual representation of linear needle segments to the scene."""
    widget = slicer.modules.NeedleFinderWidget
    realNeedleLength = widget.realNeedleLength.value
    extendNeedle = widget.extendNeedle.isChecked()

    controlPointListSorted = list(controlPoint)
    lenghtTotal = 0
    for i in range(len(controlPoint) - 1):
      lenghtTotal += self.distanceTwoPoints(controlPointListSorted[i + 1], controlPointListSorted[i])

    if lenghtTotal < realNeedleLength and extendNeedle:
      lastPoint = [controlPointListSorted[-1][0], controlPointListSorted[-1][1], controlPointListSorted[-1][2] - (realNeedleLength - lenghtTotal)]
      controlPointListSorted.append(lastPoint)

    scene = slicer.mrmlScene
    polyData = build_polyline_polydata(controlPointListSorted)
    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetAndObservePolyData(build_tube_polydata(polyData, radius=radius, sides=50))
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    scene.AddNode(model)
    model.GetDisplayNode().Visibility2DOn()
    model.GetDisplayNode().SetOpacity(1 - trans)
    if needleType == 'Validation':
      model.SetName('.manual-seg_' + str(colorVar))
    elif needleType == 'Obturator':
      model.SetName('.obturator-seg_' + str(colorVar))
    else:
      model.SetName('.auto-seg_' + str(self.round) + '-ID-' + str(model.GetID()) + '-' + manualName)
    model.SetAttribute('type', needleType)

    if needleType in ('Validation', 'Debug'):
      nth = int(colorVar) % MAXCOL
    else:
      nth = extract_model_id_number(model.GetID()) % MAXCOL
    modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
    model.SetAttribute("nth", str(nth))

    if endMarker:
      create_fiducial_node(
          name='.' + name + '_' + str(colorVar),
          ras=controlPointListSorted[-1],
          color=(0.0, 0.0, 0.0),
          glyph_scale=0.5,
          text_scale=1.0,
          locked=True,
          attributes={'TemporaryFiducial': '1'},
      )
    if bPause:
      breakbox('Inspect this situation in the viewer!')

  def unique(self, a):
    """Remove duplicates in 2D array."""
    return unique_points(a)

  def addNeedleToScene(self, controlPoint, colorVar, needleType='Detection', script=False, trans=0, manualName="", bPause=False, comment="", confidence=None):
    """Computes the Bezier curve and adds visual representation of a needle to the scene."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget

    realNeedleLength = widget.realNeedleLength.value
    extendNeedle = widget.extendNeedle.isChecked()
    controlPoint = unique_points(np.array(controlPoint)).tolist()
    controlPointListSorted = self.sortTableReverse(controlPoint, (2, 1, 0))

    lenghtTotal = 0
    for i in range(len(controlPoint) - 1):
      lenghtTotal += self.distanceTwoPoints(controlPointListSorted[i + 1], controlPointListSorted[i])
    print("Needle tube length: ", lenghtTotal)

    if lenghtTotal < realNeedleLength and extendNeedle:
      lastPoint = [controlPointListSorted[-1][0], controlPointListSorted[-1][1], controlPointListSorted[-1][2] - (realNeedleLength - lenghtTotal)]
      controlPointListSorted.append(lastPoint)

    label = None
    scene = slicer.mrmlScene
    bezier_points = compute_bezier_curve(controlPointListSorted, num_eval_points=50)
    polyData = build_polyline_polydata(bezier_points)
    tubePolyData = build_tube_polydata(polyData, radius=1.0, sides=50)

    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetAndObservePolyData(tubePolyData)
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    scene.AddNode(model)
    model.GetDisplayNode().Visibility2DOn()
    model.GetDisplayNode().SetOpacity(1 - trans)
    if needleType == 'Validation':
      model.SetName('manual-seg_' + str(colorVar))
    elif needleType == 'Obturator':
      model.SetName('obturator-seg_' + str(colorVar))
    else:
      model.SetName('auto-seg_' + str(self.round) + '-ID-' + str(model.GetID()) + '-' + manualName)
    model.SetAttribute('type', needleType)

    self.lastNeedleNames.append(model.GetName())

    if needleType == 'Validation':
      nth = int(colorVar) % MAXCOL
    else:
      if widget.algoVersParameter.value < 3:
        nth = extract_model_id_number(model.GetID()) % MAXCOL
      else:
        nth = int(colorVar) % MAXCOL
    model.SetAttribute("nth", str(nth))
    if confidence is not None:
      model.SetAttribute("confidence", f"{float(confidence):.2f}")

    use_conf_color = (
        needleType != 'Validation'
        and confidence is not None
        and getattr(widget, "colorByConfidenceCheckBox", None) is not None
        and widget.colorByConfidenceCheckBox.isChecked()
    )
    if use_conf_color:
      r_f, g_f, b_f = self.confidenceToRGB(confidence)
      modelDisplay.SetColor(r_f, g_f, b_f)
    else:
      modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])

    if needleType != 'Validation' and not script:
      widget.initTableView()
      model.SetAttribute("table_row", str(widget.model.rowCount()))
      self.addNeedleToTable(extract_model_id_number(model.GetID()), label, comment=comment, confidence=confidence)
    return model

  def estimateAxialLimitSlice(self, vol_arr: np.ndarray, sp_k: float) -> int:
    """Automatically estimate the lower axial template/insertion slice `k_lim` from 3D volume intensities."""
    from scipy.ndimage import gaussian_filter1d

    K, J, I = vol_arr.shape
    if K < 8:
      return 0
    c_roi = vol_arr[:, int(0.30 * J):int(0.70 * J), int(0.30 * I):int(0.70 * I)]
    p50 = gaussian_filter1d(np.percentile(c_roi, 50, axis=(1, 2)).astype(np.float64), sigma=max(1.0, 2.5 / max(0.5, float(sp_k))))
    if K <= 42:
      min_bot = float(np.min(p50[:max(2, int(0.25 * K))]))
      t_ref = float(np.percentile(p50[int(0.30 * K):max(int(0.30 * K) + 2, int(0.85 * K))], 65))
      if min_bot > 0.38 * t_ref:
        return max(0, int(round(3.0 / max(1.0, float(sp_k)))))
      k_lo, k_hi = int(0.18 * K), max(int(0.18 * K) + 2, int(0.70 * K))
      k_peak = k_lo + int(np.argmax(p50[k_lo:k_hi]))
      s_lo = max(1, k_peak - int(round(45.0 / max(0.5, float(sp_k)))))
    else:
      k_lo, k_hi = int(0.25 * K), max(int(0.25 * K) + 2, int(0.62 * K))
      k_peak = k_lo + int(np.argmax(p50[k_lo:k_hi]))
      s_lo = max(int(0.10 * K), k_peak - int(round(52.0 / max(0.5, float(sp_k)))))
    grad = np.gradient(p50)
    if k_peak <= s_lo:
      return max(0, int(0.25 * K))
    k_steep = s_lo + int(np.argmax(grad[s_lo:k_peak]))
    k_est = k_steep
    while k_est > s_lo and grad[k_est] > 0.22 * grad[k_steep]:
      k_est -= 1
    return int(np.clip(k_est, 0, K - 5))

  def _compute_3d_sym_response_numpy(
      self,
      vol_arr: np.ndarray,
      sp_x: float,
      sp_y: float,
      i0: int,
      i1: int,
      j0: int,
      j1: int,
      k0: int,
      k1: int,
      body_thresh: float,
  ):
    """Pure-NumPy/SciPy fallback for `rust_compute_3d_sym_response`."""
    from scipy.ndimage import gaussian_filter

    dim_z, dim_y, dim_x = vol_arr.shape
    out_resp = np.zeros((dim_z, dim_y, dim_x), dtype=np.float32)
    out_cs = np.zeros((dim_z, dim_y, dim_x), dtype=np.float32)
    out_rsym = np.zeros((dim_z, dim_y, dim_x), dtype=np.float32)

    sub = vol_arr[k0:k1, j0:j1, i0:i1].astype(np.float32)
    cs_sub = gaussian_filter(sub, sigma=(0.0, 0.85, 0.85))
    out_cs[k0:k1, j0:j1, i0:i1] = cs_sub

    K_s, J_s, I_s = cs_sub.shape
    m = 10
    if J_s <= 2 * m or I_s <= 2 * m:
      return out_resp, out_cs, out_rsym

    center_crop = cs_sub[:, m:J_s - m, m:I_s - m]
    best_sym = np.zeros_like(center_crop)
    best_med = np.zeros_like(center_crop)

    for r_mm in (1.35, 1.85, 2.5, 3.3):
      pair_mins = []
      all_12 = []
      for a_deg in (0, 30, 60, 90, 120, 150):
        th = math.radians(a_deg)
        di = int(round(r_mm * math.cos(th) / sp_x))
        dj = int(round(r_mm * math.sin(th) / sp_y))
        v_pos = cs_sub[:, m + dj:J_s - m + dj, m + di:I_s - m + di]
        v_neg = cs_sub[:, m - dj:J_s - m - dj, m - di:I_s - m - di]
        pair_mins.append(np.minimum(v_pos, v_neg))
        all_12.append(v_pos)
        all_12.append(v_neg)
      p_stack = np.sort(np.stack(pair_mins, axis=0), axis=0)
      best_sym = np.maximum(best_sym, 0.45 * p_stack[1] + 0.55 * p_stack[2])
      best_med = np.maximum(best_med, np.median(np.stack(all_12, axis=0), axis=0))

    min_tissue = max(12.0, float(body_thresh) * 0.45)
    valid = (best_med > min_tissue) & (best_sym > center_crop)
    drop = np.where(valid, best_sym - center_crop, 0.0)
    rel_drop = drop / (best_med + 12.0)
    dark_ratio = np.clip(1.0 - center_crop / (best_sym + 1e-3), 0.0, 1.0)
    resp_crop = drop * (rel_drop ** 1.5) * (dark_ratio ** 2)

    out_rsym[k0:k1, j0 + m:j1 - m, i0 + m:i1 - m] = best_sym
    out_resp[k0:k1, j0 + m:j1 - m, i0 + m:i1 - m] = resp_crop
    return out_resp, out_cs, out_rsym

  def autoDetectAllNeedles(
      self,
      volumeNode=None,
      kLim=None,
      minConfidence: float = 35.0,
      script: bool = False,
      renderNeedles: bool = True,
  ):
    """Automatically detect and segment all needles in the 3D volume without manual seed clicks.

    Uses a 2-pass iterative tract-peeling architecture:
    - Pass 1 runs the multi-radius (1.35, 1.85, 2.5, 3.3 mm) 6-diameter symmetric dark-tube filter
      (executed in multi-threaded Rust/C++), extracts local-maxima seeds, tracks bidirectionally
      with intra-pass tract attenuation and core-shaft residual tail trimming, and locks high-confidence
      template-bundle needles via consensus spatial-slope voting.
    - Pass 2 inpaints the dark cores of Pass-1 locked needles in the volume with surrounding tissue
      ring intensity so adjacent needles are unoccluded, re-runs the 3D symmetric response on the
      tight bundle ROI, masks Pass-1 tracts so Pass-2 trajectories cannot cross or re-track them,
      and performs bundle-slope-guided tracking to recover closely spaced or lower-contrast needles.
    """
    from scipy.ndimage import maximum_filter

    profprint()
    t0 = time.perf_counter()
    widget = getattr(slicer.modules, "NeedleFinderWidget", None)

    if volumeNode is None:
      try:
        volumeNode = (
            slicer.app.layoutManager()
            .sliceWidget("Red")
            .sliceLogic()
            .GetBackgroundLayer()
            .GetVolumeNode()
        )
      except Exception:
        volumeNode = None
    if volumeNode is None:
      volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    if volumeNode is None or volumeNode.GetImageData() is None:
      if not script:
        msgbox("Please load an MR/CT volume before running Auto-Detect All Needles.")
      return []

    imageData = volumeNode.GetImageData()
    vol_arr, dims = get_volume_f64_buffer(imageData)
    if vol_arr is None or vol_arr.ndim != 3:
      return []

    invert = bool(widget and hasattr(widget, "invertedContrast") and widget.invertedContrast.isChecked())
    if invert:
      v_max = float(np.max(vol_arr))
      vol_raw = np.ascontiguousarray(v_max - vol_arr, dtype=np.float64)
    else:
      vol_raw = vol_arr

    sp_i, sp_j, sp_k = [float(x) for x in volumeNode.GetSpacing()]
    K, J, I = vol_raw.shape

    vtk_mat = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(vtk_mat)
    ijk_to_ras_mat = np.array(
        [[vtk_mat.GetElement(r, c) for c in range(4)] for r in range(4)],
        dtype=np.float64,
    )
    rot_mat = ijk_to_ras_mat[:3, :3]
    trans_vec = ijk_to_ras_mat[:3, 3]
    ras_to_ijk = np.linalg.inv(ijk_to_ras_mat)
    vol_id = volumeNode.GetID() if hasattr(volumeNode, "GetID") else None

    # Determine axial base limit slice k_lim
    k_lim = None
    limit_source = "auto"
    if kLim is not None:
      k_lim = int(round(float(kLim)))
      limit_source = "explicit"
    else:
      # 1. Check if a 'template slice position*' marker exists in the MRML scene
      try:
        marker_nodes = slicer.util.getNodes('template slice position*') if hasattr(slicer, "util") else {}
      except Exception:
        marker_nodes = {}
      best_marker_z = -float("inf")
      marker_found = False
      for m_node in list(marker_nodes.values()):
        is_auto = (m_node.GetAttribute("AutoEstimated") == "1") if hasattr(m_node, "GetAttribute") else False
        m_vol_id = m_node.GetAttribute("VolumeNodeID") if hasattr(m_node, "GetAttribute") else None
        if is_auto and m_vol_id != vol_id:
          continue
        c_ras = [0.0, 0.0, 0.0]
        get_fiducial_coords(m_node, c_ras)
        if c_ras[2] > best_marker_z:
          best_marker_z = float(c_ras[2])
          marker_found = True
          limit_source = "auto" if is_auto else "marker"
      if marker_found:
        k_cand = int(round(float((ras_to_ijk @ np.array([0.0, 0.0, best_marker_z, 1.0]))[2])))
        if 0 <= k_cand < K:
          k_lim = k_cand

      # 2. Check widget.axialSegmentationLimit if explicitly set for this volume
      if k_lim is None and widget is not None and widget.axialSegmentationLimit is not None:
        auto_vid = getattr(widget, "_autoLimitVolumeID", None)
        if auto_vid is None or auto_vid == vol_id:
          k_cand = int(round(float(widget.axialSegmentationLimit)))
          if 0 <= k_cand < K:
            k_lim = k_cand
            limit_source = "widget"

      # 3. Check scene validation models via returnBasesFromNeedleModels if available
      if k_lim is None and hasattr(self, "returnBasesFromNeedleModels"):
        try:
          bases, _ = self.returnBasesFromNeedleModels()
          if len(bases) > 0:
            k_cand = int(round(float(np.min(np.array(bases)[:, 2]))))
            if 0 <= k_cand < K:
              k_lim = k_cand
              limit_source = "models"
        except Exception:
          pass

      # 4. Fallback to automatic intensity-based template slice estimation
      if k_lim is None:
        k_lim = self.estimateAxialLimitSlice(vol_raw, sp_k)
        limit_source = "auto"

    k_lim = int(np.clip(k_lim, 0, max(0, K - 5)))
    z_base_ras = float((rot_mat @ np.array([0.0, 0.0, float(k_lim)]) + trans_vec)[2])
    fov_center_ras = (rot_mat @ np.array([0.5 * I, 0.5 * J, float(k_lim)]) + trans_vec)[:2]

    if widget is not None:
      widget.axialSegmentationLimit = k_lim
      widget.axialSegmentationLimitRAS = z_base_ras
      if limit_source == "auto":
        widget._autoLimitVolumeID = vol_id
      else:
        widget._autoLimitVolumeID = None
      if getattr(widget, "templateSliceButton", None) is not None:
        tag = "auto" if limit_source == "auto" else "current"
        widget.templateSliceButton.text = (
            f"1. Select Current Axial Slice as Seg. Limit ({tag}: {z_base_ras:.1f} mm / slice {k_lim})"
        )
      if not script and hasattr(slicer, "mrmlScene") and slicer.mrmlScene is not None:
        try:
          existing_markers = slicer.util.getNodes('template slice position*') if hasattr(slicer, "util") else {}
          if not existing_markers:
            self.fiducialNode = create_fiducial_node(
                name='template slice position',
                ras=[float(fov_center_ras[0]), float(fov_center_ras[1]), float(z_base_ras)],
                color=(0.0, 1.0, 0.0),
                visibility=True,
                attributes={"AutoEstimated": "1", "VolumeNodeID": str(vol_id or "")} if limit_source == "auto" else None,
            )
          elif limit_source == "auto":
            for m_node in list(existing_markers.values()):
              set_fiducial_coords(m_node, [float(fov_center_ras[0]), float(fov_center_ras[1]), float(z_base_ras)])
              if hasattr(m_node, "SetAttribute"):
                m_node.SetAttribute("AutoEstimated", "1")
                m_node.SetAttribute("VolumeNodeID", str(vol_id or ""))
        except Exception:
          pass

    k0 = max(0, k_lim - 1)
    i0, i1 = int(0.16 * I), int(0.84 * I)
    j0, j1 = int(0.16 * J), int(0.84 * J)
    if i1 - i0 < 26 or j1 - j0 < 26:
      i0, i1 = 0, I
      j0, j1 = 0, J

    # Modal progress dialog for interactive 3D Slicer sessions
    prog_dialog = None
    n_slices_est = max(1, K - k0)
    vox_scale = float(n_slices_est * (i1 - i0) * (j1 - j0)) / float(max(1, 35 * 220 * 180))
    est_total_sec = max(0.8, min(5.0, 0.5 + 0.45 * vox_scale))
    if not script and hasattr(slicer, "app") and slicer.app is not None:
      try:
        parent_win = slicer.util.mainWindow() if hasattr(slicer, "util") else None
        prog_dialog = qt.QProgressDialog(
            f"Stage 1/6: Initializing volume & 3D symmetry filter (slices {k0}..{K - 1})...\nEstimated wait: ~{est_total_sec:.1f}s",
            None,
            0,
            100,
            parent_win,
        )
        prog_dialog.setWindowTitle("NeedleFinder — Auto-Detecting Needles")
        prog_dialog.setWindowModality(qt.Qt.WindowModal)
        prog_dialog.setMinimumDuration(0)
        prog_dialog.setMinimumWidth(440)
        prog_dialog.setAutoClose(False)
        prog_dialog.setAutoReset(False)
        prog_dialog.setValue(4)
        prog_dialog.show()
        slicer.app.processEvents()
      except Exception:
        prog_dialog = None

    def update_progress(pct, stage_msg):
      if prog_dialog is not None:
        try:
          elapsed_now = time.perf_counter() - t0
          frac = max(0.06, min(0.99, float(pct) / 100.0))
          rem_est = max(0.1, max(est_total_sec - elapsed_now, elapsed_now * (1.0 - frac) / frac))
          prog_dialog.setLabelText(
              f"{stage_msg}\nElapsed: {elapsed_now:.1f}s  |  Est. remaining: ~{rem_est:.1f}s"
          )
          prog_dialog.setValue(int(np.clip(pct, 0, 100)))
          slicer.app.processEvents()
        except Exception:
          pass

    def close_progress():
      if prog_dialog is not None:
        try:
          prog_dialog.setValue(100)
          prog_dialog.close()
          prog_dialog.deleteLater()
          slicer.app.processEvents()
        except Exception:
          pass

    try:
      roi_raw = vol_raw[k0:K, j0:j1, i0:i1]
      pos_raw = roi_raw[roi_raw > 5.0]
      if len(pos_raw):
        p30_raw, med_tissue = [float(x) for x in np.percentile(pos_raw, [30.0, 55.0])]
      else:
        p30_raw, med_tissue = 25.0, 85.0
      if med_tissue > 450.0:
        scale_factor = 160.0 / med_tissue
        vol_work = np.ascontiguousarray(vol_raw * scale_factor, dtype=np.float64)
        vol_work_owned = True
        body_thresh = p30_raw * scale_factor
      else:
        vol_work = vol_raw
        vol_work_owned = False
        body_thresh = p30_raw

      nms_px = max(2, int(round(1.25 / sp_i)))
      visit_r_px = max(1, int(round(1.10 / sp_i)))
      search_r_px = max(2, int(round(2.05 / sp_i)))
      yy_grid, xx_grid = np.mgrid[-search_r_px:search_r_px + 1, -search_r_px:search_r_px + 1]
      wy, wx = np.mgrid[-1:2, -1:2]

      max_gap_slices = 3 if sp_k < 2.2 else 2
      min_pts_slices = 5 if (sp_k < 2.6 and (K - k0) > 28) else 4

      # Multi-radius 4-axis offsets for rejecting 1D planar trenches (e.g. intergluteal cleft)
      r1_i = max(1, int(round(2.2 / sp_i)))
      r1_j = max(1, int(round(2.2 / sp_j)))
      r2_i = max(2, int(round(4.2 / sp_i)))
      r2_j = max(2, int(round(4.2 / sp_j)))
      d1_i = max(1, int(round(1.6 / sp_i)))
      d1_j = max(1, int(round(1.6 / sp_j)))
      d2_i = max(2, int(round(3.0 / sp_i)))
      d2_j = max(2, int(round(3.0 / sp_j)))

      def is_1d_trench_segment(sub_arr):
        cx_l, cy_l = [], []
        for row in sub_arr:
          ia, ja, ka = int(round(row[0])), int(round(row[1])), int(round(row[2]))
          if not (r2_i <= ia < I - r2_i and r2_j <= ja < J - r2_j and 0 <= ka < K):
            continue
          if ja < int(0.58 * J):
            return False
          c = float(vol_work[ka, ja, ia])
          # In a vertical posterior cleft (intergluteal crease), X is bright on both sides while Y is dark on both sides
          cx1 = 0.5 * (float(vol_work[ka, ja, ia - r1_i]) + float(vol_work[ka, ja, ia + r1_i])) - c
          cx2 = 0.5 * (float(vol_work[ka, ja, ia - r2_i]) + float(vol_work[ka, ja, ia + r2_i])) - c
          cy_max1 = max(float(vol_work[ka, ja - r1_j, ia]), float(vol_work[ka, ja + r1_j, ia])) - c
          cy_max2 = max(float(vol_work[ka, ja - r2_j, ia]), float(vol_work[ka, ja + r2_j, ia])) - c
          cx_l.append(max(cx1, cx2))
          cy_l.append(0.5 * (cy_max1 + cy_max2))
        if len(cx_l) < 3:
          return False
        med_x = float(np.median(cx_l))
        med_y = float(np.median(cy_l))
        return med_x > 8.0 and med_y < 0.18 * med_x

      def is_1d_trench_artifact(arr_pts, k_min_rel):
        if (K - k0) * sp_k <= 185.0 or k_min_rel * sp_k > 52.0:
          return False
        if is_1d_trench_segment(arr_pts):
          return True
        n_half = max(3, len(arr_pts) // 2)
        if len(arr_pts) >= 6 and is_1d_trench_segment(arr_pts[:n_half]):
          return True
        return False

      def fast_pct(a, q):
        s = np.sort(a)
        pos = (q * 0.01) * (len(s) - 1)
        lo = int(pos)
        hi = lo + 1 if lo + 1 < len(s) else lo
        return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))

      def fast_line_fit(z, x, y):
        n = len(z)
        zm = float(np.sum(z)) / n
        xm = float(np.sum(x)) / n
        ym = float(np.sum(y)) / n
        dz = z - zm
        denom = float(np.dot(dz, dz))
        if denom < 1e-6:
          return 0.0, xm, 0.0, ym
        sx = float(np.dot(dz, x - xm)) / denom
        sy = float(np.dot(dz, y - ym)) / denom
        return sx, xm - sx * zm, sy, ym - sy * zm

      def fast_poly_xy(z, x, y, deg):
        n = len(z)
        zm = float(np.sum(z)) / n
        xm = float(np.sum(x)) / n
        ym = float(np.sum(y)) / n
        u = z - zm
        dx = x - xm
        dy = y - ym
        s2 = float(np.dot(u, u))
        if deg == 1 or s2 < 1e-12:
          if s2 < 1e-6:
            return np.array([0.0, xm]), np.array([0.0, ym])
          sx = float(np.dot(u, dx)) / s2
          sy = float(np.dot(u, dy)) / s2
          return np.array([sx, xm - sx * zm]), np.array([sy, ym - sy * zm])
        u2 = u * u
        s3 = float(np.dot(u2, u))
        s4 = float(np.dot(u2, u2))
        a11 = s4 - (s2 * s2) / n
        a12 = s3
        a22 = s2
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
          sx = float(np.dot(u, dx)) / s2
          sy = float(np.dot(u, dy)) / s2
          return np.array([0.0, sx, xm - sx * zm]), np.array([0.0, sy, ym - sy * zm])
        inv_det = 1.0 / det
        rx1 = float(np.dot(u2, dx))
        rx2 = float(np.dot(u, dx))
        ax = (rx1 * a22 - rx2 * a12) * inv_det
        bx = (a11 * rx2 - a12 * rx1) * inv_det
        cx = -(s2 / n) * ax
        ry1 = float(np.dot(u2, dy))
        ry2 = float(np.dot(u, dy))
        ay = (ry1 * a22 - ry2 * a12) * inv_det
        by = (a11 * ry2 - a12 * ry1) * inv_det
        cy = -(s2 / n) * ay
        px = np.array([ax, bx - 2.0 * ax * zm, xm + cx + ax * zm * zm - bx * zm])
        py = np.array([ay, by - 2.0 * ay * zm, ym + cy + ay * zm * zm - by * zm])
        return px, py

      is_shallow_scan = 1 if (K - k0) * sp_k <= 185.0 else 0

      def trim_trajectory_inside_out(ras_pts, arr_pts, ks, shaft_p75):
        n = len(ks)
        if n < 6:
          return ras_pts, arr_pts, ks
        rust_bounds = rust_trim_trajectory_inside_out(ras_pts, arr_pts, shaft_p75, is_shallow_scan)
        if rust_bounds is not None:
          start_idx, end_idx = rust_bounds
          return ras_pts[start_idx:end_idx], arr_pts[start_idx:end_idx], ks[start_idx:end_idx]

        z_all = ras_pts[:, 2]
        x_all = ras_pts[:, 0]
        y_all = ras_pts[:, 1]
        if is_shallow_scan:
          deg0 = 2 if n >= 7 else 1
          px0, py0 = fast_poly_xy(z_all, x_all, y_all, deg0)
          res0 = np.hypot(x_all - np.polyval(px0, z_all), y_all - np.polyval(py0, z_all))
          if float(np.sqrt(np.mean(res0 ** 2))) <= 1.30:
            return ras_pts, arr_pts, ks

        win_n = min(n, max(5, min(16, int(round(0.55 * n)))))
        best_rmse = 1e9
        best_i0 = 0
        best_fit = None
        max_start = max(1, int(round(0.50 * (n - win_n))) + 1)
        for s_i in range(max_start):
          e_i = s_i + win_n
          sub_z = z_all[s_i:e_i]
          if abs(float(sub_z[-1] - sub_z[0])) < 3.5:
            continue
          sx, bx, sy, by = fast_line_fit(sub_z, x_all[s_i:e_i], y_all[s_i:e_i])
          dx = x_all[s_i:e_i] - (sx * sub_z + bx)
          dy = y_all[s_i:e_i] - (sy * sub_z + by)
          err = float(np.sqrt(np.mean(dx * dx + dy * dy)))
          if err < best_rmse:
            best_rmse = err
            best_i0 = s_i
            best_fit = (sx, bx, sy, by)
        if best_fit is None:
          return ras_pts, arr_pts, ks

        gsx, gbx, gsy, gby = best_fit
        end_idx = n
        bad_streak = 0
        for idx in range(best_i0 + win_n, n):
          r0 = max(best_i0, idx - 14)
          lsx, lbx, lsy, lby = fast_line_fit(z_all[r0:idx], x_all[r0:idx], y_all[r0:idx])
          zi, xi, yi = float(z_all[idx]), float(x_all[idx]), float(y_all[idx])
          dev_loc = math.hypot(xi - (lsx * zi + lbx), yi - (lsy * zi + lby))
          dev_glob = math.hypot(xi - (gsx * zi + gbx), yi - (gsy * zi + gby))
          max_glob = max(3.25, 0.048 * abs(zi - float(z_all[best_i0])))
          if dev_loc > 1.85 or dev_glob > max_glob or (dev_loc > 1.35 and arr_pts[idx, 3] < 0.35 * shaft_p75):
            bad_streak += 1
            if bad_streak >= 2 or dev_loc > 2.35 or idx == n - 1:
              end_idx = idx - bad_streak + 1
              break
          else:
            bad_streak = 0

        start_idx = 0
        bad_streak = 0
        for idx in range(best_i0 - 1, -1, -1):
          r1 = min(end_idx, idx + 15)
          lsx, lbx, lsy, lby = fast_line_fit(z_all[idx + 1:r1], x_all[idx + 1:r1], y_all[idx + 1:r1])
          zi, xi, yi = float(z_all[idx]), float(x_all[idx]), float(y_all[idx])
          dev_loc = math.hypot(xi - (lsx * zi + lbx), yi - (lsy * zi + lby))
          dev_glob = math.hypot(xi - (gsx * zi + gbx), yi - (gsy * zi + gby))
          max_glob = max(3.25, 0.048 * abs(float(z_all[best_i0 + win_n - 1]) - zi))
          if dev_loc > 1.95 or dev_glob > max_glob or (dev_loc > 1.45 and arr_pts[idx, 3] < 0.30 * shaft_p75):
            bad_streak += 1
            if bad_streak >= 2 or dev_loc > 2.45 or idx == 0:
              start_idx = idx + bad_streak
              break
          else:
            bad_streak = 0

        return ras_pts[start_idx:end_idx], arr_pts[start_idx:end_idx], ks[start_idx:end_idx]

      def fit_and_package_traj(ras_pts, arr_pts, ks, pass_idx, cand_thresh):
        z_vals = ras_pts[:, 2]
        z_span_mm = float(z_vals[-1] - z_vals[0])
        deg = 2 if len(z_vals) >= 7 else 1
        sx1, bx1, sy1, by1 = fast_line_fit(z_vals, ras_pts[:, 0], ras_pts[:, 1])
        res_l = np.hypot(ras_pts[:, 0] - (sx1 * z_vals + bx1), ras_pts[:, 1] - (sy1 * z_vals + by1))
        inliers = res_l <= max(1.65, fast_pct(res_l, 85))
        if int(np.sum(inliers)) >= (deg + 2):
          px, py = fast_poly_xy(z_vals[inliers], ras_pts[inliers, 0], ras_pts[inliers, 1], deg)
        else:
          px = np.array([sx1, bx1])
          py = np.array([sy1, by1])
        if len(px) == 3:
          fit_x = (px[0] * z_vals + px[1]) * z_vals + px[2]
          fit_y = (py[0] * z_vals + py[1]) * z_vals + py[2]
        else:
          fit_x = px[0] * z_vals + px[1]
          fit_y = py[0] * z_vals + py[1]
        res_d = np.hypot(ras_pts[:, 0] - fit_x, ras_pts[:, 1] - fit_y)
        rmse = fast_pct(res_d, 85)
        if rmse > 1.45:
          return None

        dz = z_vals[-1] - z_vals[0]
        if abs(dz) < 1e-3:
          return None
        slope_x = (fit_x[-1] - fit_x[0]) / dz
        slope_y = (fit_y[-1] - fit_y[0]) / dz
        if math.hypot(slope_x, slope_y) > 0.72:
          return None

        k_rels = arr_pts[:, 2] - k0
        si_a, bi_a, sj_a, bj_a = fast_line_fit(k_rels, arr_pts[:, 0], arr_pts[:, 1])
        pi_abs = np.array([si_a, bi_a])
        pj_abs = np.array([sj_a, bj_a])

        bx_lin = float(fit_x[0] + slope_x * (z_base_ras - z_vals[0]))
        by_lin = float(fit_y[0] + slope_y * (z_base_ras - z_vals[0]))
        if len(px) == 3:
          bx_poly = float((px[0] * z_base_ras + px[1]) * z_base_ras + px[2])
          by_poly = float((py[0] * z_base_ras + py[1]) * z_base_ras + py[2])
        else:
          bx_poly = float(px[0] * z_base_ras + px[1])
          by_poly = float(py[0] * z_base_ras + py[1])
        use_linear_ext = (
            (sp_k >= 2.8 and z_vals[0] > z_base_ras + 2.0 * sp_k)
            or ((K - k0) * sp_k > 185.0 and (abs(float(z_vals[0] - z_base_ras)) > 22.0 or math.hypot(bx_poly - bx_lin, by_poly - by_lin) > 3.5))
        )
        if use_linear_ext:
          bx, by = bx_lin, by_lin
        else:
          bx, by = bx_poly, by_poly
        base_xy = np.array([bx, by])
        if z_vals[-1] >= z_vals[0]:
          above_mask = z_vals > z_base_ras + 0.15 * sp_k
        else:
          above_mask = z_vals < z_base_ras - 0.15 * sp_k
        if np.any(above_mask):
          fx_ab = fit_x[above_mask]
          fy_ab = fit_y[above_mask]
          fz_ab = z_vals[above_mask]
          z_first = float(fz_ab[0])
          if use_linear_ext and abs(z_first - z_base_ras) > 6.0 * sp_k:
            n_ext = max(2, int(round(abs(z_first - z_base_ras) / sp_k)))
            z_ext = np.linspace(z_base_ras, z_first, n_ext, endpoint=False)
            x_ext = float(fx_ab[0]) + slope_x * (z_ext - z_first)
            y_ext = float(fy_ab[0]) + slope_y * (z_ext - z_first)
            smooth_ras = np.vstack([
                np.column_stack([x_ext, y_ext, z_ext]),
                np.column_stack([fx_ab, fy_ab, fz_ab]),
            ])
          else:
            smooth_ras = np.vstack([
                [bx, by, z_base_ras],
                np.column_stack([fx_ab, fy_ab, fz_ab]),
            ])
        else:
          smooth_ras = np.vstack([
              [bx, by, z_base_ras],
              [fit_x[-1], fit_y[-1], z_vals[-1]],
          ])

        mean_r = float(np.mean(arr_pts[:, 3]))
        p75_r = fast_pct(arr_pts[:, 3], 75)
        max_r = float(np.max(arr_pts[:, 3]))
        high_frac = float(np.mean(arr_pts[:, 3] >= cand_thresh * 0.38))
        raw_score = mean_r * (high_frac ** 1.5) * math.sqrt(max(1.0, z_span_mm)) / (1.0 + 0.5 * rmse)
        curv = math.hypot(float(px[0]), float(py[0])) if len(px) == 3 else 0.0
        return {
            "ras": smooth_ras,
            "x_min": float(np.min(smooth_ras[:, 0])),
            "x_max": float(np.max(smooth_ras[:, 0])),
            "y_min": float(np.min(smooth_ras[:, 1])),
            "y_max": float(np.max(smooth_ras[:, 1])),
            "raw_ras_pts": ras_pts,
            "raw_arr_pts": arr_pts,
            "raw_ks": list(ks),
            "ks": np.array(ks) + k0,
            "n_slices": len(ks),
            "k_min_rel": int(min(ks)),
            "k_max_rel": int(max(ks)),
            "z_start_ras": float(z_vals[0]),
            "z_end_ras": float(z_vals[-1]),
            "track_start_xy": np.array([float(fit_x[0]), float(fit_y[0])]),
            "track_end_xy": np.array([float(fit_x[-1]), float(fit_y[-1])]),
            "pi_abs": pi_abs,
            "pj_abs": pj_abs,
            "slope_ij": np.array([si_a, sj_a]),
            "base_xy": base_xy,
            "tip_ras": smooth_ras[-1].copy(),
            "slope": np.array([slope_x, slope_y]),
            "mean_r": mean_r,
            "p75_r": p75_r,
            "max_r": max_r,
            "high_frac": high_frac,
            "rmse": rmse,
            "curv": curv,
            "length_mm": float(smooth_ras[-1, 2] - smooth_ras[0, 2]),
            "z_span_tracked": z_span_mm,
            "raw_score": raw_score,
            "pass": pass_idx,
        }

      def merge_two_trajs(t_low, t_high, cand_thresh):
        by_k = {}
        for idx_pt, k_rel in enumerate(t_low["raw_ks"]):
          by_k[int(k_rel)] = (t_low["raw_ras_pts"][idx_pt], t_low["raw_arr_pts"][idx_pt])
        for idx_pt, k_rel in enumerate(t_high["raw_ks"]):
          kr = int(k_rel)
          if kr not in by_k or float(t_high["raw_arr_pts"][idx_pt, 3]) > float(by_k[kr][1][3]):
            by_k[kr] = (t_high["raw_ras_pts"][idx_pt], t_high["raw_arr_pts"][idx_pt])
        ks_m = sorted(by_k.keys())
        ras_m = np.array([by_k[kr][0] for kr in ks_m], dtype=np.float64)
        arr_m = np.array([by_k[kr][1] for kr in ks_m], dtype=np.float64)
        order = np.argsort(ras_m[:, 2])
        ras_m = ras_m[order]
        arr_m = arr_m[order]
        ks_m = [ks_m[i] for i in order]
        return fit_and_package_traj(ras_m, arr_m, ks_m, min(t_low["pass"], t_high["pass"]), cand_thresh)

      def stitch_trajs(traj_list, cand_thresh):
        if len(traj_list) < 2:
          return traj_list
        ordered = sorted(traj_list, key=lambda x: (x["k_min_rel"], -x["raw_score"]))
        used = [False] * len(ordered)
        out = []
        max_k_gap = max(6, int(round(28.0 / sp_k)))
        for i in range(len(ordered)):
          if used[i]:
            continue
          cur = ordered[i]
          used[i] = True
          merged_any = True
          while merged_any:
            merged_any = False
            best_j = -1
            best_pkg = None
            best_d = 1.85
            for j in range(len(ordered)):
              if used[j]:
                continue
              cand = ordered[j]
              if cand["k_min_rel"] < cur["k_max_rel"] - 6 or cand["k_min_rel"] > cur["k_max_rel"] + max_k_gap:
                continue
              if cand["k_max_rel"] <= cur["k_max_rel"] + 3:
                continue
              if float(np.hypot(cur["slope"][0] - cand["slope"][0], cur["slope"][1] - cand["slope"][1])) > 0.09:
                continue
              z_mid = 0.5 * (cur["z_end_ras"] + cand["z_start_ras"])
              p_cur = cur["track_end_xy"] + cur["slope"] * (z_mid - cur["z_end_ras"])
              p_cnd = cand["track_start_xy"] + cand["slope"] * (z_mid - cand["z_start_ras"])
              d_mid = float(np.hypot(p_cur[0] - p_cnd[0], p_cur[1] - p_cnd[1]))
              if d_mid <= best_d:
                pkg_m = merge_two_trajs(cur, cand, cand_thresh)
                if pkg_m is not None:
                  best_d = d_mid
                  best_j = j
                  best_pkg = pkg_m
            if best_j >= 0 and best_pkg is not None:
              cur = best_pkg
              used[best_j] = True
              merged_any = True
          if not cur.get("_upper_only", False):
            out.append(cur)
        return out

      def track_pass(resp_sub, c_smooth, ring_sym, sub_i0, sub_j0, cand_thresh, prior_slope_ij=None, pass_idx=1):
        if not (resp_sub.dtype == np.float32 and resp_sub.flags.c_contiguous):
          resp_sub = np.ascontiguousarray(resp_sub, dtype=np.float32)
        if not (c_smooth.dtype == np.float32 and c_smooth.flags.c_contiguous):
          c_smooth = np.ascontiguousarray(c_smooth, dtype=np.float32)
        if not (ring_sym.dtype == np.float32 and ring_sym.flags.c_contiguous):
          ring_sym = np.ascontiguousarray(ring_sym, dtype=np.float32)
        K_sub, J_sub, I_sub = resp_sub.shape
        pos_v = resp_sub[resp_sub > 0.5]
        if len(pos_v) > 20:
          cap_val = float(np.percentile(pos_v, 99.0))
          np.minimum(resp_sub, cap_val, out=resp_sub)

        max_seed_k = min(K_sub - 2, max(18, min(42, int(round(62.0 / sp_k)))))
        seed_cut = cand_thresh * ((0.82 if K_sub <= 28 else 0.88) if pass_idx == 1 else 0.62)
        seeds = []
        used_rust_nms = False
        for k_rel in range(0, max_seed_k):
          res_nms = rust_find_2d_nms_peaks(resp_sub[k_rel], nms_px, seed_cut)
          if res_nms is None:
            break
          used_rust_nms = True
          jj, ii, vv = res_nms
          sl_seeds = [
              (float(v_s), k_rel, float(i_s), float(j_s), False)
              for j_s, i_s, v_s in zip(jj, ii, vv)
          ]
          sl_seeds.sort(key=lambda x: -x[0])
          seeds.extend(sl_seeds[:110])
        if used_rust_nms and K_sub * sp_k > 185.0:
          up_max_k = min(K_sub - 5, int(round(165.0 / sp_k)))
          up_cut = seed_cut * 1.05
          for k_rel in range(max_seed_k, up_max_k, 2):
            res_nms = rust_find_2d_nms_peaks(resp_sub[k_rel], nms_px, up_cut)
            if res_nms is not None:
              jj, ii, vv = res_nms
              sl_seeds = [
                  (float(v_s), k_rel, float(i_s), float(j_s), False)
                  for j_s, i_s, v_s in zip(jj, ii, vv)
              ]
              sl_seeds.sort(key=lambda x: -x[0])
              seeds.extend(sl_seeds[:45])
        if not used_rust_nms:
          loc_max = resp_sub == maximum_filter(resp_sub, size=(1, 2 * nms_px + 1, 2 * nms_px + 1))
          for k_rel in range(0, max_seed_k):
            jj, ii = np.where(loc_max[k_rel] & (resp_sub[k_rel] >= seed_cut))
            sl_seeds = [
                (float(resp_sub[k_rel, j_s, i_s]), k_rel, float(i_s), float(j_s), False)
                for j_s, i_s in zip(jj, ii)
            ]
            sl_seeds.sort(key=lambda x: -x[0])
            seeds.extend(sl_seeds[:110])
          if K_sub * sp_k > 185.0:
            up_max_k = min(K_sub - 5, int(round(165.0 / sp_k)))
            up_cut = seed_cut * 1.05
            for k_rel in range(max_seed_k, up_max_k, 2):
              jj, ii = np.where(loc_max[k_rel] & (resp_sub[k_rel] >= up_cut))
              sl_seeds = [
                  (float(resp_sub[k_rel, j_s, i_s]), k_rel, float(i_s), float(j_s), False)
                  for j_s, i_s in zip(jj, ii)
              ]
              sl_seeds.sort(key=lambda x: -x[0])
              seeds.extend(sl_seeds[:45])
        seeds.sort(key=lambda x: -x[0])

        visited = np.zeros((K_sub, J_sub, I_sub), dtype=bool)
        trajs_out = []
        max_step_px = (0.65 if pass_idx == 1 else 0.52) * (sp_k / sp_i)
        decay = -0.36 if pass_idx == 1 else -0.46
        gauss_base = np.ascontiguousarray(np.exp(decay * ((xx_grid * sp_i) ** 2 + (yy_grid * sp_j) ** 2)), dtype=np.float64)
        resp_ptr = resp_sub.ctypes.data_as(_FLOAT_PTR)
        cs_ptr = c_smooth.ctypes.data_as(_FLOAT_PTR)
        sym_ptr = ring_sym.ctypes.data_as(_FLOAT_PTR)
        gauss_ptr = gauss_base.ctypes.data_as(_DOUBLE_PTR)
        has_prior = 1 if prior_slope_ij is not None else 0
        prior_si = float(prior_slope_ij[0]) if prior_slope_ij is not None else 0.0
        prior_sj = float(prior_slope_ij[1]) if prior_slope_ij is not None else 0.0

        for s_val, k_start, i_start, j_start, is_up_seed in seeds:
          if visited[k_start, int(j_start), int(i_start)]:
            continue
          visited[k_start, int(j_start), int(i_start)] = True

          tr_res = rust_trace_seed_3d(
              resp_ptr,
              cs_ptr,
              sym_ptr,
              I_sub,
              J_sub,
              K_sub,
              k_start,
              i_start,
              j_start,
              s_val,
              cand_thresh,
              max_step_px,
              search_r_px,
              max_gap_slices,
              min_pts_slices,
              pass_idx,
              has_prior,
              prior_si,
              prior_sj,
              gauss_ptr,
          )
          if tr_res is not None:
            ks_sub, i_sub, j_sub, r_sub, shaft_p75 = tr_res
            if len(ks_sub) == 0:
              continue
            arr_pts = np.column_stack([i_sub + sub_i0, j_sub + sub_j0, ks_sub + k0, r_sub])
            ks = ks_sub.tolist()
          else:
            pts = {k_start: (i_start, j_start, s_val)}
            peak_r = min(s_val, cand_thresh * 40.0)
            for direction in (1, -1):
              curr_i, curr_j = i_start, j_start
              if prior_slope_ij is not None:
                vel_i = float(prior_slope_ij[0]) * direction
                vel_j = float(prior_slope_ij[1]) * direction
              else:
                vel_i, vel_j = 0.0, 0.0
              gap = 0
              k_cur = k_start + direction
              while 0 <= k_cur < K_sub:
                pred_i = curr_i + vel_i
                pred_j = curr_j + vel_j
                ic, jc = int(round(pred_i)), int(round(pred_j))
                if not (
                    search_r_px <= ic < I_sub - search_r_px
                    and search_r_px <= jc < J_sub - search_r_px
                ):
                  break
                patch_r = resp_sub[k_cur, jc - search_r_px:jc + search_r_px + 1, ic - search_r_px:ic + search_r_px + 1]
                eff_score = patch_r * gauss_base
                best_idx = np.unravel_index(np.argmax(eff_score), eff_score.shape)
                best_r = float(patch_r[best_idx])
                best_c = float(c_smooth[k_cur, jc - search_r_px + best_idx[0], ic - search_r_px + best_idx[1]])
                best_sym = float(ring_sym[k_cur, jc - search_r_px + best_idx[0], ic - search_r_px + best_idx[1]])
                ni = float(ic - search_r_px + best_idx[1])
                nj = float(jc - search_r_px + best_idx[0])
                ii0, jj0 = int(ni), int(nj)
                if 1 <= ii0 < I_sub - 1 and 1 <= jj0 < J_sub - 1 and best_r > 0.5:
                  w_patch = np.maximum(resp_sub[k_cur, jj0 - 1:jj0 + 2, ii0 - 1:ii0 + 2], 0.0)
                  w_sum = float(np.sum(w_patch))
                  if w_sum > 0:
                    ni = ii0 + float(np.sum(wx * w_patch)) / w_sum
                    nj = jj0 + float(np.sum(wy * w_patch)) / w_sum

                min_step_r = max(cand_thresh * (0.28 if pass_idx == 1 else 0.20), 0.045 * peak_r)
                is_dark = (best_r >= min_step_r) or (best_sym > 14.0 and best_c < 0.56 * best_sym and best_r >= 0.25 * min_step_r)
                if is_dark:
                  pts[k_cur] = (ni, nj, best_r)
                  peak_r = max(peak_r, min(best_r, cand_thresh * 42.0))
                  step_i = max(-max_step_px, min(max_step_px, ni - curr_i))
                  step_j = max(-max_step_px, min(max_step_px, nj - curr_j))
                  vel_i = 0.65 * vel_i + 0.35 * step_i
                  vel_j = 0.65 * vel_j + 0.35 * step_j
                  curr_i, curr_j = ni, nj
                  gap = 0
                else:
                  gap += 1
                  if gap > max_gap_slices:
                    break
                  curr_i, curr_j = pred_i, pred_j
                k_cur += direction

            ks = sorted(pts.keys())
            if len(ks) < min_pts_slices:
              continue
            r_arr_ks = np.array([pts[k][2] for k in ks])
            shaft_p75 = float(np.percentile(r_arr_ks, 75))
            tip_cut = max(cand_thresh * 0.35, 0.08 * shaft_p75)
            base_cut = max(cand_thresh * 0.22, 0.05 * shaft_p75)
            while len(ks) > 2 and pts[ks[-1]][2] < tip_cut:
              ks.pop()
            while len(ks) > 2 and pts[ks[0]][2] < base_cut:
              ks.pop(0)
            if len(ks) < min_pts_slices:
              continue

            arr_pts = np.array([[pts[k][0] + sub_i0, pts[k][1] + sub_j0, k + k0, pts[k][2]] for k in ks], dtype=np.float64)

          ras_pts = arr_pts[:, :3] @ rot_mat.T + trans_vec
          order = np.argsort(ras_pts[:, 2])
          ras_pts = ras_pts[order]
          arr_pts = arr_pts[order]
          ks = [ks[idx] for idx in order]

          ras_pts, arr_pts, ks = trim_trajectory_inside_out(ras_pts, arr_pts, ks, shaft_p75)
          if len(ks) < min_pts_slices:
            continue

          k_min_rel = min(ks)
          z_span_mm = float(ras_pts[-1, 2] - ras_pts[0, 2])
          if z_span_mm < (8.5 if K_sub <= 28 else 11.5):
            continue
          if k_min_rel > 12 and z_span_mm < (18.0 if K_sub <= 36 else 26.0):
            continue

          if is_1d_trench_artifact(arr_pts, k_min_rel):
            continue

          high_frac = float(np.mean(arr_pts[:, 3] >= cand_thresh * ((0.36 if K_sub <= 28 else 0.40) if pass_idx == 1 else 0.28)))
          if high_frac < ((0.46 if K_sub <= 28 else 0.50) if pass_idx == 1 else 0.40):
            continue

          pkg = fit_and_package_traj(ras_pts, arr_pts, ks, pass_idx, cand_thresh)
          if pkg is None:
            continue
          pkg["_upper_only"] = bool(is_up_seed)

          # Mark visited and attenuate resp_sub along the accepted tract so adjacent seeds do not jump onto it
          if not is_up_seed:
            for idx_v, k_v in enumerate(ks):
              iv = int(round(arr_pts[idx_v, 0] - sub_i0))
              jv = int(round(arr_pts[idx_v, 1] - sub_j0))
              j_lo, j_hi = max(0, jv - visit_r_px), min(J_sub, jv + visit_r_px + 1)
              i_lo, i_hi = max(0, iv - visit_r_px), min(I_sub, iv + visit_r_px + 1)
              visited[k_v, j_lo:j_hi, i_lo:i_hi] = True
              resp_sub[k_v, j_lo:j_hi, i_lo:i_hi] *= 0.08

          trajs_out.append(pkg)
        if K_sub * sp_k > 185.0:
          trajs_out = stitch_trajs(trajs_out, cand_thresh)
        return trajs_out

      def dedup_trajs(traj_list, existing_list=None):
        out = list(existing_list) if existing_list else []
        is_deep = ((K - k0) * sp_k > 185.0)
        for t in sorted(traj_list, key=lambda x: -x["raw_score"]):
          dup = False
          t_bx, t_by = float(t["base_xy"][0]), float(t["base_xy"][1])
          t_xmin, t_xmax = t["x_min"], t["x_max"]
          t_ymin, t_ymax = t["y_min"], t["y_max"]
          for idx_p, p in enumerate(out):
            base_d = math.hypot(t_bx - float(p["base_xy"][0]), t_by - float(p["base_xy"][1]))
            if base_d < 2.15:
              dup = True
              break
            xy_close = not (
                t_xmax + 3.15 < p["x_min"]
                or p["x_max"] + 3.15 < t_xmin
                or t_ymax + 3.15 < p["y_min"]
                or p["y_max"] + 3.15 < t_ymin
            )
            if xy_close:
              z_min = max(t["ras"][0, 2], p["ras"][0, 2])
              z_max = min(t["ras"][-1, 2], p["ras"][-1, 2])
              if z_max >= z_min:
                zs = np.linspace(z_min, z_max, 12)
                tx = np.interp(zs, t["ras"][:, 2], t["ras"][:, 0])
                ty = np.interp(zs, t["ras"][:, 2], t["ras"][:, 1])
                px_i = np.interp(zs, p["ras"][:, 2], p["ras"][:, 0])
                py_i = np.interp(zs, p["ras"][:, 2], p["ras"][:, 1])
                if float(np.mean(np.hypot(tx - px_i, ty - py_i))) < 2.40:
                  dup = True
                  break
            # Also check actual tracked Z-overlap and collinear continuation on deep scans
            if is_deep:
              z_ov0 = max(t["z_start_ras"], p["z_start_ras"])
              z_ov1 = min(t["z_end_ras"], p["z_end_ras"])
              if z_ov1 >= z_ov0 + 4.0:
                if xy_close:
                  zs_ov = np.linspace(z_ov0, z_ov1, 8)
                  tx_o = np.interp(zs_ov, t["ras"][:, 2], t["ras"][:, 0])
                  ty_o = np.interp(zs_ov, t["ras"][:, 2], t["ras"][:, 1])
                  px_o = np.interp(zs_ov, p["ras"][:, 2], p["ras"][:, 0])
                  py_o = np.interp(zs_ov, p["ras"][:, 2], p["ras"][:, 1])
                  if float(np.mean(np.hypot(tx_o - px_o, ty_o - py_o))) < 3.10:
                    if t["z_end_ras"] > p["z_end_ras"] + 8.0 and p["k_min_rel"] < t["k_min_rel"]:
                      pkg_m = merge_two_trajs(p, t, cand_thresh)
                      if pkg_m is not None:
                        out[idx_p] = pkg_m
                    dup = True
                    break
              elif abs(t["z_start_ras"] - p["z_end_ras"]) <= 20.0 or abs(p["z_start_ras"] - t["z_end_ras"]) <= 20.0:
                t_lo, t_hi = (p, t) if p["z_start_ras"] <= t["z_start_ras"] else (t, p)
                if float(np.hypot(t_lo["slope"][0] - t_hi["slope"][0], t_lo["slope"][1] - t_hi["slope"][1])) <= 0.09:
                  z_m = 0.5 * (t_lo["z_end_ras"] + t_hi["z_start_ras"])
                  xy_lo = t_lo["track_end_xy"] + t_lo["slope"] * (z_m - t_lo["z_end_ras"])
                  xy_hi = t_hi["track_start_xy"] + t_hi["slope"] * (z_m - t_hi["z_start_ras"])
                  if float(np.hypot(xy_lo[0] - xy_hi[0], xy_lo[1] - xy_hi[1])) <= 1.85:
                    pkg_m = merge_two_trajs(t_lo, t_hi, cand_thresh)
                    if pkg_m is not None:
                      out[idx_p] = pkg_m
                    dup = True
                    break
          if not dup:
            out.append(t)
        return out

      # --- PASS 1: Full-ROI Detection & Stage 1 3D Multi-Head Neural Network Segmentation ---
      update_progress(12, "Stage 1/6: Computing 3D radial symmetry & 3D U-Net labelmap (Pass 1)...")
      res1 = rust_compute_3d_sym_response(vol_work, sp_i, sp_j, i0, i1, j0, j1, k0, K, body_thresh)
      if res1 is None:
        res1 = self._compute_3d_sym_response_numpy(vol_work, sp_i, sp_j, i0, i1, j0, j1, k0, K, body_thresh)
      resp_full, cs_full, rsym_full = res1
      resp1 = np.ascontiguousarray(resp_full[k0:K, j0:j1, i0:i1], dtype=np.float32)
      cs1 = np.ascontiguousarray(cs_full[k0:K, j0:j1, i0:i1], dtype=np.float32)
      rsym1 = np.ascontiguousarray(rsym_full[k0:K, j0:j1, i0:i1], dtype=np.float32)
      K_sub = K - k0

      nn_preds = None
      try:
        nn_preds = run_nn_3d_segmentation(vol_work, sp_i, sp_j, sp_k, i0, i1, j0, j1, k0, K, resp1, cs1, rsym1)
        if nn_preds is not None and not script:
          create_or_update_slicer_labelmap(volumeNode, nn_preds, k0, i0, i1, j0, j1, K, J, I, threshold=0.40)
      except Exception:
        nn_preds = None

      pos_vals = resp1[resp1 > 0.5]
      cand_thresh = max(1.2, float(np.percentile(pos_vals, 50))) if len(pos_vals) else 1.2

      update_progress(34, "Stage 2/6: Seeding & tracking candidate 3D trajectories (Pass 1)...")
      trajs1 = track_pass(resp1, cs1, rsym1, i0, j0, cand_thresh, pass_idx=1)
      if not trajs1:
        return []
      prelim1 = dedup_trajs(trajs1)

      update_progress(48, "Stage 3/6: Estimating needle bundle consensus & inpainting locked tracts...")
      k_tol = 4 if sp_k <= 2.1 else max(3, int(round(7.5 / sp_k)))
      pelvic_cands = [
          t for t in prelim1
          if abs(t["base_xy"][0] - fov_center_ras[0]) <= 48.0
          and abs(t["base_xy"][1] - fov_center_ras[1]) <= 64.0
      ]
      base_pool = pelvic_cands if len(pelvic_cands) >= 3 else prelim1
      low_k_cands = [t for t in base_pool if t["k_min_rel"] <= k_tol]
      vote_pool = low_k_cands[:min(30, len(low_k_cands))] if len(low_k_cands) >= 3 else base_pool[:min(30, len(base_pool))]

      raw_arr = np.array([t["raw_score"] for t in vote_pool])
      ref_raw = float(np.percentile(raw_arr[:min(10, len(raw_arr))], 65)) if len(raw_arr) >= 3 else float(raw_arr[0])
      clamped_w = np.sqrt(np.minimum(raw_arr, 1.5 * ref_raw))
      bases = np.array([t["base_xy"] for t in vote_pool])
      slopes = np.array([t["slope"] for t in vote_pool])

      best_cluster_score = -1e18
      best_idx = 0
      for idx in range(len(vote_pool)):
        d_xy = np.hypot(bases[:, 0] - bases[idx, 0], bases[:, 1] - bases[idx, 1])
        d_sl = np.hypot(slopes[:, 0] - slopes[idx, 0], slopes[:, 1] - slopes[idx, 1])
        inliers = (d_xy <= 42.0) & (d_sl <= 0.25)
        dx_fov = float(bases[idx, 0] - fov_center_ras[0])
        dy_fov = float(bases[idx, 1] - fov_center_ras[1])
        fov_prior = np.exp(-0.5 * ((dx_fov / 35.0) ** 2 + (dy_fov / 48.0) ** 2))
        c_score = float(np.sum(clamped_w[inliers])) * (int(np.sum(inliers)) ** 0.80) * fov_prior
        if c_score > best_cluster_score:
          best_cluster_score = c_score
          best_idx = idx

      seed_center = bases[best_idx]
      seed_slope = slopes[best_idx]
      d_seed_xy = np.hypot(bases[:, 0] - seed_center[0], bases[:, 1] - seed_center[1])
      d_seed_sl = np.hypot(slopes[:, 0] - seed_slope[0], slopes[:, 1] - seed_slope[1])
      inlier_mask = ((d_seed_xy <= 44.0) & (d_seed_sl <= 0.28)) | (
          (d_seed_xy <= 56.0) & (d_seed_sl <= 0.18) & (raw_arr >= 0.25 * ref_raw)
      )
      if np.sum(inlier_mask) >= 3:
        inlier_bases = bases[inlier_mask]
        inlier_w = clamped_w[inlier_mask]
        med_center = np.average(inlier_bases, axis=0, weights=inlier_w)
        med_slope = np.median(slopes[inlier_mask], axis=0)
        med_slope_ij = np.median(np.array([vote_pool[i]["slope_ij"] for i in np.where(inlier_mask)[0]]), axis=0)
        bbox_mid = 0.5 * (np.min(inlier_bases, axis=0) + np.max(inlier_bases, axis=0))
      else:
        med_center = seed_center
        med_slope = seed_slope
        med_slope_ij = vote_pool[best_idx]["slope_ij"]
        bbox_mid = seed_center

      spatial_inliers = [
          t for t in base_pool
          if np.hypot(t["base_xy"][0] - bbox_mid[0], t["base_xy"][1] - bbox_mid[1]) <= 44.0
          and np.hypot(t["slope"][0] - med_slope[0], t["slope"][1] - med_slope[1]) <= 0.30
      ]
      if spatial_inliers:
        bundle_k_min = int(np.percentile([t["k_min_rel"] for t in spatial_inliers[:min(12, len(spatial_inliers))]], 25))
      else:
        bundle_k_min = 0

      is_deep_scan = ((K - k0) * sp_k > 185.0)
      valid_k_inliers = [
          t for t in spatial_inliers
          if t["k_min_rel"] <= bundle_k_min + k_tol
          or (t["z_span_tracked"] >= 28.0 and t["mean_r"] >= cand_thresh * 1.8)
      ]
      if len(valid_k_inliers) >= 2:
        sc_sorted = sorted([t["raw_score"] for t in valid_k_inliers], reverse=True)
        bundle_max_raw = float(min(sc_sorted[0], 1.40 * sc_sorted[min(2, len(sc_sorted) - 1)]))
      elif len(valid_k_inliers) == 1:
        bundle_max_raw = float(valid_k_inliers[0]["raw_score"])
      else:
        bundle_max_raw = ref_raw

      core_inliers = [t for t in valid_k_inliers if t["raw_score"] >= 0.14 * bundle_max_raw]
      if len(core_inliers) >= 3:
        c_bases = np.array([t["base_xy"] for t in core_inliers])
        bbox_mid = 0.5 * (np.min(c_bases, axis=0) + np.max(c_bases, axis=0))
        med_center = np.median(c_bases, axis=0)
        c_slopes = np.array([t["slope"] for t in core_inliers])
        med_slope = np.median(c_slopes, axis=0)
        med_slope_ij = np.median(np.array([t["slope_ij"] for t in core_inliers]), axis=0)
        r_max_core = float(np.max(np.hypot(c_bases[:, 0] - bbox_mid[0], c_bases[:, 1] - bbox_mid[1])))
        adaptive_rad = float(np.clip(max(37.5, r_max_core + 8.5), 37.5, 48.0))
        sdev_p75 = float(np.percentile(np.hypot(c_slopes[:, 0] - med_slope[0], c_slopes[:, 1] - med_slope[1]), 75))
        adaptive_sdev = float(np.clip(max(0.28, 2.4 * sdev_p75), 0.28, 0.38))
      else:
        adaptive_rad = 38.5
        adaptive_sdev = 0.34

      def passes_bundle_filter(t, pass_num=1):
        max_fov_dx = 56.5 if is_deep_scan else 49.5
        if abs(t["base_xy"][0] - fov_center_ras[0]) > max_fov_dx or abs(t["base_xy"][1] - fov_center_ras[1]) > 66.0:
          return False
        dist_start = float(np.hypot(t["track_start_xy"][0] - bbox_mid[0], t["track_start_xy"][1] - bbox_mid[1]))
        dist_end = float(np.hypot(t["track_end_xy"][0] - bbox_mid[0], t["track_end_xy"][1] - bbox_mid[1]))
        if is_deep_scan and (dist_start > adaptive_rad + 12.5 or dist_end > adaptive_rad + 20.0):
          return False
        is_deep_uterine = (
            is_deep_scan
            and t["k_max_rel"] * sp_k >= 105.0
            and t["z_span_tracked"] >= 22.0
            and t["mean_r"] >= cand_thresh * 1.45
        )
        if is_deep_scan and t["k_min_rel"] > bundle_k_min + int(round(36.0 / sp_k)) and t["z_span_tracked"] < 42.0 and not is_deep_uterine:
          return False
        if is_deep_scan and t["base_xy"][1] < med_center[1] - 15.0 and t["k_max_rel"] * sp_k < 58.0 and t["z_span_tracked"] < 45.0:
          return False
        dist_mid = float(np.hypot(t["base_xy"][0] - bbox_mid[0], t["base_xy"][1] - bbox_mid[1]))
        dist_med = float(np.hypot(t["base_xy"][0] - med_center[0], t["base_xy"][1] - med_center[1]))
        slope_dev = float(np.hypot(t["slope"][0] - med_slope[0], t["slope"][1] - med_slope[1]))
        k_ok = (
            t["k_min_rel"] <= bundle_k_min + k_tol
            or (
                t["z_span_tracked"] >= 28.0
                and t["mean_r"] >= cand_thresh * 1.6
                and t["raw_score"] >= 0.16 * bundle_max_raw
                and slope_dev <= min(0.22, adaptive_sdev)
            )
            or (is_deep_uterine and t["raw_score"] >= 0.12 * bundle_max_raw and slope_dev <= min(0.24, adaptive_sdev))
            or (sp_k >= 4.5 and t["k_min_rel"] <= 11 and t["z_span_tracked"] >= 24.0 and t["raw_score"] >= 0.22 * bundle_max_raw)
        )
        if not k_ok:
          return False
        if pass_num == 1:
          max_sdev = adaptive_sdev if (dist_mid <= 32.0 or len(core_inliers) >= 12) else min(adaptive_sdev, 0.23)
          return (
              dist_mid <= adaptive_rad
              and dist_med <= adaptive_rad + 18.0
              and slope_dev <= max_sdev
              and t["raw_score"] >= 0.10 * bundle_max_raw
          )
        else:
          return (
              dist_mid <= adaptive_rad
              and dist_med <= adaptive_rad + 15.0
              and slope_dev <= min(0.25, adaptive_sdev * 0.85)
              and t["z_span_tracked"] >= (11.0 if K_sub <= 28 else 16.5)
              and t["rmse"] <= 1.25
              and t["raw_score"] >= 0.10 * bundle_max_raw
          )

      locked = [t for t in prelim1 if passes_bundle_filter(t, pass_num=1)]

      # --- PASS 2 & PASS 3: Tight-ROI Volume Tract Inpainting, Guided Re-Detection & JEV Control-Point Association ---
      if len(locked) >= 2:
        corners_ras = np.array([
            [bbox_mid[0] - (adaptive_rad + 8.0), bbox_mid[1] - (adaptive_rad + 8.0), z_base_ras, 1.0],
            [bbox_mid[0] + (adaptive_rad + 8.0), bbox_mid[1] + (adaptive_rad + 8.0), z_base_ras, 1.0],
            [bbox_mid[0] - (adaptive_rad + 8.0), bbox_mid[1] + (adaptive_rad + 8.0), z_base_ras + 90.0, 1.0],
            [bbox_mid[0] + (adaptive_rad + 8.0), bbox_mid[1] - (adaptive_rad + 8.0), z_base_ras + 90.0, 1.0],
        ])
        corners_ijk = (ras_to_ijk @ corners_ras.T).T
        p2_i0 = max(i0, int(np.floor(np.min(corners_ijk[:, 0]))) - 6)
        p2_i1 = min(i1, int(np.ceil(np.max(corners_ijk[:, 0]))) + 6)
        p2_j0 = max(j0, int(np.floor(np.min(corners_ijk[:, 1]))) - 6)
        p2_j1 = min(j1, int(np.ceil(np.max(corners_ijk[:, 1]))) + 6)

        if p2_i1 - p2_i0 >= 24 and p2_j1 - p2_j0 >= 24:
          ri_in = max(1, int(round(1.35 / sp_i)))
          rj_in = max(1, int(round(1.35 / sp_j)))
          yy_in, xx_in = np.mgrid[-rj_in:rj_in + 1, -ri_in:ri_in + 1]
          disk_in = ((xx_in * sp_i) ** 2 + (yy_in * sp_j) ** 2) <= (1.35 ** 2)

          saved_patches = [] if not vol_work_owned else None
          for t in locked:
            ks_r = np.arange(max(0, t["k_min_rel"] - 1), min(K_sub, t["k_max_rel"] + 2))
            ics = np.round(np.polyval(t["pi_abs"], ks_r)).astype(int)
            jcs = np.round(np.polyval(t["pj_abs"], ks_r)).astype(int)
            for kr, ic_a, jc_a in zip(ks_r, ics, jcs):
              if p2_i0 <= ic_a < p2_i1 and p2_j0 <= jc_a < p2_j1:
                k_a = kr + k0
                bg_v = float(rsym1[kr, jc_a - j0, ic_a - i0])
                if bg_v > 5.0:
                  j_lo, j_hi = max(0, jc_a - rj_in), min(J, jc_a + rj_in + 1)
                  i_lo, i_hi = max(0, ic_a - ri_in), min(I, ic_a + ri_in + 1)
                  d_sub = disk_in[(j_lo - (jc_a - rj_in)):(j_hi - (jc_a - rj_in)), (i_lo - (ic_a - ri_in)):(i_hi - (ic_a - ri_in))]
                  patch = vol_work[k_a, j_lo:j_hi, i_lo:i_hi]
                  if saved_patches is not None:
                    saved_patches.append((k_a, j_lo, j_hi, i_lo, i_hi, patch.copy()))
                  patch[d_sub] = np.maximum(patch[d_sub], bg_v * 0.92)

          update_progress(60, "Stage 4/6: Re-computing symmetry & guided tracking (Pass 2 & JEV Pass 3)...")
          try:
            res2 = rust_compute_3d_sym_response(vol_work, sp_i, sp_j, p2_i0, p2_i1, p2_j0, p2_j1, k0, K, body_thresh)
            if res2 is None:
              res2 = self._compute_3d_sym_response_numpy(vol_work, sp_i, sp_j, p2_i0, p2_i1, p2_j0, p2_j1, k0, K, body_thresh)
          finally:
            if saved_patches:
              for k_a, j_lo, j_hi, i_lo, i_hi, orig_p in reversed(saved_patches):
                vol_work[k_a, j_lo:j_hi, i_lo:i_hi] = orig_p
          resp2_full, cs2_full, rsym2_full = res2
          resp2 = np.ascontiguousarray(resp2_full[k0:K, p2_j0:p2_j1, p2_i0:p2_i1], dtype=np.float32)
          cs2 = np.ascontiguousarray(cs2_full[k0:K, p2_j0:p2_j1, p2_i0:p2_i1], dtype=np.float32)
          rsym2 = np.ascontiguousarray(rsym2_full[k0:K, p2_j0:p2_j1, p2_i0:p2_i1], dtype=np.float32)
          J2, I2 = p2_j1 - p2_j0, p2_i1 - p2_i0

          ri_m = max(1, int(round(1.60 / sp_i)))
          rj_m = max(1, int(round(1.60 / sp_j)))
          yy_m, xx_m = np.mgrid[-rj_m:rj_m + 1, -ri_m:ri_m + 1]
          disk_m = ((xx_m * sp_i) ** 2 + (yy_m * sp_j) ** 2) <= (1.60 ** 2)
          all_kr = np.arange(K_sub)

          for t in locked:
            ics = np.round(np.polyval(t["pi_abs"], all_kr)).astype(int) - p2_i0
            jcs = np.round(np.polyval(t["pj_abs"], all_kr)).astype(int) - p2_j0
            for kr, ic_s, jc_s in zip(all_kr, ics, jcs):
              if 0 <= ic_s < I2 and 0 <= jc_s < J2:
                j_lo, j_hi = max(0, jc_s - rj_m), min(J2, jc_s + rj_m + 1)
                i_lo, i_hi = max(0, ic_s - ri_m), min(I2, ic_s + ri_m + 1)
                d_sub = disk_m[(j_lo - (jc_s - rj_m)):(j_hi - (jc_s - rj_m)), (i_lo - (ic_s - ri_m)):(i_hi - (ic_s - ri_m))]
                resp2[kr, j_lo:j_hi, i_lo:i_hi][d_sub] = 0.0

          trajs2 = track_pass(resp2, cs2, rsym2, p2_i0, p2_j0, cand_thresh, prior_slope_ij=med_slope_ij, pass_idx=2)
          trajs2_valid = [t for t in trajs2 if passes_bundle_filter(t, pass_num=2)]
          combined = dedup_trajs(trajs2_valid, existing_list=locked)

          # --- PASS 3: JEV Auto-Manual Control-Point Association ---
          for t in combined[len(locked):]:
            ics = np.round(np.polyval(t["pi_abs"], all_kr)).astype(int) - p2_i0
            jcs = np.round(np.polyval(t["pj_abs"], all_kr)).astype(int) - p2_j0
            for kr, ic_s, jc_s in zip(all_kr, ics, jcs):
              if 0 <= ic_s < I2 and 0 <= jc_s < J2:
                j_lo, j_hi = max(0, jc_s - rj_m), min(J2, jc_s + rj_m + 1)
                i_lo, i_hi = max(0, ic_s - ri_m), min(I2, ic_s + ri_m + 1)
                d_sub = disk_m[(j_lo - (jc_s - rj_m)):(j_hi - (jc_s - rj_m)), (i_lo - (ic_s - ri_m)):(i_hi - (ic_s - ri_m))]
                resp2[kr, j_lo:j_hi, i_lo:i_hi][d_sub] = 0.0

          ctl_cut = cand_thresh * 0.55
          ctl_pts = []
          ref_bases = np.array([t["base_xy"] for t in combined])
          ref_slopes = np.array([t["slope"] for t in combined])
          max_ctl_kr = min(K_sub - 1, int(round(105.0 / sp_k)))
          lm2 = None
          for kr in range(0, max_ctl_kr):
            res_nms3 = rust_find_2d_nms_peaks(resp2[kr], nms_px, ctl_cut)
            if res_nms3 is not None:
              jj, ii, _ = res_nms3
            else:
              if lm2 is None:
                lm2 = resp2 == maximum_filter(resp2, size=(1, 2 * nms_px + 1, 2 * nms_px + 1))
              jj, ii = np.where(lm2[kr] & (resp2[kr] >= ctl_cut))
            if len(ii) == 0:
              continue
            order_r = np.argsort(-resp2[kr, jj, ii])[:45]
            for idx_p in order_r:
              i_a = float(ii[idx_p] + p2_i0)
              j_a = float(jj[idx_p] + p2_j0)
              r_v = float(resp2[kr, jj[idx_p], ii[idx_p]])
              pt_ras = rot_mat @ np.array([i_a, j_a, float(kr + k0)]) + trans_vec
              dz_b = float(pt_ras[2] - z_base_ras)
              approx_b = pt_ras[:2] - med_slope * dz_b
              d_ref = np.hypot(ref_bases[:, 0] - approx_b[0], ref_bases[:, 1] - approx_b[1])
              w_ref = 1.0 / (d_ref + 4.0) ** 2
              loc_slope = np.average(ref_slopes, axis=0, weights=w_ref)
              x0_p = float(pt_ras[0] - loc_slope[0] * dz_b)
              y0_p = float(pt_ras[1] - loc_slope[1] * dz_b)
              if math.hypot(x0_p - bbox_mid[0], y0_p - bbox_mid[1]) <= adaptive_rad:
                ctl_pts.append((i_a, j_a, kr, r_v, float(pt_ras[0]), float(pt_ras[1]), float(pt_ras[2]), x0_p, y0_p))

          if len(ctl_pts) >= min_pts_slices:
            ctl_arr = np.array(ctl_pts, dtype=np.float64)
            used_ctl = np.zeros(len(ctl_pts), dtype=bool)
            jev_trajs = []
            for seed_i in np.argsort(-ctl_arr[:, 3]):
              if used_ctl[seed_i]:
                continue
              sx0, sy0 = ctl_arr[seed_i, 7], ctl_arr[seed_i, 8]
              dists_proj = np.hypot(ctl_arr[:, 7] - sx0, ctl_arr[:, 8] - sy0)
              cand_idxs = np.where((~used_ctl) & (dists_proj <= 1.85))[0]
              if len(cand_idxs) < min_pts_slices:
                continue
              by_k = {}
              for ci in cand_idxs:
                kr_c = int(ctl_arr[ci, 2])
                if kr_c not in by_k or ctl_arr[ci, 3] > ctl_arr[by_k[kr_c], 3]:
                  by_k[kr_c] = ci
              ks_j = sorted(by_k.keys())
              if len(ks_j) < (min_pts_slices + 1):
                continue
              sel_idxs = [by_k[k] for k in ks_j]
              arr_j = np.column_stack([ctl_arr[sel_idxs, 0], ctl_arr[sel_idxs, 1], ctl_arr[sel_idxs, 2] + k0, ctl_arr[sel_idxs, 3]])
              ras_j = ctl_arr[sel_idxs, 4:7]
              shaft_p75 = fast_pct(arr_j[:, 3], 75)
              ras_j, arr_j, ks_j = trim_trajectory_inside_out(ras_j, arr_j, ks_j, shaft_p75)
              if len(ks_j) < (min_pts_slices + 1):
                continue
              z_span_j = float(ras_j[-1, 2] - ras_j[0, 2])
              if z_span_j < (13.0 if K_sub <= 28 else 20.0):
                continue
              if min(ks_j) > bundle_k_min + k_tol + 2 and z_span_j < 32.0:
                continue
              if is_1d_trench_artifact(arr_j, min(ks_j)):
                continue
              pkg_j = fit_and_package_traj(ras_j, arr_j, ks_j, pass_idx=3, cand_thresh=cand_thresh)
              if pkg_j is not None and pkg_j["rmse"] <= 1.15 and passes_bundle_filter(pkg_j, pass_num=3):
                jev_trajs.append(pkg_j)
                for ci in cand_idxs:
                  used_ctl[ci] = True
            combined = dedup_trajs(jev_trajs, existing_list=combined)
        else:
          combined = locked
      else:
        combined = locked

      # --- HybridNeedleNet (14-Feature Residual SiLU MLP + JEV Orientation Field) Scoring ---
      update_progress(82, "Stage 5/6: Scoring trajectories with HybridNeedleNet & JEV field...")
      all_bases = np.array([t["base_xy"] for t in combined]) if combined else np.zeros((0, 2))
      all_slopes = np.array([t["slope"] for t in combined]) if combined else np.zeros((0, 2))
      all_scores = np.array([t["raw_score"] for t in combined]) if combined else np.zeros(0)

      filtered = []
      for idx_t, t in enumerate(combined):
        raw_rel = float(t["raw_score"] / max(1e-6, bundle_max_raw))
        heur_conf = min(99.9, 100.0 * (raw_rel ** 0.45))
        if heur_conf < 35.0:
          continue

        log_raw_rel = math.log1p(raw_rel * 5.0)
        mean_r_rel = float(t["mean_r"] / max(1e-6, cand_thresh))
        p75_r_rel = float(t["p75_r"] / max(1e-6, cand_thresh))
        high_frac = float(t["high_frac"])
        rmse_norm = float(t["rmse"] / 1.45)
        z_span_norm = float(t["z_span_tracked"] / 60.0)
        len_norm = float(t["length_mm"] / 80.0)
        k_min_mm_norm = float(max(0, t["k_min_rel"] - bundle_k_min) * sp_k / 30.0)
        k_span = max(1, t["k_max_rel"] - t["k_min_rel"] + 1)
        continuity = float(t["n_slices"] / k_span)
        dist_mid = float(np.hypot(t["base_xy"][0] - bbox_mid[0], t["base_xy"][1] - bbox_mid[1]))
        dist_med = float(np.hypot(t["base_xy"][0] - med_center[0], t["base_xy"][1] - med_center[1]))
        slope_dev = float(np.hypot(t["slope"][0] - med_slope[0], t["slope"][1] - med_slope[1]))
        dist_mid_norm = dist_mid / max(1.0, adaptive_rad)
        dist_med_norm = dist_med / max(1.0, adaptive_rad + 15.0)
        slope_dev_norm = slope_dev / max(0.05, adaptive_sdev)

        if len(combined) >= 2:
          d_other = np.hypot(all_bases[:, 0] - t["base_xy"][0], all_bases[:, 1] - t["base_xy"][1])
          d_other[idx_t] = 1e9
          w_jev = np.minimum(all_scores, bundle_max_raw) / ((d_other + 5.0) ** 2)
          w_jev[idx_t] = 0.0
          if float(np.sum(w_jev)) > 1e-9:
            loc_sl = np.average(all_slopes, axis=0, weights=w_jev)
            jev_sdev = float(np.hypot(t["slope"][0] - loc_sl[0], t["slope"][1] - loc_sl[1]))
          else:
            jev_sdev = slope_dev
        else:
          jev_sdev = slope_dev

        jev_sdev_norm = min(2.0, jev_sdev / 0.25)
        if t["pass"] == 3:
          if not (t["z_span_tracked"] >= 45.0 and t["rmse"] <= 0.75 and jev_sdev_norm <= 0.20 and heur_conf >= 47.0):
            continue

        feat_14 = np.array([
            log_raw_rel,
            min(3.0, raw_rel),
            min(10.0, mean_r_rel) / 5.0,
            min(12.0, p75_r_rel) / 5.0,
            high_frac,
            rmse_norm,
            min(2.5, z_span_norm),
            min(2.5, len_norm),
            min(2.0, k_min_mm_norm),
            continuity,
            dist_mid_norm,
            dist_med_norm,
            slope_dev_norm,
            jev_sdev_norm,
        ], dtype=np.float64)
        nn_prob, conf = score_trajectory_hybrid_nn(feat_14)

        # Fuse Stage 1 3D U-Net Labelmap & Peak probabilities into trajectory verification
        if nn_preds is not None:
          pm_arr = nn_preds["p_mask"]
          pp_arr = nn_preds["p_peak"]
          po_arr = nn_preds["p_obt"]
          H_nn, W_nn = pm_arr.shape[1], pm_arr.shape[2]
          pm_vals, pp_vals = [], []
          for kr_v in range(max(0, t["k_min_rel"]), min(K_sub, t["k_max_rel"] + 1)):
            ic_v = int(round(float(np.polyval(t["pi_abs"], kr_v)) - i0))
            jc_v = int(round(float(np.polyval(t["pj_abs"], kr_v)) - j0))
            if 1 <= ic_v < W_nn - 1 and 1 <= jc_v < H_nn - 1:
              pm_vals.append(max(float(np.max(pm_arr[kr_v, jc_v - 1:jc_v + 2, ic_v - 1:ic_v + 2])), float(np.max(po_arr[kr_v, jc_v - 1:jc_v + 2, ic_v - 1:ic_v + 2]))))
              pp_vals.append(float(np.max(pp_arr[kr_v, jc_v - 1:jc_v + 2, ic_v - 1:ic_v + 2])))
          if len(pm_vals) >= 3:
            mean_pm_v = float(np.mean(pm_vals))
            p75_pm_v = float(np.percentile(pm_vals, 75))
            mean_pp_v = float(np.mean(pp_vals))
            if mean_pm_v < 0.14 and p75_pm_v < 0.22 and mean_pp_v < 0.12 and conf < 78.0:
              continue
            if mean_pm_v >= 0.58 and mean_pp_v >= 0.44 and conf >= 38.0:
              conf = max(conf, min(99.5, 58.0 + 38.0 * mean_pm_v))

        if conf >= (25.0 if nn_preds is not None else float(minConfidence)):
          t["nn_prob"] = nn_prob
          t["confidence"] = conf
          filtered.append(t)

      # Stage 2 Direct 3D U-Net Labelmap + Tangent-Field Tube Reconstruction + NeedleSliceMPNN + SynthBrachy3T TipAwareBundleGraphTransformer
      if nn_preds is not None:
        try:
          nn_tubes = extract_nn_labelmap_tubes(
              nn_preds, resp1, k0, i0, j0, sp_i, sp_j, sp_k,
              ijk_to_ras_mat, ras_to_ijk, filtered, include_obturator=True,
          )
          for nt in nn_tubes:
            nt["ras"] = nt["points"]
            filtered.append(nt)
          filtered = run_mpnn_and_graph_transformer(
              vol_work, nn_preds, resp1, cs1, rsym1,
              k0, i0, j0, sp_i, sp_j, sp_k,
              ijk_to_ras_mat, ras_to_ijk, filtered,
              min_confidence=float(minConfidence),
              script=script,
          )
        except Exception:
          filtered = [t for t in filtered if t.get("confidence", 0.0) >= float(minConfidence)]

      if renderNeedles and widget is not None:
        update_progress(88, f"Stage 6/6: Rendering {len(filtered)} detected needles in 3D scene...")
        self.deleteNeedleDetectionModelsFromScene()
        self.cleanTable()
        widget.initTableView()
        for idx, t in enumerate(filtered):
          if prog_dialog is not None and (idx % 3 == 0 or idx == len(filtered) - 1):
            pct_r = 88 + int(round(11.0 * (idx + 1) / max(1, len(filtered))))
            update_progress(pct_r, f"Stage 6/6: Rendering needle {idx + 1}/{len(filtered)} in 3D scene...")
          ras_arr = t["ras"]
          # Sample up to 15 control points from tip (high Z) to base (low Z)
          n_ctrl = min(15, len(ras_arr))
          sample_idx = np.linspace(len(ras_arr) - 1, 0, n_ctrl, dtype=int)
          ctrl_pts = ras_arr[sample_idx].tolist()
          conf_val = float(t.get("confidence", 0.0))
          len_val = float(t.get("length_mm", 0.0))
          comment = f"Conf: {conf_val:.1f}% | L: {len_val:.1f} mm"
          self.addNeedleToScene(ctrl_pts, idx, "Detection", script=script, comment=comment, confidence=conf_val)
        if filtered:
          widget.deleteNeedleButton.setEnabled(1)
          widget.resetDetectionButton.setEnabled(1)
          widget.fiducialButton.setEnabled(1)
          if hasattr(widget, "_NeedleFinderWidget__reportFrame"):
            widget._NeedleFinderWidget__reportFrame.collapsed = 0

      elapsed = time.perf_counter() - t0
      p1_cnt = sum(1 for x in filtered if x.get("pass", 1) == 1)
      p2_cnt = sum(1 for x in filtered if x.get("pass", 1) == 2)
      p3_cnt = sum(1 for x in filtered if x.get("pass", 1) == 3)
      print(f"[NeedleFinder] Auto-detected {len(filtered)} needles (P1={p1_cnt}, P2={p2_cnt}, JEV={p3_cnt}) in {elapsed:.2f}s (k_lim={k_lim}, minConf={minConfidence:.1f}%)")
      return filtered
    finally:
      close_progress()
