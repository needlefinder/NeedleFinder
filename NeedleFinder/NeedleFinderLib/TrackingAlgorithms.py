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
    rust_find_tip,
    rust_search_cone_13,
    rust_search_cone_15_1,
    rust_search_cone_current_dev,
)
from .LegacyEditorCompat import (
    EditorLib,
    create_fiducial_node,
    get_all_fiducial_nodes,
    get_fiducial_coords,
    set_fiducial_coords,
)
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
      if len(bases) > 0:
        bases2 = np.array(bases)[:, 2]
        i = bases2.argmin()
        asl = self.ijk2ras(bases[i])
        asl = [coord[0], coord[1], asl[2]]
      else:
        asl = [0.0, 0.0, 0.0]
      if not node:
        self.fiducialNode = create_fiducial_node(
            name='template slice position',
            ras=asl,
            color=(0.0, 1.0, 0.0),
            visibility=True,
        )
      else:
        set_fiducial_coords(node, asl)
    if asl[2] == -float("inf"):
      asl = [0.0, 0.0, 0.0]
    return int(round(self.ras2ijk(asl)[2])), asl[2]

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

  def addNeedleToScene(self, controlPoint, colorVar, needleType='Detection', script=False, trans=0, manualName="", bPause=False):
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
    modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
    model.SetAttribute("nth", str(nth))

    if needleType != 'Validation' and not script:
      self.addNeedleToTable(extract_model_id_number(model.GetID()), label)
