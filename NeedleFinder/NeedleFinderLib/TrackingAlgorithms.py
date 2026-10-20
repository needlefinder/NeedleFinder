# -*- coding: utf-8 -*-
"""Needle detection, tip finding, mechanical deflection models, and curve builders."""

import math
import os
import random
import time
import time as t
import numpy
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
from .LegacyEditorCompat import EditorLib
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


class TrackingAlgorithmsMixin(object):
  """Mixin providing needle tracking, tip optimization, rationality checks, and 3D curve building."""

  def needleDetection(self):
    """
    This solution is optional but not used anymore in the workflow.
    Use the label map of the needle tips
    Apply the island effect
    Extract the coordinates of the islands (self.array2)
    Start a detection for each island (self.needleDetectionThread)
    TODO: multi-processing
    """
    # research #obsolete
    profbox(whoami())
    # Apply Island Effect
    editUtil = EditorLib.EditUtil.EditUtil()
    parameterNode = editUtil.getParameterNode()
    sliceLogic = editUtil.getSliceLogic()
    lm = slicer.app.layoutManager()
    sliceWidget = lm.sliceWidget('Red')
    islandsEffect = EditorLib.IdentifyIslandsEffectOptions()
    islandsEffect.setMRMLDefaults()
    islandsEffect.__del__()
    islandTool = EditorLib.IdentifyIslandsEffectLogic(sliceLogic)
    parameterNode.SetParameter("IslandEffect,minimumSize", '0')
    islandTool.removeIslands()
    # select the image node from the Red slice viewer
    m = vtk.vtkMatrix4x4()
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    volumeNode.GetIJKToRASMatrix(m)
    imageData = volumeNode.GetImageData()
    spacing = volumeNode.GetSpacing()
    # chrono starts
    self.t0 = time.perf_counter()
    # get the coordinates from the label map
    label = self.array2()
    for I in range(len(label)):
      A = label[I]
      colorVar = I  # ??? /(len(label))
      self.needleDetectionThread(A, imageData, colorVar, spacing)
      if slicer.modules.NeedleFinderWidget.autoStopTip.isChecked():
        self.needleDetectionUPThread(A, imageData, colorVar, spacing)

  def findNextStepNumber(self, needleNumber):
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode')
    nbNode = modelNodes.GetNumberOfItems()
    stepValue = 0
    for nthNode in range(nbNode):
        modelNode = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLAnnotationFiducialNode')
        if modelNode.GetAttribute("ValidationNeedle") == "1" and int(modelNode.GetAttribute("NeedleNumber")) == needleNumber:
          if int(modelNode.GetAttribute("NeedleStep")) > stepValue:
            stepValue = int(modelNode.GetAttribute("NeedleStep"))

    return stepValue + 1

  def lockControlPoints(self, needleNumber):
    '''
    Lock every control points except the one of this needle number
    :param needleNumber:
    :return:
    '''
    profprint()
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode')
    nbNode = modelNodes.GetNumberOfItems()
    stepValue = 0
    for nthNode in range(nbNode):
        modelNode = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLAnnotationFiducialNode')
        if modelNode.GetAttribute("ValidationNeedle") == "1" and int(modelNode.GetAttribute("NeedleNumber")) != needleNumber:
          modelNode.SetLocked(1)


  def unlockControlPoints(self, needleNumber):
    '''
    Unlock control points from this needle
    :param needleNumber:
    :return:
    '''
    profprint()
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode')
    nbNode = modelNodes.GetNumberOfItems()
    stepValue = 0
    for nthNode in range(nbNode):
        modelNode = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLAnnotationFiducialNode')
        if modelNode.GetAttribute("ValidationNeedle") == "1" and int(modelNode.GetAttribute("NeedleNumber")) == needleNumber:
          modelNode.SetLocked(0)

  def deleteDuplicateMarkers(self):
    """
    Remove duplice markers
    :param needleNumber:
    :param stepValue:
    :return:
    """
    profprint()
    scene = slicer.mrmlScene
    nodes = scene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode','Fiducials List')
    if nodes.GetNumberOfItems()>1:
      for i in range(1,nodes.GetNumberOfItems()):
        hnode = nodes.GetItemAsObject(i)
        if(hnode):
          children = vtk.vtkCollection()
          hnode.GetAssociatedChildrenNodes(children, "vtkMRMLAnnotationFiducialNode")

          for i in range(children.GetNumberOfItems()):
            annotNode = children.GetItemAsObject(i)

            oneToOneHierarchyNode = hnode.GetAssociatedHierarchyNode(annotNode.GetScene(), annotNode.GetID())
            pointDisplayNode = annotNode.GetAnnotationPointDisplayNode()
            textDisplayNode = annotNode.GetAnnotationTextDisplayNode()
            storageNode = annotNode.GetStorageNode()
            if (oneToOneHierarchyNode):
              try:
                scene.RemoveNode(oneToOneHierarchyNode)
              except:
                logprint(msg = 'pb removing onetoone')
            if (pointDisplayNode):
              try:
                scene.RemoveNode(pointDisplayNode)
              except:
                logprint(msg = 'pb removing pointDisplayNode')
            if (textDisplayNode):
              try:
                scene.RemoveNode(textDisplayNode)
              except:
                logprint(msg = 'pb removing textDisplayNode')
            if (storageNode):
              try:
                scene.RemoveNode(storageNode)
              except:
                logprint(msg = 'pb removing storageNode')
            if(annotNode):
              try:
                scene.RemoveNode(annotNode)
              except:
                logprint(msg = 'pb removing annotNode')

    # look for duplicate list and delete it if necessary
    nodes = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode','Fiducials List')
    if nodes.GetNumberOfItems()>1:
      for i in range(1,nodes.GetNumberOfItems()):
        try:
          slicer.mrmlScene.RemoveNode(nodes.GetItemAsObject(i))
        except:
          pass

    nodes = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode','All Annotations')
    if nodes.GetNumberOfItems()>1:
      for i in range(1,nodes.GetNumberOfItems()):
        try:
          slicer.mrmlScene.RemoveNode(nodes.GetItemAsObject(i))
        except:
          pass

  def deleteDuplicateNeedles(self):
    '''
    Remove duplicate needle by filtering names
    :return:
    '''
    profprint()
    nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    for i in range(nodes.GetNumberOfItems()):
      node = nodes.GetItemAsObject(i)
      name = node.GetName()
      if len(name.split('manual-seg'))>1:
        if len(name.split('_'))>2:
          slicer.mrmlScene.RemoveNode(node)


  def placeNeedleShaftEvalMarker(self, A, needleNumber, stepValue, type = 'ijk', createPoint=1):
    """
    Add a fiducial point to the vtkMRMLScence, where the mouse click was triggered. The fiducial points reprents a control
    point for a manually segmented needle (validation needle)

    :param A: RAS coordinates of the mouse click
    :param imageData: volumeNode.GetImageDate()
    :param colorVar: color of the fiducial point
    :param spacing: volumeNode.GetSpacing()
    :return: tableValueCtrPt : table containing control points of every validation needle
    """
    # productive #onClick
    profprint()
    pointName = '.' + str(needleNumber) + "-" + str(stepValue)
    pointName2 = '.' + str(needleNumber) + "-" + str(stepValue) + '\r'
    if getNode(pointName) == None and getNode(pointName2) == None and createPoint == 1:
      print('creating fiducial point...')
      # if createPoint == 1:
      fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      # stepValue = self.findNextStepNumber(widget.editNeedleTxtBox.value)
      fiducial.SetName(pointName)
      fiducial.Initialize(slicer.mrmlScene)
      if type=='ijk':
        fiducial.SetFiducialCoordinates(self.ijk2ras(A))
      else:
        fiducial.SetFiducialCoordinates(A)
      fiducial.SetAttribute('ValidationNeedle', '1')
      fiducial.SetAttribute('NeedleNumber', str(needleNumber))
      fiducial.SetAttribute('NeedleStep', str(stepValue))

      nth = int(needleNumber)
      # print nth
      displayNode = fiducial.GetDisplayNode()
      displayNode.SetGlyphScale(2)
      displayNode.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
      textNode = fiducial.GetAnnotationTextDisplayNode()
      textNode.SetTextScale(4)
      textNode.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])

    self.addPointToTable(pointName, needleNumber, stepValue, None, sortTable= createPoint)

  def removeNeedleShaftEvalMarker(self, listArgs):
    """
    Remove a control points from the table and from the scene
    :param listArgs: [id, needlenumber, needlestep]
    :return:
    """
    profprint()
    print(listArgs)
    ID = listArgs[0]
    profprint()

    pointToRemove = getNode(ID)
    print(pointToRemove.GetID())
    slicer.mrmlScene.RemoveNode(pointToRemove)
    self.drawValidationNeedles()
    self.deletePointFromTable(ID)

    return 0

  def obturatorNeedle(self, A, imageData, colorVar, spacing):
    """ Use the mouse click coordinates to draw obturator needles:
    * For the first obturator needle, two points are necessary: give both extremities of an obturator needle
    * For the following needles, only give the tip of the needle. It will draw parallel to the first obturator needle
    and having the same length

    :param A: RAS coordinates of the mouse click
    :param imageData: volumeNode.GetImageDate()
    :param colorVar: color of the fiducial point
    :param spacing: volumeNode.GetSpacing()
    :return: obturator needle
    """
    # productive
    profprint()
    widget = slicer.modules.NeedleFinderWidget

    fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')

    fiducial.Initialize(slicer.mrmlScene)
    fiducial.SetFiducialCoordinates(self.ijk2ras(A))
    fiducial.SetAttribute('ObturatorNeedle', '1')
    if widget.obtuNeedle <= 1:
      needleNumber = 0
    else:
      needleNumber = widget.obtuNeedle - 1
      needleStep = 0
    if widget.obtuNeedle == 1:
      needleStep = 1
    elif widget.obtuNeedle == 0:
      needleStep = 0

    fiducial.SetAttribute('NeedleNumber', str(needleNumber))
    fiducial.SetAttribute('NeedleStep', str(needleStep))
    fiducial.SetName('.obtu-' + str(needleNumber))

    nth = int(needleNumber)

    displayNode = fiducial.GetDisplayNode()
    displayNode.SetGlyphScale(2)
    displayNode.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
    textNode = fiducial.GetAnnotationTextDisplayNode()
    textNode.SetTextScale(4)
    textNode.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])

    widget.obtuNeedleValueCtrPt[needleNumber][needleStep] = self.ijk2ras(A)
    # print self.ijk2ras(A)
    # print "ctr pt: ",self.obtuNeedleValueCtrPt

    if needleNumber >= 1:
      Vx = widget.obtuNeedleValueCtrPt[0][1][0] - widget.obtuNeedleValueCtrPt[0][0][0]
      Vy = widget.obtuNeedleValueCtrPt[0][1][1] - widget.obtuNeedleValueCtrPt[0][0][1]
      Vz = widget.obtuNeedleValueCtrPt[0][1][2] - widget.obtuNeedleValueCtrPt[0][0][2]

      L = float(Vx ** 2 + Vy ** 2 + Vz ** 2) ** 0.5

      E = self.ijk2ras(A)
      Ex = E[0] + 100 * Vx / L
      Ey = E[1] + 100 * Vy / L
      Ez = E[2] + 100 * Vz / L
      widget.obtuNeedleValueCtrPt[needleNumber][1] = [Ex, Ey, Ez]

    self.drawObturatorNeedles()
    widget.stop()
    widget.fiducialObturatorButton.checked = 1
    widget.start(3)

  def objectiveFunction(self, imageData, ijk, radiusNeedleParameter, spacing, gradientPonderation):
    """
    used by needleDetectionUPThread
    """
    # research #frequent
    if frequent: profprint()
    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))
    ijk[0] = int(round(ijk[0]))
    ijk[1] = int(round(ijk[1]))
    ijk[2] = int(round(ijk[2]))
    center = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
    center += imageData.GetScalarComponentAsDouble(ijk[0] + 1, ijk[1], ijk[2], 0)
    center += imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + 1, ijk[2], 0)
    center += imageData.GetScalarComponentAsDouble(ijk[0] - 1, ijk[1], ijk[2], 0)
    center += imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - 1, ijk[2], 0)

    if gradientPonderation != 0:
      g1 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
      g2 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
      g3 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
      g4 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)
      g5 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
      g6 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
      g7 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
      g8 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)

      total = center / float(5) - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / float(8)) * gradientPonderation
    else:
      total = center / float(5)

    return total

  def objectiveFunctionLOG(self, imageData, ijk, radiusNeedleParameter, spacing, gradientPonderation):
    """
    ??? used?
    """
    # obsolete
    print("objectiveFunctionLOG")
    msgbox(whoami())
    """
    not used.
    idea was to test a different objective function
    """
    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))

    center = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
    center += imageData.GetScalarComponentAsDouble(ijk[0] + 1, ijk[1], ijk[2], 0)
    center += imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + 1, ijk[2], 0)
    center += imageData.GetScalarComponentAsDouble(ijk[0] - 1, ijk[1], ijk[2], 0)
    center += imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - 1, ijk[2], 0)

    if gradientPonderation != 0:
      g1 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
      g2 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
      g3 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
      g4 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)
      g5 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
      g6 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
      g7 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
      g8 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)

      total = center / float(5) - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / float(8))
    else:
      total = center / float(5)

    return math.log(total / float(30) + 0.1)


  #------------------------------------------------------------------------------
  #
  #
  # # FIND THE OPTIMAL TIP IN A CIRCLE (center = mouse click)
  #
  #
  #------------------------------------------------------------------------------

  def findTip(self, A, imageData, radiusNeedle, coeff, sigmaValue, gradientPonderation, X, Y, Z):
    """
    Find tip
    """
    # productive #onClick
    profprint()
    radiusNeedle = int(radiusNeedle)
    A = [int(A[0]), int(A[1]), int(A[2])]
    print(A)
    minTotalTip = 0
    X = int(X)
    Y = int(Y)
    Z = int(Z)
    for I in range(-X, X):
      i = int(I / float(coeff))
      for J in range(-Y, Y):
        j = int(J / float(coeff))
        for k in range(1):
          v0 = 0
          totalTip = 0
          for l in range (-3, 1):
            v0 = 8 * imageData.GetScalarComponentAsDouble(A[0] + i, A[1] + j, A[2] + k + l, 0)

            v1 = imageData.GetScalarComponentAsDouble(A[0] + radiusNeedle + i, A[1] + j, A[2] + k + l, 0)
            v2 = imageData.GetScalarComponentAsDouble(A[0] - radiusNeedle + i, A[1] + j, A[2] + k + l, 0)
            v3 = imageData.GetScalarComponentAsDouble(A[0] + i, A[1] + radiusNeedle + j, A[2] + k + l, 0)
            v4 = imageData.GetScalarComponentAsDouble(A[0] + i, A[1] - radiusNeedle + j, A[2] + k + l, 0)

            totalTip += v0 - ((v1 + v2 + v3 + v4) / float(4)) * gradientPonderation

          rgauss = (i ** 2
                            + j ** 2
                            + k ** 2) ** 0.5

          gaussianAttenuation = math.exp(-(rgauss / float(10)) ** 2 / float((2 * (sigmaValue / float(10)) ** 2)))
          totalTip = gaussianAttenuation * totalTip

          # totalTip = v0
          if totalTip < minTotalTip or minTotalTip == 0:
            minTotalTip = totalTip
            IBest = A[0] + i
            JBest = A[1] + j
            KBest = A[2] + k
            P = [ 0 for n in range(5)]
            P[0] = [A[0] + i, A[1] + j, A[2] + k + l]
            P[1] = [A[0] + radiusNeedle + i, A[1] + j, A[2] + k + l]
            P[2] = [A[0] - radiusNeedle + i, A[1] + j, A[2] + k + l]
            P[3] = [A[0] + i, A[1] + radiusNeedle + j, A[2] + k + l]
            P[4] = [A[0] + i, A[1] - radiusNeedle + j, A[2] + k + l]

    """
    # Draw template
    for pi in range(5):
      fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      fiducial.Initialize(slicer.mrmlScene)
      fiducial.SetName(str(pi))
      fiducial.SetFiducialCoordinates(self.ijk2ras(P[pi]))
    """
    # print "bestTip:",IBest,JBest,KBest, minTotalTip
    # print "initialtip:", A
    AInit = A

    A = [IBest, JBest, KBest]

    return A

  #------------------------------------------------------------------------------
  #
  #
  # # TRACKING NEEDLE IN -Z DIRECTION
  #
  #
  #------------------------------------------------------------------------------

  def findAxialSegmentationLimitFromMarker(self, bForceFallback=False):
    """
    Find the limit marker in the scene and return its z-coord.
    """
    # productive
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    asl = [0, 0, -float("inf")]
    coord = [0, 0, 0]
    nodes = slicer.util.getNodes('template slice position*')
    found = False
    node=None
    for node in list(nodes.values()):
      node.GetFiducialCoordinates(coord)
      aslNew = coord
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
      bases, names=self.returnBasesFromNeedleModels()
      bases2=numpy.array(bases)[:,2]
      i = bases2.argmin() #argmax highest, argmin lowest
      asl=bases[i]
      #print "ijk bases: ",repr(bases)
      #print "ijk asl: ",asl
      asl=self.ijk2ras(asl)
      asl = [coord[0], coord[1], asl[2]]
      if not node:      
        self.fiducialNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
        self.fiducialNode.Initialize(slicer.mrmlScene)
        self.fiducialNode.SetName('template slice position')
        self.fiducialNode.SetFiducialCoordinates(asl)
        fd = self.fiducialNode.GetDisplayNode()
        fd.SetVisibility(1)
        fd.SetColor([0, 1, 0])
      else:
        node.SetFiducialCoordinates(asl)
    if asl[2] == -float("inf"):
      asl = [0, 0, 0]
    return int(round(self.ras2ijk(asl)[2])), asl[2]

  def interp3(self,V, x, y, z,interpswitch):
    '''
    Ruibin: comment on this interpolation...
    '''
    if interpswitch ==1:
        xf = int(math.floor(x))
        yf = int(math.floor(y))
        zf = int(math.floor(z))
        xc = int(math.ceil(x))
        yc = int(math.ceil(y))
        zc = int(math.ceil(z))
        xd = x - xf
        yd = y - yf
        zd = z - zf
        zdd= 1-zd
        i1 = V.GetScalarComponentAsDouble(xf,yf,zf,0)*zdd + V.GetScalarComponentAsDouble(xf,yf,zc,0)*zd
        i2 = V.GetScalarComponentAsDouble(xf,yc,zf,0)*zdd + V.GetScalarComponentAsDouble(xf,yc,zc,0)*zd
        j1 = V.GetScalarComponentAsDouble(xc,yf,zf,0)*zdd + V.GetScalarComponentAsDouble(xc,yf,zc,0)*zd
        j2 = V.GetScalarComponentAsDouble(xc,yc,zf,0)*zdd + V.GetScalarComponentAsDouble(xc,yc,zc,0)*zd
        w1 = i1*(1-yd) + i2*yd
        w2 = j1*(1-yd) + j2*yd
        value = w1*(1-xd) + w2*xd
    else:
        value = V.GetScalarComponentAsDouble(int(round(x)),int(round(y)),int(round(z)),0)
    return value
      
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
                        center = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
    
                        radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
                        radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))
    
                        g1 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
                        g2 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
                        g3 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
                        g4 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)
                        g5 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                        g6 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
                        g7 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                        g8 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
        
                        value = 8 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8) * gradientPonderation
                    else:
                        value = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
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
    Is this a threaded function? No it is not yet unfortunately, but the idea was to do it!

    :param A: RAS coordinates of the needle tip
    :param imageData: volumeNode.GetImageData()
    :param colorVar: color of the needle
    :param spacing: volumneNode.GetSpacing()
    :return: a needle in 3D!!
    """
    # research
    profprint()
    global conesColor
    if conesColor: conesColor=(conesColor+1)%308;
    if conesColor==0: conesColor=300
    # ## initialisation of the parameters
    ijk = [0, 0, 0]
    bestPoint = [0, 0, 0]
    mode = "circle"

    # ## load parameters from GUI

    widget = slicer.modules.NeedleFinderWidget

    distanceMax = widget.radiusMax.value
    gradientPonderation = widget.gradientPonderation.value
    sigmaValue = widget.sigmaValue.value
    # stepsize                    = widget.stepsize.value
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

    # ## length needle = distance Aijk[2]*0.9
    # lenghtNeedle = abs(self.ijk2ras(A)[2]*0.9)
    A = [int(A[0]), int(A[1]), int(A[2])]
    if axialSegmentationLimit != None:
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    elif axialSegmentationLimit == None and widget.maxLength.isChecked():
      axialSegmentationLimit = 0
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    else:
      lenghtNeedle = lenghtNeedleParameter

    rMax = distanceMax / float(spacing[0])
    NbStepsNeedle = numberOfPointsPerNeedle - 1
    nbRotatingStep = nbRotatingIterations

    dims = [0, 0, 0]
    imageData.GetDimensions(dims)
    pixelValue = numpy.zeros(shape=(dims[0], dims[1], dims[2]))

    A0 = A
    # print A0

    self.controlPoints = []
    controlPointsIJK = []
    bestControlPoints = []

    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))

    #---------------------------------------------------------------------------------
    # look for the best tip in the neighboorhood of the mouse click
    coeff = 1  # to look every 0.2mm
    radiusSphere = 5
    X = coeff * radiusSphere / float(spacing[0])
    Y = coeff * radiusSphere / float(spacing[1])
    Z = 5 / float(spacing[2])
    # print "X,Y,Z",X,Y,Z
    #---------------------------------------------------------------------------------
    # find the tip in a circle
    if autoCorrectTip:
      A = self.findTip(A, imageData, radiusNeedle, coeff, sigmaValue, gradientPonderation, X, Y, Z)
    # keep the first control point as A0
    A0 = A
    #---------------------------------------------------------------------------------
    # Add points to the control point list

    self.controlPoints.append(self.ijk2ras(A))
    controlPointsIJK.append(A)
    bestControlPoints.append(self.ijk2ras(A))
    #---------------------------------------------------------------------------------
    # Draw fiducial points initial tip and found tip
    if drawFiducialPoints:
      oFiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      oFiducial.Initialize(slicer.mrmlScene)
      oFiducial.SetName('.c0_'+str(colorVar))
      oFiducial.SetFiducialCoordinates(self.ijk2ras(A))
      oFiducial.GetDisplayNode().SetColor(0,0,1)
      """
      fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      fiducial.Initialize(slicer.mrmlScene)
      fiducial.SetName('Best tip')
      fiducial.SetFiducialCoordinates(self.ijk2ras(A))

      fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      fiducial.Initialize(slicer.mrmlScene)
      fiducial.SetName('A init')
      fiducial.SetFiducialCoordinates(self.ijk2ras(AInit))
      """

    for step in range(0, NbStepsNeedle + 1):

      # step 0
      #------------------------------------------------------------------------------
      if step == 0:

        stepSize = self.stepSize(step, NbStepsNeedle) * lenghtNeedle

        Vx = 0
        Vy = 0
        Vz = -stepSize

        rMax = distanceMax / float(spacing[0])
        rIter = int(round(rMax))
        tIter = max(1, int(round(stepSize)))  # ## ??? stepSize can be smaller 1 and it is in mm not int index coordinates

        # print "stepsize 0:",stepSize

        tot = stepSize

      else:

        """
          step 1,2,...
          ------------------------------------------------------------------------------
                 [   vector V   ]
                 *--------------*---------------------X   -> direction of tracking
                tip0            A                    C0

                then, for the following step:
                                              tip0<-A, A<-C, C0 = A + K.V
        """
        """
        stepSize = self.stepSize(step+1,NbStepsNeedle+1)*lenghtNeedle
        #print '\nstepsize',step, ':',stepSize


        C0      = [ 2*A[0]-tip0[0],
                    2*A[1]-tip0[1],
                    A[2]-stepSize   ]
        """


        stepSize = self.stepSize(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        rMax = max(stepSize, distanceMax / float(spacing[0])) # ??? why is stepSize not divided by spacing[0]
        rIter = max(15, min(20, int(rMax / float(spacing[0])))) # ??? why divide rMax again by spacing[0]
        tIter = max(1, int(round(stepSize)))  # ## ??? stepSize can be smaller 1 and it is in mm not int index coordinates

        # Vector V
        Vx = A[0] - tip0[0]
        Vy = A[1] - tip0[1]
        Vz = A[2] - tip0[2]

      coeffSize = abs(stepSize) / spacing[2]
      if Vz != 0 :
        K = coeffSize / float(abs(Vz))
      else:
        break

      P0 = A
      C0 = [ P0[0] + K * Vx,
                    P0[1] + K * Vy,
                    P0[2] + K * Vz ]

      if drawFiducialPoints and 1: # show cone base markers b
        oFiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
        oFiducial.Initialize(slicer.mrmlScene)
        oFiducial.SetName('.b'+str(step+1)+'_'+str(colorVar))
        oFiducial.SetFiducialCoordinates(self.ijk2ras(C0))
        oFiducial.GetDisplayNode().SetColor(0,0,1)

      estimator = 0
      minEstimator = 0

      # radius variation
      for R in range(rIter + 1):

        r = R * (rMax / float(rIter))

        # ## angle variation from 0 to 360
        for thetaStep in range(nbRotatingStep):

          angleInDegree = (thetaStep * 360) / float(nbRotatingStep)
          theta = math.radians(angleInDegree)

          C = [ C0[0] + r * (math.cos(theta)),
                            C0[1] + r * (math.sin(theta)),
                            C0[2]]

          total = 0
          M = [[0, 0, 0] for i in range(int(tIter) + 1)]


          # calculates tIter = number of points per segment
          # print tIter
          for t in range(tIter + 1):

            tt = t / float(tIter)

            # x,y,z coordinates
            for i in range(3):

              M[t][i] = (1 - tt) * A[i] + tt * C[i]
              ijk[i] = int(round(M[t][i]))

            # first, test if points are in the image space
            if ijk[0] < dims[0] and ijk[0] > 0 and  ijk[1] < dims[1] and ijk[1] > 0 and ijk[2] < dims[2] and ijk[2] > 0:

              center = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
              center += imageData.GetScalarComponentAsDouble(ijk[0] + 1, ijk[1], ijk[2], 0)
              center += imageData.GetScalarComponentAsDouble(ijk[0] - 1, ijk[1], ijk[2], 0)
              center += imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + 1, ijk[2], 0)
              center += imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - 1, ijk[2], 0)
              center += imageData.GetScalarComponentAsDouble(ijk[0] + 1, ijk[1] + 1, ijk[2], 0)
              center += imageData.GetScalarComponentAsDouble(ijk[0] + 1, ijk[1] - 1, ijk[2], 0)
              center += imageData.GetScalarComponentAsDouble(ijk[0] - 1, ijk[1] + 1, ijk[2], 0)
              center += imageData.GetScalarComponentAsDouble(ijk[0] - 1, ijk[1] - 1, ijk[2], 0)

              total += center ** exponent
              if gradient == 1 and mode == "circle" :

                g1 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
                g2 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
                g3 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
                g4 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)
                g5 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                g6 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
                g7 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                g8 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)

                # total += (center - ((g1+g2+g3+g4+g5+g6+g7+g8)/float(8))*gradientPonderation)/float(tIter)
                total += ((center - g1) ** gradientPonderation) / float(tIter)
                total += ((center - g2) ** gradientPonderation) / float(tIter)
                total += ((center - g3) ** gradientPonderation) / float(tIter)
                total += ((center - g4) ** gradientPonderation) / float(tIter)
                total += ((center - g5) ** gradientPonderation) / float(tIter)
                total += ((center - g6) ** gradientPonderation) / float(tIter)
                total += ((center - g7) ** gradientPonderation) / float(tIter)
                total += ((center - g8) ** gradientPonderation) / float(tIter)

                # total = self.objectiveFunctionLOG(imageData, ijk, radiusNeedleParameter, spacing, 1)

              if gradient == 1 and mode == "square" :

                g1 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
                g2 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
                g3 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
                g4 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)

                total += (8 * center - ((g1 + g2 + g3 + g4) / float(4)) * gradientPonderation) / float(tIter)

                # total = self.objectiveFunctionLOG(imageData, ijk, radiusNeedleParameter, spacing, 1)
              # >>>>>>>>>>>>>>>>>>>>>> exp.02
              if imgLabelData:
                fLabel = imgLabelData.GetScalarComponentAsFloat(ijk[0], ijk[1], ijk[2], 0)
                if fLabel and fLabel < 300:
                  #print "# force high influence of found label: ",fLabel
                  total -= 10000
                if conesColor: imgLabelData.SetScalarComponentFromFloat(ijk[0], ijk[1], ijk[2], 0, conesColor) #mark the search cones in fLabel volume
              # <<<<<<<<<<<<<<<<<<<<
          if R == 0:

            initialIntensity = total
            estimator = total

          if gaussianAttenuationChecked == 1 and step >= 2 :
            if step == NbStepsNeedle:
                sigmaValue = 100

            if Vz != 0:
              """
              stepSize    =(A[2] - C0[2])
              K           =stepSize/float(tip0[2]-A[2])

              X           = [ A[0] + K * (A[0]-tip0[0]),
                              A[1] + K * (A[1]-tip0[1]),
                              A[2] + K * (A[2]-tip0[2]) ]
              """

              rgauss = ((C[0] - C0[0]) ** 2
                              + (C[1] - C0[1]) ** 2
                              + (C[2] - C0[2]) ** 2) ** 0.5

              gaussianAttenuation = math.exp(-(rgauss / float(rMax)) ** 2 / float((2 * (sigmaValue / float(10)) ** 2)))  # 1 for x=0, 0.2 for x=5
              estimator = (total) * gaussianAttenuation # ??? this doesn't work as intended for negative values (they get larger!)
            else:
              estimator = total


          else:
            estimator = (total)

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

      # save the value of the objective function for the initial step.
      # will be used as reference for the initial step of the ascending tracking
      if step == 1:
        self.estimatorReference = minEstimator


      if A[2] < axialSegmentationLimit and A != A0 and (tip0[2] - A[2]) != 0:

        asl = axialSegmentationLimit
        l = (A[2] - asl) / float(tip0[2] - A[2])

        A = [  A[0] - l * (tip0[0] - A[0]),
                A[1] - l * (tip0[1] - A[1]),
                A[2] - l * (tip0[2] - A[2])]

      self.controlPoints.append(self.ijk2ras(A))
      controlPointsIJK.append(A)
      # print('step:',step,':',minEstimator )
      if drawFiducialPoints:
        fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
        fiducial.Initialize(slicer.mrmlScene)
        fiducial.SetName('.c' +str(step)+'_'+ str(colorVar))
        fiducial.SetFiducialCoordinates(self.controlPoints[step + 1])

      if A[2] <= axialSegmentationLimit and A != A0:
        break

    if not autoStopTip:
      self.addNeedleToScene(self.controlPoints, colorVar, 'Detection', script,0, manualName=manualName)

  def needleDetectionThread13_1(self, Aori, imageData, colorVar, spacing, script=False, imgLabelData=None,manName="",bDrawNeedle=False,whetherfeedback=False,ControlPointsPackage=None,ControlPointsPackageIJK=None,WrongPosition=None,GlobalDirection=None, Punish=None):
    '''MICCAI2013 suspect version, 3/11/13
    iGyne_old b16872c19a3bc6be1f4a9722e5daf16a603393f6
    https://github.com/gpernelle/iGyne_old/commit/b16872c19a3bc6be1f4a9722e5daf16a603393f6#diff-8ab0fe8b431d2af8b1aff51977e85ca2

    From the needle tip, the algorithm looks for a direction maximizing the "needle likelihood" of a small segment in a conic region.
    The second extremity of this segment is saved as a control point (in controlPoints), used later.
    Then, this step is iterated, replacing the needle tip by the latest control point.
    The height of the new conic region (stepsize) is increased as well as its base diameter (rMax) and its normal is collinear to the previous computed segment. (cf. C0)
    NbStepsNeedle iterations give NbStepsNeedle-1 control points, the last one being used as an extremity as well as the needle tip.
    From these NbStepsNeedle-1 control points and 2 extremities a Bezier curve is computed, approximating the needle path.
    '''
    # productive #probablyMiccai13
    profprint()
    global conesColor
    if conesColor: conesColor=(conesColor+1)%308;
    if conesColor==0: conesColor=300
    # ## initialisation of the parameters
    ijk = [0, 0, 0]
    bestPoint = [0, 0, 0]
    widget = slicer.modules.NeedleFinderWidget

    # ## load parameters from GUI
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

    # >>>>>> Ruibins feedback >>>>>>>>>>
    if(whetherfeedback):
        WrongIndex = WrongPosition[0]
        NumberOfRelatedIndexs = len(WrongPosition) - 1
        RelatedIndex = []    
        for k in range(NumberOfRelatedIndexs):
            RelatedIndex.append(WrongPosition[k+1])
        A = ControlPointsPackageIJK[WrongIndex][0]
        NeighbourAttenuation = 0
    else:
        A = Aori
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    # ## length needle = distance Aijk[2]*0.9
    # lenghtNeedle = abs(self.ijk2ras(A)[2]*0.9)

    if axialSegmentationLimit != None:
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    else:
      lenghtNeedle = A[2] * 0.9 * spacing[2]

    rMax = distanceMax / float(spacing[0])
    NbStepsNeedle = numberOfPointsPerNeedle - 1
    nbRotatingStep = nbRotatingIterations

    dims = [0, 0, 0]
    imageData.GetDimensions(dims)
    pixelValue = numpy.zeros(shape=(dims[0], dims[1], dims[2]))

    A0 = A
    print(A0)
    if widget.drawFiducialPoints.isChecked():
      oFiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      oFiducial.Initialize(slicer.mrmlScene)
      oFiducial.SetName('.c0_'+str(colorVar))
      oFiducial.SetFiducialCoordinates(self.ijk2ras(A))
      oFiducial.GetDisplayNode().SetColor(0,0,1)

    controlPoints = []
    controlPointsIJK = []
    bestControlPoints = []

    controlPoints.append(self.ijk2ras(A))
    controlPointsIJK.append(A)
    bestControlPoints.append(self.ijk2ras(A))

    for step in range(0, NbStepsNeedle + 2):
      print("length", lenghtNeedle)
      # step 0
      #------------------------------------------------------------------------------
      if step == 0:

        L = self.stepSize13(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        C0 = [A[0], A[1], A[2] - L]
        rMax = distanceMax / float(spacing[0])
        rIter = rMax
        tIter = int(round(L))  # ??? L can be smaller 1 and it is in mm not int index coordinates

      # step 1,2,...
      #------------------------------------------------------------------------------
      else:

        stepSize = self.stepSize13(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        print(stepSize)

        C0 = [ 2 * A[0] - tip0[0],
                    2 * A[1] - tip0[1],
                    A[2] - stepSize   ]  # ??? this is buggy vector calculus, now its a feature ;-)

        rMax = max(stepSize, distanceMax / float(spacing[0]))
        rIter = max(15, min(20, int(rMax / float(spacing[0])))) # ??? why divide again by spacing[0]
        tIter = stepSize  # ## ??? stepSize can be smaller 1 and it is in mm not int index coordinates

      if not script and widget.drawFiducialPoints.isChecked(): # show cone base markers b
        oFiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
        oFiducial.Initialize(slicer.mrmlScene)
        oFiducial.SetName('.b'+str(step+1)+'_'+str(colorVar))
        oFiducial.SetFiducialCoordinates(self.ijk2ras(C0))
        oFiducial.GetDisplayNode().SetColor(0,0,1)

      estimator = 0
      minEstimator = 0

      # radius variation
      for R in range(int(rIter) + 1):

        r = R * (rMax / float(rIter))

        # ## angle variation from 0 to 360
        for thetaStep in range(int(nbRotatingStep)):

          angleInDegree = (thetaStep * 360) / float(nbRotatingStep)
          theta = math.radians(angleInDegree)

          C = [ C0[0] + r * (math.cos(theta)),
                            C0[1] + r * (math.sin(theta)),
                            C0[2]]

          total = 0
          M = [[0, 0, 0] for i in range(int(tIter) + 1)]


          # calculates tIter = number of points per segment
          for t in range(int(tIter) + 1):

            tt = t / float(tIter)

            # x,y,z coordinates
            for i in range(3):

              M[t][i] = (1 - tt) * A[i] + tt * C[i]
              ijk[i] = int(round(M[t][i]))

            # first, test if points are in the image space
            if ijk[0] < dims[0] and ijk[0] > 0 and  ijk[1] < dims[1] and ijk[1] > 0 and ijk[2] < dims[2] and ijk[2] > 0:

              center = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
              total += center
              
              #>>>>>>>>>>>>>>> Ruibins feedback >>>>>>>>>>>>>>>>>
              if(whetherfeedback):
                  NeighbourAttenuation = self.Feedback(ControlPointsPackage, Punish, radiusNeedleParameter, NumberOfRelatedIndexs, len, RelatedIndex, range, M[t]) 
                  total +=NeighbourAttenuation
              #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              
              if gradient == 1 :

                radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
                radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))

                g1 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
                g2 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
                g3 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
                g4 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)
                g5 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                g6 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
                g7 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                g8 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)

                total += 8 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8) * gradientPonderation
              # >>>>>>>>>>>>>>>>>>>>>> exp.02
              if imgLabelData:
                fLabel = imgLabelData.GetScalarComponentAsFloat(ijk[0], ijk[1], ijk[2], 0)
                if fLabel and fLabel < 300:
                  #print "# force high influence of found label: ",fLabel
                  total -= 10000
                if conesColor: imgLabelData.SetScalarComponentFromFloat(ijk[0], ijk[1], ijk[2], 0, conesColor) #mark the search cones in fLabel volume
              # <<<<<<<<<<<<<<<<<<<<
          if R == 0:

            initialIntensity = total
            estimator = total

          if gaussianAttenuationChecked == 1 and step >= 2 :

            if tip0[2] - A[2] != 0:

                stepSize = (A[2] - C0[2])
                K = stepSize / float(tip0[2] - A[2])

                X = [ A[0] + K * (A[0] - tip0[0]),
                                A[1] + K * (A[1] - tip0[1]),
                                A[2] + K * (A[2] - tip0[2]) ]

                rgauss = ((C[0] - X[0]) ** 2
                                + (C[1] - X[1]) ** 2
                                + (C[2] - X[2]) ** 2) ** 0.5

                gaussianAttenuation = math.exp(-(rgauss / float(rMax)) ** 2 / float((2 * (sigmaValue / float(10)) ** 2)))  # 1 for x=0, 0.2 for x=5
                estimator = (total) * gaussianAttenuation # ??? this doesn't work as intended for negative values (they get larger!)
            else:
                estimator = total


          else:
            estimator = (total)

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

        A = [  A[0] - l * (tip0[0] - A[0]),
                A[1] - l * (tip0[1] - A[1]),
                A[2] - l * (tip0[2] - A[2])]

      controlPoints.append(self.ijk2ras(A))
      controlPointsIJK.append(A)

      if not script and widget.drawFiducialPoints.isChecked():
        fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
        fiducial.Initialize(slicer.mrmlScene)
        fiducial.SetName('.c_'+str(colorVar))
        fiducial.SetFiducialCoordinates(controlPoints[step + 1])

      if A[2] <= axialSegmentationLimit and A != A0:
        break

    # self.addNeedleToScene(controlPoints,colorVar)
    if not autoStopTip:
      if bDrawNeedle: self.addNeedleToScene(controlPoints, colorVar, 'Detection', script,0,manualName=manName)
    
    return controlPoints,controlPointsIJK # <<< Ruibin

  def needleDetectionThread13_2(self, A, imageData, colorVar, spacing, script=False, imgLabelData=None,manualName=""):
    '''MICCAI2013 suspect version, 3/11/13
    iGyne_old b16872c19a3bc6be1f4a9722e5daf16a603393f6
    https://github.com/gpernelle/iGyne_old/commit/b16872c19a3bc6be1f4a9722e5daf16a603393f6#diff-8ab0fe8b431d2af8b1aff51977e85ca2

    >>> same as 13_1, with bugfixes (e.g. concerning down stepSize or L "/spacing[2]") <<<

    From the needle tip, the algorithm looks for a direction maximizing the "needle likelihood" of a small segment in a conic region.
    The second extremity of this segment is saved as a control point (in controlPoints), used later.
    Then, this step is iterated, replacing the needle tip by the latest control point.
    The height of the new conic region (stepsize) is increased as well as its base diameter (rMax) and its normal is collinear to the previous computed segment. (cf. C0)
    NbStepsNeedle iterations give NbStepsNeedle-1 control points, the last one being used as an extremity as well as the needle tip.
    From these NbStepsNeedle-1 control points and 2 extremities a Bezier curve is computed, approximating the needle path.
    '''
    #productive #probablyMiccai13 #smallChanges
    profprint()
    t0 = time.perf_counter()
    global conesColor
    if conesColor: conesColor=(conesColor+1)%308;
    if conesColor==0: conesColor=300
    # ## initialisation of the parameters
    ijk = [0, 0, 0]
    bestPoint = [0, 0, 0]
    widget = slicer.modules.NeedleFinderWidget

    # ## load parameters from GUI
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

    # ## length needle = distance Aijk[2]*0.9
    # lenghtNeedle = abs(self.ijk2ras(A)[2]*0.9)

    if axialSegmentationLimit != None:
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    else:
      lenghtNeedle = A[2] * 0.9 * spacing[2]

    rMax = distanceMax / float(spacing[0])
    NbStepsNeedle = numberOfPointsPerNeedle - 1
    nbRotatingStep = nbRotatingIterations

    dims = [0, 0, 0]
    imageData.GetDimensions(dims)
    pixelValue = numpy.zeros(shape=(dims[0], dims[1], dims[2]))

    A0 = A
    print(A0)
    if widget.drawFiducialPoints.isChecked():
      oFiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      oFiducial.Initialize(slicer.mrmlScene)
      oFiducial.SetName('.c0_'+str(colorVar))
      oFiducial.SetFiducialCoordinates(self.ijk2ras(A))
      oFiducial.GetDisplayNode().SetColor(0,0,1)

    controlPoints = []
    controlPointsIJK = []
    bestControlPoints = []

    controlPoints.append(self.ijk2ras(A))
    controlPointsIJK.append(A)
    bestControlPoints.append(self.ijk2ras(A))

    for step in range(0, NbStepsNeedle + 2):
      print("length", lenghtNeedle)
      # step 0
      #------------------------------------------------------------------------------
      if step == 0:

        L = self.stepSize13(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        #L = self.stepSizeAndre(step + 1, NbStepsNeedle + 2) * lenghtNeedle #<<< not better
        #L=lenghtNeedle/NbStepsNeedle #<o><< equal step size: better!
        C0 = [A[0], A[1], A[2] - L /spacing[2]] #<<< /spacing[2] bugfix
        rMax = distanceMax / float(spacing[0])
        rIter = rMax
        tIter = int(round(L))  ## /spacing[2] worse at beginnng! more steps??? L can be smaller 1 and it is in mm not int index coordinates

      # step 1,2,...
      #------------------------------------------------------------------------------
      else:

        stepSize = self.stepSize13(step + 1, NbStepsNeedle + 1) * lenghtNeedle
        #stepSize = self.stepSizeAndre(step + 1, NbStepsNeedle + 2) * lenghtNeedle #<<< not better
        #stepSize=lenghtNeedle/NbStepsNeedle #<o><< equal step size: better!
        print(stepSize)

        C0 = [ 2 * A[0] - tip0[0],
                    2 * A[1] - tip0[1],
                    A[2] - stepSize /spacing[2]   ]  #<<< /spacing[2] bugfix # ??? this is buggy vector calculus, now its a feature ;-)
        #C0 = [A[0], A[1], A[2]+int(round(stepSize/spacing[2])) ] #<<< TODO test simply going down along z-axis
        rMax = max(stepSize, distanceMax / float(spacing[0]))
        rIter = max(15, min(20, int(rMax / float(spacing[0])))) # ??? why divide again by spacing[0]
        tIter = stepSize     # /spacing[2]  #more steps, slightly better! ??? stepSize can be smaller 1 and it is in mm not int index coordinates

      if not script and widget.drawFiducialPoints.isChecked(): # show cone base markers b
        oFiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
        oFiducial.Initialize(slicer.mrmlScene)
        oFiducial.SetName('.b'+str(step+1)+'_'+str(colorVar))
        oFiducial.SetFiducialCoordinates(self.ijk2ras(C0))
        oFiducial.GetDisplayNode().SetColor(0,0,1)

      estimator = 0
      minEstimator = 0

      # radius variation
      for R in range(int(rIter) + 1):

        r = R * (rMax / float(rIter))

        # ## angle variation from 0 to 360
        for thetaStep in range(int(nbRotatingStep)):

          angleInDegree = (thetaStep * 360) / float(nbRotatingStep)
          theta = math.radians(angleInDegree)

          C = [ C0[0] + r * (math.cos(theta)),
                            C0[1] + r * (math.sin(theta)),
                            C0[2]]

          total = 0
          M = [[0, 0, 0] for i in range(int(tIter) + 1)]


          # calculates tIter = number of points per segment
          for t in range(int(tIter) + 1):

            tt = t / float(tIter)

            # x,y,z coordinates
            for i in range(3):

              M[t][i] = (1 - tt) * A[i] + tt * C[i]
              ijk[i] = int(round(M[t][i]))

            # first, test if points are in the image space
            if ijk[0] < dims[0] and ijk[0] > 0 and  ijk[1] < dims[1] and ijk[1] > 0 and ijk[2] < dims[2] and ijk[2] > 0:

              center = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
              total += center
              if gradient == 1 :

                radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
                radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))

                g1 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
                g2 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
                g3 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
                g4 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)
                g5 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                g6 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
                g7 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                g8 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)

                total += 8 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8) * gradientPonderation
              # >>>>>>>>>>>>>>>>>>>>>> exp.02
              if imgLabelData:
                fLabel = imgLabelData.GetScalarComponentAsFloat(ijk[0], ijk[1], ijk[2], 0)
                if fLabel and fLabel < 300:
                  #print "# force high influence of found label: ",fLabel
                  total -= 10000
                if conesColor: imgLabelData.SetScalarComponentFromFloat(ijk[0], ijk[1], ijk[2], 0, conesColor) #mark the search cones in fLabel volume
              # <<<<<<<<<<<<<<<<<<<<
          if R == 0:

            initialIntensity = total
            estimator = total

          if gaussianAttenuationChecked == 1 and step >= 2 :

            if tip0[2] - A[2] != 0:

                stepSize = (A[2] - C0[2])
                K = stepSize / float(tip0[2] - A[2])

                X = [ A[0] + K * (A[0] - tip0[0]),
                                A[1] + K * (A[1] - tip0[1]),
                                A[2] + K * (A[2] - tip0[2]) ]

                rgauss = ((C[0] - X[0]) ** 2
                                + (C[1] - X[1]) ** 2
                                + (C[2] - X[2]) ** 2) ** 0.5

                gaussianAttenuation = math.exp(-(rgauss / float(rMax)) ** 2 / float((2 * (sigmaValue / float(10)) ** 2)))  # 1 for x=0, 0.2 for x=5
                estimator = (total) * gaussianAttenuation # ??? this might not work as intended for negative values (they get larger!)
                # >>> bugfix ? makes it even worse on average: bugfixbug ;-)
                #estimator=total
                #estimatorAtt = total * gaussianAttenuation
                #estimator-=np.abs(estimator-estimatorAtt)
                # <<< xifgub
            else:
                estimator = total


          else:
            estimator = (total)

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

        A = [  A[0] - l * (tip0[0] - A[0]),
                A[1] - l * (tip0[1] - A[1]),
                A[2] - l * (tip0[2] - A[2])]

      controlPoints.append(self.ijk2ras(A))
      controlPointsIJK.append(A)

      if not script and widget.drawFiducialPoints.isChecked():
        fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
        fiducial.Initialize(slicer.mrmlScene)
        fiducial.SetName('.c_'+str(colorVar))
        fiducial.SetFiducialCoordinates(controlPoints[step + 1])

      if A[2] <= axialSegmentationLimit and A != A0:
        break

    # self.addNeedleToScene(controlPoints,colorVar)
    if not autoStopTip:
      self.addNeedleToScene(controlPoints, colorVar, 'Detection', script, manualName=manualName)
    print(time.perf_counter() - t0, "seconds process time")

  def addPlaneToScene(self, foot, x, y):
    """
    Add a plane to the scene.
    :param foot: base point
    :param x: vector
    :param y: vector
    :return: visible plane in scene
    """
    #research
    profprint()
    scene = slicer.mrmlScene
    # Create model node
    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetName(scene.GenerateUniqueName(".ObturatorPlane"))

    planeSource = vtk.vtkPlaneSource()
    foot-=25*(x+y)
    #planeSource.SetOrigin(np.array(foot))
    planeSource.SetOrigin(list(foot))
    planeSource.SetPoint1(np.array(foot)+50*x)
    planeSource.SetPoint2(np.array(foot)+50*y)
    planeSource.Update()
    model.SetAndObservePolyData(planeSource.GetOutput())

    # Create display node
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetColor(1,1,0) # yellow
    modelDisplay.SetBackfaceCulling(0)
    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())

    # Add to scene
    scene.AddNode(model)
    # transform = slicer.vtkMRMLLinearTransformNode()
    # scene.AddNode(transform)
    # model.SetAndObserveTransformNodeID(transform.GetID())
    #
    # vTransform = vtk.vtkTransform()
    # vTransform.Scale(50,50,50)
    # #vTransform.RotateX(30)
    # transform.SetAndObserveMatrixTransformToParent(vTransform.GetMatrix())

  def addFiducialToScene(self, ijk, name, rgbColor=(1,1,0), glyphScale=2, textScale=2):
    """
    Add fiducial marker to scene.
    @param ijk: IJK point marker to draw 
    @param name: string for the marker
    """
    #research
    #profprint()
    print() #.addFiducial"
    fid = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
    fid.Initialize(slicer.mrmlScene)
    ras = self.ijk2ras(ijk)
    fid.SetFiducialCoordinates(ras)
    fid.SetName(name)
    fid.GetDisplayNode().SetColor(rgbColor) #(r,g,b)
    fid.GetDisplayNode().SetGlyphScale(glyphScale)
    fid.GetAnnotationTextDisplayNode().SetTextScale(textScale)

  def needleDetectionThread15_1(self, ijkAori, imgData, imgLabelData, lrasTempPoints, iColorVar, fvSpacing, bUp=False, bScript=False, strManualName="", tipOnly=False,bDrawNeedle=False,whetherfeedback=False,ControlPointsPackage=None,ControlPointsPackageIJK=None,WrongPosition=None,GlobalDirection=None, Punish=None):
    r'''MICCAI2015 version, 4/16/15
    based on iGyne_old b16872c19a3bc6be1f4a9722e5daf16a603393f6
    https://github.com/gpernelle/iGyne_old/commit/b16872c19a3bc6be1f4a9722e5daf16a603393f6#diff-8ab0fe8b431d2af8b1aff51977e85ca2

    >>> Developers version: Andre's bug fixes & experiments here: e.g. use additional user information to fix outliers.
    >>> \!/ This site is under heavy construction. /!\
    From the needle tip, the algorithm looks for a direction maximizing the "needle likelihood" of a small segment in a conic region.
    The second extremity of this segment is saved as a control point (in lvControlPointsRAS), used later.
    Then, this iStep is iterated, replacing the needle tip by the latest control point.
    The height of the new conic region (stepsize) is increased as well as its base diameter (iRMax) and its normal is collinear to the previous computed segment. (cf. ijkB)
    nStepsNeedle iterations give nStepsNeedle-1 control points, the last one being used as an extremity as well as the needle tip.
    From these nStepsNeedle-1 control points and 2 extremities a Bezier curve is computed, approximating the needle path.
    '''
    #research
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
    
    #>>>>>>>>>>>>>>>>> Ruibins feedback >>>>>>>>>>>>>>>>>>
    if(whetherfeedback):
        WrongIndex = WrongPosition[0]
        NumberOfRelatedIndexs = len(WrongPosition) - 1
        RelatedIndex = []    
        for k in range(NumberOfRelatedIndexs):
            RelatedIndex.append(WrongPosition[k+1])
        ijkA = ControlPointsPackageIJK[WrongIndex][0]
        NeighbourAttenuation = 0
    else:
        ijkA = ijkAori
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # refactored local functions:
    def norm(fv):
        return np.sqrt(np.dot(fv,fv))
    def normalized(fv):
        fLen=np.sqrt(np.dot(fv,fv))
        return fv/fLen, fLen
    def defLocalNeedleCooSys(fvX, fvY, fvZ, iStep, rasA, rasDeflectionDirection, rasNeedlePlaneNormal,bDraw=False):
        rasSegmentVector=rasA-rasAPrev
        fLenSV=norm(rasSegmentVector)
        rasSegmentVector1=rasSegmentVector/fLenSV
        fAngleZvSeg_rad=np.arccos(np.dot(fvZ,rasSegmentVector1))
        if not rasNeedlePlaneNormal.any(): rasNeedlePlaneNormal = np.cross(rasSegmentVector1,fvZ)
        if np.arccos(np.dot(rasNeedlePlaneNormal,fvX))>np.pi/2: rasNeedlePlaneNormal*=-1 # check vector direction and turn if not in rhs normal X direction
        if not sum(rasNeedlePlaneNormal):
          rasNeedlePlaneNormal=np.cross(fvY,fvZ)
        fLenNPN = norm(rasNeedlePlaneNormal)
        rasNeedlePlaneNormal1=rasNeedlePlaneNormal/fLenNPN
        if not rasDeflectionDirection.any(): rasDeflectionDirection=  \
          +   np.cross(fvZ,rasNeedlePlaneNormal1) #TODO / CAVE: mind the sign
        if np.arccos(np.dot(rasDeflectionDirection,fvY))>np.pi/2: rasDeflectionDirection*=-1 # check vector direction and turn for RHS
        fLenDD=norm(rasDeflectionDirection)
        rasDeflectionDirection1=rasDeflectionDirection/fLenDD
        #re-estimate reference needle direction  #TODO/CAVE: mind the sign
        if 1:
          fvZ=    +     np.cross(rasNeedlePlaneNormal1,rasDeflectionDirection1)
          fvZ/=norm(fvZ)
        return rasDeflectionDirection1, rasNeedlePlaneNormal1, fvZ
    def initNeedleModel(fAngleRefNeedle_rad, fEstNeedleLength_mm, fvZ, rasA, bDraw=False):
        rasAPrev2A=rasA-rasAPrev; rasAPrev2A/=norm(rasAPrev2A)
        fAngleLongSegVZ_rad=np.arccos(np.dot(fvZ,rasAPrev2A))
        fEstDeflection_mm=fEstNeedleLength_mm*1/np.cos(fAngleRefNeedle_rad)*np.sin(fAngleLongSegVZ_rad) # estimate deflection as from z-axis
        # look up the needle length model matrices and find best matches
        faArcLen_mm=list(zip(*matArcLen_mm))[0] #transpose
        ixArcLenColMin=find_nearest(faArcLen_mm,fEstNeedleLength_mm)[0]
        # estimator using est. deflection
        faYDefl_mm=matYDefl_mm[ixArcLenColMin] #zip(*matYDefl_mm)[ixArcLenColMin] #transpose
        ixYDeflRowMin=find_nearest(faYDefl_mm,fEstDeflection_mm)[0]
        fEstF0_mmN=matFs_mN[ixArcLenColMin][ixYDeflRowMin]/1000
        fEstEndSegAngle_rad=matEndSegAngles_rad[ixArcLenColMin][ixYDeflRowMin]
        fSumAngle_rad=fEstEndSegAngle_rad
        fF_mmN=fEstF0_mmN #*np.cos(fSumAngle_rad)/(np.pi) #estimate
        return fF_mmN, fSumAngle_rad
    def runNeedleModel(fStepAngle_rad, fF_mmN, fStepSize_mm, fvZ, iModelIt, rasDeflectionDirection1, fSumAngle_rad, rasB, bDraw=False):
        rasModelStepVector=rasDeflectionDirection1*np.sin(fSumAngle_rad)+fvZ*np.cos(fSumAngle_rad)
        rasEstStepVector1=rasModelStepVector/norm(rasModelStepVector)
        rasBDef=rasB
        rasB=rasA  +   rasEstStepVector1*fStepSize_mm # TODO check sign, and if this still makes sense
        fDistBDef2BMod=norm(rasBDef-rasB)
        fMaxDistBDef2BMod_mm=0 #.25 # CONST
        if 1 and fDistBDef2BMod>fMaxDistBDef2BMod_mm:
          #compromise between model and heuristic based default vector
          rasBDef2BMod=rasB-rasBDef; rasBDef2BMod1, fDistBDef2BMod_mm=normalized(rasBDef2BMod)
          rasB=rasBDef+rasBDef2BMod1*min(fMaxDistBDef2BMod_mm,fDistBDef2BMod_mm/2.)
        fAngleMultiSegments_rad=0
        if fStepSize_mm/fModelSegmentLength_mm < 1: iFineStep=5; print("# fine steps ")
        else: iFineStep=1; print("normal steps")
        for j in range(0, int(round(fStepSize_mm/(fModelSegmentLength_mm/iFineStep)))):
          fF_mmN=fF_mmN/np.cos(fSumAngle_rad)
          fStepAngle_rad=fF_mmN/(fK*iFineStep); angle_deg=np.degrees(fStepAngle_rad)
          fSumAngle_rad-=fStepAngle_rad; fSumAngle_deg=np.degrees(fSumAngle_rad)
          if fSumAngle_rad<0: fStepAngle_rad=fSumAngle_rad=0;# TODO check this
          fAngleMultiSegments_rad+=fStepAngle_rad
          iModelIt+=1
        fAngleMultiSegments_deg=np.degrees(fAngleMultiSegments_rad)
        return fF_mmN, fAngleMultiSegments_deg, rasEstStepVector1, fSumAngle_deg, rasB
    def find_nearest(t,x):
      from bisect import bisect_left
      i = bisect_left(t, x)
      if i==len(t): i-=1
      if t[i] - x > 0.5:
        i-=1
      return i,t[i]
    ijkAPrev=rasCBestPrev=ijkCBest=ijkCBestPrev=None
    rasA2CBestInit=[]
    rasA2CBest=rasCBestPrev2CBest=rasCBestPrev2CBestPrev1=np.array([])
    ijk = [0, 0, 0]
    fLabel = None
    if not bScript: widget.createAddOrSelectLabelMapNode(bScript) #<<< always have label map to draw search cones
    #widget.clearLabelMap()

    if iAxialSegmentationLimit != None:
      fEstNeedleLength_mm = abs(ijkA[2] - iAxialSegmentationLimit) * 1.15 * fvSpacing[2] #<<< CONST was 1.15
    else:
      fEstNeedleLength_mm = ijkA[2] * 0.9 * fvSpacing[2] # CONST
    #load external needle model matrices (variable = file name)
    # Mechanical deflection lookup matrices imported from DeflectionModelData
    fAngleRefNeedle_rad=np.deg2rad(22.5)# <<< to rotate z-axis down 22.5 degrees around x
    fvX=np.array([1,0,0]); fvY=-np.array([0,np.cos(fAngleRefNeedle_rad),np.sin(fAngleRefNeedle_rad)])
    fvZ=np.array([0,np.sin(fAngleRefNeedle_rad),  -   np.cos(fAngleRefNeedle_rad)])
    fSumAngle_rad=fStepAngle_rad=fAngleA2CBestVZ_rad=None
    rasA=np.array(self.ijk2ras(ijkA))#<<<
    rasA0=rasA
    ijkA0 = ijkA
    nStepsNeedle = nPointsPerNeedle - 1 # one point (mouse click on tip) is already there
    if bUp:
      iZDirectionSign = 1
    else:
      iZDirectionSign = -1

    ivDims = [0, 0, 0]
    imgData.GetDimensions(ivDims)

    if not bUp:
      self.controlPoints = [] # in up mode, more interpol. points will be added to this array
    lvControlPointsRAS = []
    lvControlPointsIJK = []
    lvBestControlPoints = []
    rasNeedlePlaneNormal=rasDeflectionDirection=np.zeros(3)

    lvControlPointsRAS.append(self.ijk2ras(ijkA))
    lvControlPointsIJK.append(ijkA)
    lvBestControlPoints.append(self.ijk2ras(ijkA))
    print("#-----------------------------------------------------------")
    fStepsSum_mm=0
    iModelIt=0
    iStart=-1 # <<<o> use init its. or not
    for iAbsStep, iStep in enumerate(range(iStart, nStepsNeedle)):
      print("#--------------------------------- %s: iStart: %d / iStep: %d / b,c%d----------------"%(strManualName, iStart,iStep,iStep+1))
      fStepSize_mm = self.stepSize13(iStep+1,nStepsNeedle+1)*fEstNeedleLength_mm
      #fStepSize_mm = (1.-self.stepSizeAndre(iStep+1,nStepsNeedle+1))*fEstNeedleLength_mm
      fStepSize_mm = fEstNeedleLength_mm / (nStepsNeedle) # <<< equal step size
      if iStep>-1: fStepsSum_mm+=fStepSize_mm
      print("fStepSize_mm, fStepsSum_mm, fEstNeedleLength_mm= %f, %f, %f"%(fStepSize_mm, fStepsSum_mm, fEstNeedleLength_mm))

      # init. iSteps for model est.
      #------------------------------------------------------------------------------
      if iStep < 0:

        fFac=1./((((2)))*abs(iStart))*(iAbsStep+1) #<o> CONST how much of the nedle used for init. (1=full, 2=half)
        fStepSizeInit_mm = fFac * fEstNeedleLength_mm
        rasEstStepVector1=np.array([fvZ[0],fvZ[1], fvZ[2]])
        rasB= rasA+rasEstStepVector1*fStepSizeInit_mm #<<< first step along reference needle axis
        ijkB=self.ras2ijk(rasB)
        iRMax = max(fStepSizeInit_mm, iRadiusMax_mm / float(fvSpacing[0])) # / nStepsNeedle
        nRIter = max(15, min(20, int(iRMax/fvSpacing[0])))
        nTIter=fStepSizeInit_mm #???
        print("iRMax, nRIter, nTIter= %d, %d, %d"%(iRMax,nRIter,nTIter))

      # iStep 0
      #------------------------------------------------------------------------------
      elif iStep==0: # and iStart == 0:

        ijkB = [ijkA[0], ijkA[1], ijkA[2]+iZDirectionSign*fStepSize_mm/fvSpacing[2] ] #<<< going down along z-axis performs better on average on MICCAI13 cases
        rasB=self.ijk2ras(ijkB)
        ijkEstStepVector=np.array([0,0,iZDirectionSign * fStepSize_mm/fvSpacing[2]])

        if len(rasA2CBestInit)==0: rasA2CBestInit.append(fvZ)
        rasInitStepVector=sum(rasA2CBestInit)/len(rasA2CBestInit)
        rasInitStepVector1, dum=normalized(rasInitStepVector) #TODO try
        print("rasInitStepVector1= ",rasInitStepVector1)
        rasB=rasA+rasInitStepVector1*fStepSize_mm #<o>
        ijkB=self.ras2ijk(rasB)
        rasEstStepVector=self.ijk2ras(ijkEstStepVector)
        rasEstStepVector1, dum=normalized(rasEstStepVector)
        print("ijkB= ",ijkB,"# line: ",lineno())
        iRMax = iRadiusMax_mm / fvSpacing[0] # / nStepsNeedle
        nRIter = iRMax
        nTIter=int(round(fStepSize_mm))
        print("iRMax, nRIter, nTIter= %d, %d, %d"%(iRMax,nRIter,nTIter))
        if iStart<0 and iStep==0:
          fF_mmN, fSumAngle_rad = initNeedleModel(fAngleRefNeedle_rad, fEstNeedleLength_mm, fvZ, rasA)

      # iStep other
      #------------------------------------------------------------------------------
      else:

        ijkB = [ 2 * ijkA[0] - ijkAPrev[0],  # ??? why do you go double iStep in xy-plane
                  2 * ijkA[1] - ijkAPrev[1],
                  ijkA[2] + iZDirectionSign * fStepSize_mm /fvSpacing[2]   ]  # ??? this was buggy vector calculus, now its a feature ;-)

        rasB=self.ijk2ras(ijkB)
        print("default_ijkB= ", ijkB)
        iRMax = max(fStepSize_mm,iRadiusMax_mm / fvSpacing[0])
        nRIter =max(15,min(20,int(iRMax/ fvSpacing[0])))
        nTIter = fStepSize_mm # ??? coarse step size from MICCAI13 TODO check this
        print("iRMax, nRIter, nTIter= %d, %d, %d"%(iRMax,nRIter,nTIter))

        #>>>>>> exp.04
        # define local needle coordinate system
        rasDeflectionDirection1, rasNeedlePlaneNormal1, fvZ = defLocalNeedleCooSys(fvX, fvY, fvZ, iStep, rasA, rasDeflectionDirection, rasNeedlePlaneNormal,bDraw=True)
        # initialise needle model
        if (iStart==0 and iStep==1):
          fF_mmN, fSumAngle_rad = initNeedleModel(fAngleRefNeedle_rad, fEstNeedleLength_mm, fvZ, rasA, bDraw=True)
        # run needle model
        if (iStart==0 and iStep>=1) or (iStart<0 and iStep>=0):
          fF_mmN, fAngleMultiSegments_deg, rasEstStepVector1, fSumAngle_deg,    rasB   = runNeedleModel(fStepAngle_rad, fF_mmN, fStepSize_mm, fvZ, iModelIt, rasDeflectionDirection1, fSumAngle_rad, rasB, bDraw=True)
        ijkB=self.ras2ijk(rasB)
        #<<<<<< exp.04

      ##############
      # cone search: radius, fStepAngle_rad and line length variation
      if ijkCBest: ijkCBestPrev = ijkCBest
      ijkCBest = [np.nan, np.nan, np.nan]
      print("# searching cone:")
      fMinTotal = np.inf
      for iR in range(int(nRIter) + 1):

        fR = iR * (iRMax / float(nRIter))

        # ## fStepAngle_rad variation from 0 to 360
        for iThetaStep in range(int(nRotatingIts)):

          fAngle_deg = (iThetaStep * 360.) / float(nRotatingIts)
          fThetaStep_rad = math.radians(fAngle_deg)

          ijkC = [ ijkB[0] + fR * (math.cos(fThetaStep_rad)),
                            ijkB[1] + fR * (math.sin(fThetaStep_rad)),
                            ijkB[2]]

          fTotal = 0
          lijkM = [[0, 0, 0] for i in range(int(nTIter) + 1)]

          fLabel =  0
          # calculates nTIter = number of points per segment
          for iTStep in range(int(nTIter) + 1):

            fTStep = iTStep / float(nTIter)

            # x,y,z coordinates
            for i in range(3):

              lijkM[iTStep][i] = (1 - fTStep) * ijkA[i] + fTStep * ijkC[i]
              ijk[i] = int(round(lijkM[iTStep][i]))

            # first, test if points are in the image space
            if ijk[0] < ivDims[0] and ijk[0] > 0 and  ijk[1] < ivDims[1] and ijk[1] > 0 and ijk[2] < ivDims[2] and ijk[2] > 0:

              dCenter = imgData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
              fTotal += dCenter
              
              # >>>>>>>>>>>>>>> Ruibins feedback >>>>>>>>>>>>>>
              if(whetherfeedback):
                  NeighbourAttenuation = self.Feedback(ControlPointsPackage, Punish, iRadiusNeedle_mm, NumberOfRelatedIndexs, len, RelatedIndex, range, lijkM[iTStep])
                  fTotal +=NeighbourAttenuation
              #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              
              if 1 and bGradient == 1 : #<<< feature on/off

                iRadiusNeedle = int(round(iRadiusNeedle_mm / float(fvSpacing[0])))
                iRadiusNeedleCorner = int(round((iRadiusNeedle_mm / float(fvSpacing[0]) / 1.414)))

                g1 = imgData.GetScalarComponentAsDouble(ijk[0] + iRadiusNeedle, ijk[1], ijk[2], 0)
                g2 = imgData.GetScalarComponentAsDouble(ijk[0] - iRadiusNeedle, ijk[1], ijk[2], 0)
                g3 = imgData.GetScalarComponentAsDouble(ijk[0], ijk[1] + iRadiusNeedle, ijk[2], 0)
                g4 = imgData.GetScalarComponentAsDouble(ijk[0], ijk[1] - iRadiusNeedle, ijk[2], 0)
                g5 = imgData.GetScalarComponentAsDouble(ijk[0] + iRadiusNeedleCorner, ijk[1] + iRadiusNeedleCorner, ijk[2], 0)
                g6 = imgData.GetScalarComponentAsDouble(ijk[0] - iRadiusNeedleCorner, ijk[1] - iRadiusNeedleCorner, ijk[2], 0)
                g7 = imgData.GetScalarComponentAsDouble(ijk[0] - iRadiusNeedleCorner, ijk[1] + iRadiusNeedleCorner, ijk[2], 0)
                g8 = imgData.GetScalarComponentAsDouble(ijk[0] + iRadiusNeedleCorner, ijk[1] - iRadiusNeedleCorner, ijk[2], 0)

                fTotal += 8 * dCenter - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8) * iGradientPonderation
              #fi gradient
              # >>>>>>>>>>>>>>>>>>>>>> exp.02
              if imgLabelData:
                fLabel = imgLabelData.GetScalarComponentAsFloat(ijk[0], ijk[1], ijk[2], 0)
                if fLabel and fLabel < 300:
                  fTotal -= 10000
                if conesColor: imgLabelData.SetScalarComponentFromFloat(ijk[0], ijk[1], ijk[2], 0, conesColor) #mark the search cones in fLabel volume
              # <<<<<<<<<<<<<<<<<<<<
            #fi points in image space
          #rof iTStep

          if 1 and bGaussianAttenuation == 1 and iStep >= 2 : #<<< feature on/off

            if ijkAPrev[2] - ijkA[2] != 0:

                fK = (ijkA[2] - ijkB[2]) / float(ijkAPrev[2] - ijkA[2])

                ijkX = [ ijkA[0] + fK * (ijkA[0] - ijkAPrev[0]),
                                ijkA[1] + fK * (ijkA[1] - ijkAPrev[1]),
                                ijkA[2] + fK * (ijkA[2] - ijkAPrev[2]) ]

                fRgauss = ((ijkC[0] - ijkX[0]) ** 2
                                + (ijkC[1] - ijkX[1]) ** 2
                                + (ijkC[2] - ijkX[2]) ** 2) ** 0.5

                fGaussianAttenuation = math.exp(-(fRgauss / float(iRMax)) ** 2 / float((2 * (iSigmaValue / float(10)) ** 2)))  # 1 for x=0, 0.2 for x=5
                if fGaussianAttenuation<0 or fGaussianAttenuation>1:
                  print("fGaussianAttenuation= ",fGaussianAttenuation," # /!\\ ")
                fTotal = fTotal * fGaussianAttenuation
          #fi gauss

          if fTotal < fMinTotal:
            fMinTotal = fTotal
            ijkCBest = ijkC; rasCBest=self.ijk2ras(ijkCBest)
        #rof iThetaStep
      #rof iR
      if ijkCBest==[np.nan,np.nan,np.nan]:
        breakbox("/!!\\ no ijkCBest found")

      ijkAPrev = ijkA; rasAPrev=np.array(self.ijk2ras(ijkAPrev))

      #>>> constraints: compare distances
      if 1: # <o>
        #fAngleA2CBestVStepVector_rad=0
        print("ijkCBest= ", ijkCBest,'# line: ', lineno())
        rasB2CBest=rasCBest-np.array(rasB)
        fDistB2CBest_mm=norm(rasB2CBest)
        print("fDistB2CBest_mm= ",fDistB2CBest_mm)
        if ijkCBestPrev:
          rasCBestPrev=np.array(self.ijk2ras(ijkCBestPrev))
          if rasCBestPrev2CBest.any(): rasCBestPrev2CBestPrev=rasCBestPrev2CBest; rasCBestPrev2CBestPrev1=rasCBestPrev2CBestPrev/norm(rasCBestPrev2CBestPrev)
          rasCBestPrev2CBest=rasCBest-rasCBestPrev; rasCBestPrev2CBest1=rasCBestPrev2CBest/norm(rasCBestPrev2CBest)
        rasA=np.array(self.ijk2ras(ijkA))
        if rasA2CBest.any(): rasA2CBestPrev=rasA2CBest; rasA2CBestPrev1=rasA2CBestPrev/norm(rasA2CBestPrev)
        rasA2CBest=rasCBest-rasA; rasA2CBest1=rasA2CBest/norm(rasA2CBest)
        # in first initializing iterations, model not estimated yet, accept ijkCBest anyway
        if (iStart<0 and iStep<0) or (iStart==0 and iStep==0):
          fDistB2CBest_mm=-np.inf
          fLimitCrit4=fMaxDistB2CBest_mm=np.inf #CONST
        if ((iStart==0 and iStep>=1) or (iStart<0 and iStep>=0)) and ijkCBest != [np.nan, np.nan, np.nan]: # <<< feature off/on
          fMaxDistB2CBest_mm=(((1))) #<o> CONST mm
          fLimitCrit4=fMaxDistB2CBest_mm #CONST
        bReject=not fDistB2CBest_mm<fLimitCrit4
        bUseB=False
        bRejected=False
        ########################
        if bReject:print("#^^^^^^^^^^^^^ crit. rejects best cone point")
        ########################
        if ijkCBest == [np.nan, np.nan, np.nan] or bReject: #<<<o> use criteria, when we dont use const. behaves as 13_2
          if 1: #<o> TODO try, compromise between model and image feature based vector
            if rasB2CBest.any():
              rasB2CBest1, dum=normalized(rasB2CBest)
            else: rasB2CBest1=np.array([0,0,0])
            rasA=rasB+rasB2CBest1*min(fMaxDistB2CBest_mm,fDistB2CBest_mm/2.)
            ijkA=self.ras2ijk(rasA)
            print('#middle vector, but within constraint: rasA= ',rasA)
            #<<<
          bUseB=True
          print("using_point_near_ijkB= ", ijkA)
          if iStep==0-abs(iStart): breakbox("/!!\\ not using CBest in first it.??") #<<<
        elif ijkCBest != ijkAPrev:
          if iStep<=-1: #prepare init. of model
            print("# iStep -1")
            if iStep==-1:
              rasA2CBestInit.append(rasA2CBest) #2nd init. seg.
              rasAPrev=rasA0-rasA2CBest #trick to restart from user tip click
              ijkAPrev=self.ras2ijk(rasAPrev)
              ijkA=ijkA0; rasA=self.ijk2ras(ijkA)
            elif iStep<=-2: rasA2CBestInit.append(rasA2CBest) #1st init segment
            continue #rof iStep, skip rest of loop using this helper segments
          ijkA = ijkCBest; rasA=self.ijk2ras(ijkA)
          rasNeedlePlaneNormal=rasDeflectionDirection=np.zeros(3) #<o>force recalc. of deflection direction
          print("using_ijkC= ",ijkCBest)
          print("best_fEstimator_value= ",fTotal)
        else: breakbox("/!!| uncaught ijkCBest==ijkAPrev")
        #file ijkCBest != ijkAPrev
      #fi constraints <<<
      
      #################
      # drag back a too far low control point to the axial limit plane
      if ijkA[2] < iAxialSegmentationLimit and ijkA != ijkA0 and (ijkAPrev[2] - ijkA[2]) != 0:

        asl = iAxialSegmentationLimit
        l = (ijkA[2] - asl) / float(ijkAPrev[2] - ijkA[2])

        ijkA = [  ijkA[0] - l * (ijkAPrev[0] - ijkA[0]),
                ijkA[1] - l * (ijkAPrev[1] - ijkA[1]),
                ijkA[2] - l * (ijkAPrev[2] - ijkA[2])]

      if iStep>=0:
        lvControlPointsRAS.append(self.ijk2ras(ijkA))
        lvControlPointsIJK.append(ijkA)

        if ijkA[2] <= iAxialSegmentationLimit and ijkA != ijkA0:
          print("# /!\\ needle too long, break")
          break
    #rof iStep
    for i in range(len(lvControlPointsRAS)): self.controlPoints.append(lvControlPointsRAS[i])
    if not bAutoStopTip:
      if bDrawNeedle: self.addNeedleToScene(lvControlPointsRAS, (205)% MAXCOL, 'Detection', bScript,0, manualName=strManualName)
      self.controlPoints = []
    if bAutoStopTip and bUp:
      if bDrawNeedle: self.addNeedleToScene(self.controlPoints, (205)% MAXCOL, 'Detection', bScript,0, manualName=strManualName)
      self.controlPoints = []
    print(time.perf_counter() - t0, "seconds process time")
    #self.deleteTempModels() # otw. too much clutter on view
    if bUp: return lvControlPointsIJK[-1] #return last control point as tip estimate
    
    return lvControlPointsRAS,lvControlPointsIJK # <<< Ruibin

  def Feedback(self, ControlPointsPackage, Punish, iRadiusNeedle_mm, NumberOfRelatedIndexs, len, RelatedIndex, range, ijk):
      ''' Ruibin: comment... '''
      ras = [0,0,0]
      ras = self.ijk2ras(ijk)
      NeighbourAttenuation = 0
      for w in range(NumberOfRelatedIndexs):
          distance = 9999
          NumberOfPointsOnThatNeedle = len(ControlPointsPackage[RelatedIndex[w]])
          if ras[2] >= ControlPointsPackage[RelatedIndex[w]][0][2]:
              distance = ((ras[2] - ControlPointsPackage[RelatedIndex[w]][0][2]) ** 2 + (ras[2] - ControlPointsPackage[RelatedIndex[w]][0][2]) ** 2 + (ras[2] - ControlPointsPackage[RelatedIndex[w]][0][2]) ** 2) ** 0.5
          elif ras[2] <= ControlPointsPackage[RelatedIndex[w]][NumberOfPointsOnThatNeedle - 1][2]:
              distance = ((ras[2] - ControlPointsPackage[RelatedIndex[w]][NumberOfPointsOnThatNeedle - 1][2]) ** 2 + (ras[2] - ControlPointsPackage[RelatedIndex[w]][NumberOfPointsOnThatNeedle - 1][2]) ** 2 + (ras[2] - ControlPointsPackage[RelatedIndex[w]][NumberOfPointsOnThatNeedle - 1][2]) ** 2) ** 0.5
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
          
          # using attenuation like Gaussian
          if distance == 0:
              NeighbourAttenuation += 3000
          elif distance >= iRadiusNeedle_mm * Punish:
              NeighbourAttenuation += 0
          else:
              NeighbourAttenuation += 1000 * (iRadiusNeedle_mm / distance)
      
      return NeighbourAttenuation

  #------------------------------------------------------------------------------
  #
  #
  # # TRACKING NEEDLE IN +Z DIRECTION
  #
  #
  #------------------------------------------------------------------------------
  def needleDetectionUPThread(self, A, imageData, colorVar, spacing, script=False, strName="", tipOnly=False):
    """
    Call different versions of needle detection up thread.
    """
    # research
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.axialSegmentationLimit, widget.axialSegmentationLimitRAS = self.findAxialSegmentationLimitFromMarker()
    # select algo version
    if widget.algoVersParameter.value == 0:
      return self.needleDetectionUPThreadCurrentDev(A, imageData, colorVar, spacing, script=script, tipOnly=tipOnly, manualName=strName)
    if widget.algoVersParameter.value == 1:
      msgbox ("/!\\ needleDetectionUPThread N/A for algoVersion 1")
    if widget.algoVersParameter.value == 2:
      msgbox ("/!\\ needleDetectionUPThread N/A for algoVersion 2")
    if widget.algoVersParameter.value == 3:
      if widget.labelMapNode:
        return self.needleDetectionThread15_1(A, imageData, widget.labelMapNode.GetImageData(), widget.tempPointList, colorVar, spacing, bUp=True, bScript=script, strManualName=strName)
      else:
        return self.needleDetectionThread15_1(A, imageData, None, widget.tempPointList, colorVar, spacing, bUp=True, bScript=script, strManualName=strName)
    if widget.algoVersParameter.value == 4:
      if widget.labelMapNode:
        return self.needleDetectionThread15_1(A, imageData, widget.labelMapNode.GetImageData(), widget.tempPointList, colorVar, spacing, bUp=True, bScript=script, strName=strName)
      else:
        return self.needleDetectionThread15_1(A, imageData, None, widget.tempPointList, colorVar, spacing, bUp=True, bScript=script, strName=strName)
    else: 
      msgbox ("/!\\ needleDetectionUPThread not defined")

  def needleDetectionUPThreadCurrentDev(self, A, imageData, colorVar, spacing, script=False, tipOnly=False, manualName=""):
    """
    From the needle tip, the algorithm looks for a direction maximizing the "needle likelihood" of a small segment in a conic region.
    The second extremity of this segment is saved as a control point (in controlPoints), used later.
    Then, this step is iterated, replacing the needle tip by the latest control point.
    The height of the new conic region (stepsize) is increased as well as its base diameter (rMax) and its normal is collinear to the previous computed segment. (cf. C0)
    NbStepsNeedle iterations give NbStepsNeedle-1 control points, the last one being used as an extremity as well as the needle tip.
    From these NbStepsNeedle-1 control points and 2 extremities a Bezier curve is computed, approximating the needle path.
    """
    # research
    profprint()
    widget = slicer.modules.NeedleFinderWidget

    # ## initialisation of the parameters
    ijk = [0, 0, 0]
    bestPoint = [0, 0, 0]

    # ## load parameters from GUI
    distanceMax = widget.radiusMax.value
    gradientPonderation = widget.gradientPonderation.value
    sigmaValue = widget.sigmaValue.value
    stepsize = widget.stepsize.value
    gaussianAttenuationChecked = widget.gaussianAttenuationButton.isChecked()
    lookNeighborhood = widget.gradient.isChecked()
    numberOfPointsPerNeedle = max(1, widget.numberOfPointsPerNeedle.value - 1)
    nbRotatingIterations = widget.nbRotatingIterations.value
    radiusNeedleParameter = widget.radiusNeedleParameter.value
    # radiusNeedleParameter       = 1
    axialSegmentationLimit = widget.axialSegmentationLimit
    lenghtNeedleParameter = widget.lenghtNeedleParameter.value / (spacing[2])

    # ## length needle = distance Aijk[2]*0.9
    # lenghtNeedle = abs(self.ijk2ras(A)[2]*0.9)

    if axialSegmentationLimit != None:
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    elif axialSegmentationLimit == None and widget.maxLength.isChecked():
      axialSegmentationLimit = 0
      lenghtNeedle = abs(A[2] - axialSegmentationLimit) * 1.15 * spacing[2]
    else:
      lenghtNeedle = lenghtNeedleParameter

    rMax = distanceMax / float(spacing[0])
    NbStepsNeedle = numberOfPointsPerNeedle - 1
    nbRotatingStep = nbRotatingIterations

    dims = [0, 0, 0]
    imageData.GetDimensions(dims)
    pixelValue = numpy.zeros(shape=(dims[0], dims[1], dims[2]))

    A0 = A
    # print A0

    controlPointsUP = []
    controlPointsIJK = []
    bestControlPoints = []
    minEstimator0 = None
    stopTracking = 0

    controlPointsUP.append(self.ijk2ras(A))
    controlPointsIJK.append(A)
    bestControlPoints.append(self.ijk2ras(A))

    radiusNeedle = int(round(radiusNeedleParameter / float(spacing[0])))
    radiusNeedleCorner = int(round((radiusNeedleParameter / float(spacing[0]) / 1.414)))
    """
    #---------------------------------------------------------------------------------
    # look for the best tip in the neighboorhood of the mouse click
    X=10/float(spacing[0])
    Y=10/float(spacing[1])
    Z=10/float(spacing[2])
    #print "X,Y,Z",X,Y,Z
    minTotalTip=0

    for i in range(-3,3):
      for j in range(-3,3):
          for k in range(-3,3):
            for l in range (-3,1):
                v0 = 8 * imageData.GetScalarComponentAsDouble(ijk[0]+i, ijk[1]+j, ijk[2]+k+l, 0)
                v1 = imageData.GetScalarComponentAsDouble(ijk[0]+radiusNeedle+i, ijk[1]+j, ijk[2]+k+l, 0)
                v2 = imageData.GetScalarComponentAsDouble(ijk[0]-radiusNeedle+i, ijk[1]+j, ijk[2]+k+l, 0)
                v3 = imageData.GetScalarComponentAsDouble(ijk[0]+i, ijk[1]+radiusNeedle+j, ijk[2]+k+l, 0)
                v4 = imageData.GetScalarComponentAsDouble(ijk[0]+i, ijk[1]-radiusNeedle+j, ijk[2]+k+l, 0)
                v5 = imageData.GetScalarComponentAsDouble(ijk[0]+radiusNeedleCorner+i, ijk[1]+radiusNeedleCorner+j, ijk[2]+k+l, 0)
                v6 = imageData.GetScalarComponentAsDouble(ijk[0]-radiusNeedleCorner+i, ijk[1]-radiusNeedleCorner+j, ijk[2]+k+l, 0)
                v7 = imageData.GetScalarComponentAsDouble(ijk[0]-radiusNeedleCorner+i, ijk[1]+radiusNeedleCorner+j, ijk[2]+k+l, 0)
                v8 = imageData.GetScalarComponentAsDouble(ijk[0]+radiusNeedleCorner+i, ijk[1]-radiusNeedleCorner+j, ijk[2]+k+l, 0)
            totalTip = v0-(v1+v2+v3+v4+v5+v6+v7+v8/float(8))*gradientPonderation
            if totalTip<minTotalTip or minTotalTip==0:
              minTotalTip = totalTip
              IBest = ijk[0]+i
              JBest = ijk[1]+j
              KBest = ijk[2]+k


    #print "bestTip:",IBest,JBest,KBest, minTotalTip
    #print "initialtip:", A
    #---------------------------------------------------------------------------------
    """

    for step in range(0, NbStepsNeedle + 1):

      # step 0
      #------------------------------------------------------------------------------
      if step == 0:

        stepSize = self.stepSize(step, NbStepsNeedle) * lenghtNeedle

        Vx = 0
        Vy = 0
        Vz = stepSize

        C0 = [A[0], A[1], A[2] + stepSize]
        rMax = stepSize / float(spacing[0])
        rIter = int(round(rMax))
        tIter = max(1, int(round(stepSize)))  # ## ??? stepSize can be smaller 1 and it is in mm not int index coordinates

        # print "stepsize 0:",stepSize

        # tot     = stepSize

      # step 1,2,...
      #------------------------------------------------------------------------------
      else:

        stepSize = self.stepSize(step + 1, NbStepsNeedle + 1) * lenghtNeedle

        Vx = A[0] - tip0[0]
        Vy = A[1] - tip0[1]
        Vz = A[2] - tip0[2]

      coeffSize = abs(stepSize) / spacing[2]
      if Vz != 0 :
        K = coeffSize / float(abs(Vz))
      else:
        break

      P0 = A
      P1 = [ P0[0] + K * Vx,
                    P0[1] + K * Vy,
                    P0[2] + K * Vz ]

      # value of the objective function. We want to minimize it.
      dichoStep = 0
      continueDichotomy = 1

      # In order to find the tip, if the last point is beyond the tip (tested function of
      # the relative value of the objective function), we start a dichotomy
      # the dichotomy runs until a) point below the tip  or b) more than 7 loops
      testSuccess = None
      while continueDichotomy == 1 and dichoStep <= 7 and stopTracking == 0 :
        # reset values
        estimator = 0
        minEstimator = 0
        dichoStep += 1
        if step != 100:

          if dichoStep > 1:

            K /= float(2)
            stepSize /= float(2)

            if testSuccess == 1:
              P0 = P1
            else:
              P0 = P0

            P1 = [ P0[0] + K * Vx,
                      P0[1] + K * Vy,
                      P0[2] + K * Vz ]
          # print "\t----------------------------------------"
          # print '\nstepsize',step, ':', stepSize
          # tot     += stepSize

          """
          C0      = [ 2*A[0]-tip0[0],
                      2*A[1]-tip0[1],
                      A[2]+stepSize   ]
          """

          """
          coeffSize   = abs(stepSize)
          K           = coeffSize/float(abs(Vz))


          C0           = [ A[0] + K * Vx,
                          A[1] + K * Vy,
                          A[2] + K * Vz ]
          """

          C0 = P1

          rMax = stepSize / float(spacing[0])
          rIter = max(15, min(20, int(rMax / float(spacing[0]))))
          tIter = max(3, int(round(stepSize)))
          # print "\trMax:",rMax, "rIter:",rIter, "tIter",tIter
          # print "\tP1:",P1[2]

        # radius variation
        for R in range(rIter + 1):

          r = R * (rMax / float(rIter))

          # ## angle variation from 0 to 360
          for thetaStep in range(nbRotatingStep):

            angleInDegree = (thetaStep * 360) / float(nbRotatingStep)
            theta = math.radians(angleInDegree)

            C = [ C0[0] + r * (math.cos(theta)),
                              C0[1] + r * (math.sin(theta)),
                              C0[2]]

            total = 0
            M = [[0, 0, 0] for i in range(int(tIter) + 1)]

            # calculates tIter = number of points per segment
            for t in range(tIter + 1):

              tt = t / float(tIter)

              # x,y,z coordinates
              for i in range(3):

                M[t][i] = (1 - tt) * A[i] + tt * C[i]
                ijk[i] = int(round(M[t][i]))

              # first, test if points are in the image space
              if ijk[0] < dims[0] and ijk[0] > 0 and  ijk[1] < dims[1] and ijk[1] > 0 and ijk[2] < dims[2] and ijk[2] > 0:

                center = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
                total += center / float(tIter)
                if lookNeighborhood == 1 :

                  g1 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedle, ijk[1], ijk[2], 0)
                  g2 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedle, ijk[1], ijk[2], 0)
                  g3 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] + radiusNeedle, ijk[2], 0)
                  g4 = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1] - radiusNeedle, ijk[2], 0)
                  g5 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                  g6 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)
                  g7 = imageData.GetScalarComponentAsDouble(ijk[0] - radiusNeedleCorner, ijk[1] + radiusNeedleCorner, ijk[2], 0)
                  g8 = imageData.GetScalarComponentAsDouble(ijk[0] + radiusNeedleCorner, ijk[1] - radiusNeedleCorner, ijk[2], 0)

                  total += (8 * center - ((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / float(8)) * gradientPonderation) / float(tIter)

            if R == 0:

              initialIntensity = total
              estimator = total

            if gaussianAttenuationChecked == 1 and step >= 1 :

              if Vz != 0:

                  """
                  coeffSize    =abs((A[2] - C0[2]))
                  K           =coeffSize/float(abs(tip0[2]-A[2]))

                  X           = [ A[0] + K * (A[0]-tip0[0]),
                                  A[1] + K * (A[1]-tip0[1]),
                                  A[2] + K * (A[2]-tip0[2]) ]
                  """

                  rgauss = ((C[0] - C0[0]) ** 2
                                  + (C[1] - C0[1]) ** 2
                                  + (C[2] - C0[2]) ** 2) ** 0.5

                  gaussianAttenuation = math.exp(-(rgauss / float(rMax)) ** 2 / float((2 * (sigmaValue / float(10)) ** 2)))  # 1 for x=0, 0.2 for x=5
                  estimator = (total) * gaussianAttenuation
              else:
                  estimator = total

            else:
              estimator = (total)

            if estimator < initialIntensity:

              if estimator < minEstimator or minEstimator == 0:
                minEstimator = estimator

                if minEstimator != 0:
                  bestPoint = C

        # print "\tInitial Intensity:",initialIntensity
        # print "\tMin Estimator:",minEstimator

        if step == 0:
          minEstimator0 = self.estimatorReference

        if minEstimator0 != None and minEstimator0 != 0:
          RelMinEstimator = abs((minEstimator - minEstimator0) / float(minEstimator0))

          # print "\tmin Estimator 0 :",minEstimator0,"\n"
          # print "\tRel Estimator :",RelMinEstimator,"\n"
          valCtrPt = self.objectiveFunction(imageData, A, 3, spacing, 1)
          # print "\tObjective function Ctl Pt: ", valCtrPt

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
      if bestPoint == [0, 0, 0]:  # if the initial point (center of the cone) is indeed the optimal ctrl point
        A = C0
      elif bestPoint != tip0:
        A = bestPoint

      controlPointsUP.append(self.ijk2ras(A))
      controlPointsIJK.append(A)

      if widget.drawFiducialPoints.isChecked():
        fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
        fiducial.Initialize(slicer.mrmlScene)
        fiducial.SetName('..' + str(step) + '.' + str(self.objectiveFunction(imageData, A, 3, spacing, 1))+'_'+str(colorVar))
        fiducial.SetFiducialCoordinates(controlPointsUP[step + 1])
        fiducial.GetDisplayNode().SetColor(1, 1, 0)

      if A[2] <= axialSegmentationLimit and A != A0:
        break

    for i in range(len(controlPointsUP)):
      self.controlPoints.append(controlPointsUP[i])
    if not tipOnly: self.addNeedleToScene(self.controlPoints, colorVar, 'Detection', script,manualName=manualName)
    # print 'length:',tot
    return controlPointsIJK[-1] #return last control point as tip estimate

  def drawObturatorNeedles(self):
    """
    Draw needles around the obturator of length realNeedleLength
    Draw straigth lines, parallel to the one defined by two control points
    given by the first two clicks
    """
    # productive
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    realNeedleLength = widget.realNeedleLength.value

    # reset report table
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


    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode')
    nbNode = modelNodes.GetNumberOfItems()
    for nthNode in range(nbNode):
      modelNode = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLAnnotationFiducialNode')
      if modelNode.GetAttribute("ObturatorNeedle") == "1":
        needleNumber = int(modelNode.GetAttribute("NeedleNumber"))
        needleStep = int(modelNode.GetAttribute("NeedleStep"))
        coord = [0, 0, 0]
        modelNode.GetFiducialCoordinates(coord)
        widget.obtuNeedleValueCtrPt[needleNumber][needleStep] = coord

    for i in range(0, len(widget.obtuNeedleValueCtrPt)):

      sign = cmp(widget.obtuNeedleValueCtrPt[0][0][2], widget.obtuNeedleValueCtrPt[0][1][2])
      if (i == 0 and sign <= -1):
        AA = widget.obtuNeedleValueCtrPt[0][0][2]
        BB = widget.obtuNeedleValueCtrPt[0][1][2]
        widget.obtuNeedleValueCtrPt[0][0][2] = BB
        widget.obtuNeedleValueCtrPt[0][1][2] = AA

      # As we give the tip of the obturator needles, we only want to g in the increasing z direction.

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
            # print needleNumber,needleStep,coord

    for i in range(len(widget.obtuNeedlePt)):
      if widget.obtuNeedlePt[i][0][0] != 999:
          controlPointsUnsorted = [val for val in widget.obtuNeedlePt[i] if val != [999, 999, 999]]
          controlPoints = controlPointsUnsorted
          if ((i == 0 and len(controlPoints) >= 1) or i >= 1) :
            self.addNeedleToScene(controlPoints, i, 'Obturator')

  def drawValidationNeedles(self, nb=None):
    """This function takes the values inside the table tableValueCtrPt and add attributes such as color, name ,.. for every
     validation needle.
     * To add the needle to the scene, the points given for every needle has to be sorted sagitally,so they can be used as
     control points in a Beziers curve
     * For each sets of needle points, the function addNeedleToScene fits a Bezier curve and render it as a
     tube (vtkMRMLModelNode)
    """
    # productive #onButton
    profprint()
    # reset report table
    # print "Draw manually segmented needles..."
    # self.table =None
    # self.row=0
    widget = slicer.modules.NeedleFinderWidget
    if nb: widget.editNeedleTxtBox.value = nb
    widget.initTableView()
    self.deleteEvaluationNeedlesFromTable()
    while slicer.util.getNodes('manual-seg_'+str(widget.editNeedleTxtBox.value)) != {}:
      nodes = slicer.util.getNodes('manual-seg_'+str(widget.editNeedleTxtBox.value))
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)

    tableValueCtrPt = [[[999, 999, 999] for i in range(100)] for j in range(100)]
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode')
    nbNode = modelNodes.GetNumberOfItems()
    for nthNode in range(nbNode):
        modelNode = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLAnnotationFiducialNode')
        modelNodeName = modelNode.GetName().strip('.')
        modelNodeName = modelNodeName.strip('\r')
        parts = modelNodeName.split('-')
        if modelNode.GetName().startswith('.') and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
          needleNumber = int(parts[0])
          needleStep = int(parts[1])
          # if modelNode.GetAttribute("ValidationNeedle") == "1":
          #   needleNumber = int(modelNode.GetAttribute("NeedleNumber"))
          if needleNumber == widget.editNeedleTxtBox.value:
            # needleStep = int(modelNode.GetAttribute("NeedleStep"))
            coord = [0, 0, 0]
            modelNode.GetFiducialCoordinates(coord)
            tableValueCtrPt[needleNumber][needleStep] = coord
            print(needleNumber, needleStep, coord)
            # print self.tableValueCtrPt[needleNumber][needleStep]

    for i in range(len(tableValueCtrPt)):
      if not all(e == [999, 999, 999] for e in tableValueCtrPt[i]):
      # if self.tableValueCtrPt[i][1] != [999, 999, 999]:
        colorVar = random.randrange(50, 100, 1)  # ??? /(100.)
        controlPointsUnsorted = [val for val in tableValueCtrPt[i] if val != [999, 999, 999]]
        controlPoints = self.sortTable(controlPointsUnsorted, (2, 1, 0))
        # print "Control points unsorted", controlPointsUnsorted
        print("Control points", controlPoints)
        self.addNeedleToScene(controlPoints, i, 'Validation')
        self.observeManualNeedles()
      else:
        # print i
        pass
    self.findAxialSegmentationLimitFromMarker(bForceFallback=True) #AM force the presence of the limit marker

  def addCSplineToScene(self, controlPoint, colorVar, needleType='Detection', endMarker=False, name="^", script=False, manualName=""):
    """Adds visual needle representation as interpolating cardinal spline to the scene.
    Alternative to Bezier curves.

    :param controlPoint: array of RAS coordinates of points of a needle (used as control point for the Bezier's curve
    :param colorVar: color of the needle
    :param needleType: 'validation' for a manually segmentated needle, 'detection' for an automatically segmented needle,
    'obturator' for an obturator needle
    :return: visually, needle added to the scene
    """
    #research
    #profprint()
    """
    Create a model of the linear needle segments
    """
    widget = slicer.modules.NeedleFinderWidget

    realNeedleLength = widget.realNeedleLength.value
    extendNeedle = widget.extendNeedle.isChecked()

    # sort the points in a decreasing order (from tip to bottom)
    controlPointListSorted = controlPoint #self.sortTableReverse(controlPoint, (2, 1, 0))

    # calculate the length of the list of ctr points
    lenghtTotal = 0
    for i in range(len(controlPoint) - 1):
        lenghtTotal += self.distanceTwoPoints(controlPointListSorted[i + 1], controlPointListSorted[i])
    #print "Polygon added to scene, length of tube: ", lenghtTotal

    # in case we want to extend the needle to the wanted length
    """
    The extension is currently done by adding a point to the control point list.
    """
    if lenghtTotal < realNeedleLength and extendNeedle:
      lastPoint = [controlPointListSorted[-1][0], controlPointListSorted[-1][1], controlPointListSorted[-1][2] - (realNeedleLength - lenghtTotal)]
      controlPointListSorted.append(lastPoint)

    scene = slicer.mrmlScene
    # One spline for each direction.
    aSplineX = vtk.vtkCardinalSpline()
    aSplineY = vtk.vtkCardinalSpline()
    aSplineZ = vtk.vtkCardinalSpline()
    numberOfInputPoints = len(controlPointListSorted)
    inputPoints = vtk.vtkPoints()
    # start read out
    for i in range(numberOfInputPoints):
      x,y,z = controlPointListSorted[i][0], controlPointListSorted[i][1], controlPointListSorted[i][2]
      aSplineX.AddPoint(i, x)
      aSplineY.AddPoint(i, y)
      aSplineZ.AddPoint(i, z)
      inputPoints.InsertPoint(i, x, y, z)
    # Generate the polyline for the spline.
    points = vtk.vtkPoints()
    profileData = vtk.vtkPolyData()
    # Number of points on the spline
    numberOfOutputPoints = 50
    # Interpolate x, y and z by using the three spline filters and
    # create new points
    for i in range(0, numberOfOutputPoints):
      t = (numberOfInputPoints-1.0)/(numberOfOutputPoints-1.0)*i
      points.InsertPoint(i, aSplineX.Evaluate(t), aSplineY.Evaluate(t), aSplineZ.Evaluate(t))
    # Create the polyline.
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(numberOfOutputPoints)
    for i in range(0, numberOfOutputPoints): lines.InsertCellPoint(i)
    profileData.SetPoints(points)
    profileData.SetLines(lines)
    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(8)
    profileTubes.SetInputData(profileData)
    profileTubes.SetRadius(1)
    profileTubes.SetNumberOfSides(50)
    profileTubes.Update()

    # ## Create model node
    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetAndObservePolyData(profileTubes.GetOutput())
    # ## Create display node
    modelDisplay = slicer.vtkMRMLModelDisplayNode()

    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    # ## Add to scene
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    scene.AddNode(model)
    model.GetDisplayNode().Visibility2DOn()
    if needleType == 'Validation':
      model.SetName('.manual-seg_' + str(colorVar))
    elif needleType == 'Obturator':
      model.SetName('.obturator-seg_' + str(colorVar))
    else:
      model.SetName('.auto-seg_' + str(self.round) + '-ID-' + str(model.GetID())+'-'+manualName)
    model.SetAttribute('type', needleType)

    if needleType == 'Validation':
      nth = int(colorVar) % MAXCOL
      modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
      model.SetAttribute("nth", str(nth))

    else:
      nth = int(model.GetID().strip('vtkMRMLModelNode')) % MAXCOL
      modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
      model.SetAttribute("nth", str(nth))

    if endMarker:
      fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      fiducial.SetName('.'+name+'_'+str(colorVar))
      fiducial.Initialize(slicer.mrmlScene)
      fiducial.SetFiducialCoordinates(controlPointListSorted[-1])
      fiducial.SetAttribute('TemporaryFiducial', '1')
      fiducial.SetLocked(True)
      displayNode = fiducial.GetDisplayNode()
      displayNode.SetGlyphScale(2.1)
      displayNode.SetColor(0, 0, 0)
      textNode = fiducial.GetAnnotationTextDisplayNode()
      textNode.SetTextScale(3)
      textNode.SetColor(0, 0, 0)

  def addPolyLineToScene(self, controlPoint, colorVar, needleType='Detection',script=False, endMarker=False, name="^", trans=0, manualName="",radius=1,bPause=False):
    """Just adds visual representation of linear (needle) segments to the scene.
    Useful for drawing the control polygon of a smooth curve.

    :param controlPoint: array of RAS coordinates of points of a needle (used as control point for the Bezier's curve
    :param colorVar: color of the needle
    :param needleType: 'validation' for a manually segmentated needle, 'detection' for an automatically segmented needle,
    'obturator' for an obturator needle
    :return: visually, needle added to the scene
    """
    #research
    #profprint()
    """
    Create a model of the linear needle segments
    """
    widget = slicer.modules.NeedleFinderWidget

    realNeedleLength = widget.realNeedleLength.value
    extendNeedle = widget.extendNeedle.isChecked()

    # sort the points in a decreasing order (from tip to bottom)
    controlPointListSorted = controlPoint #self.sortTableReverse(controlPoint, (2, 1, 0))

    # calculate the length of the list of ctr points
    lenghtTotal = 0
    for i in range(len(controlPoint) - 1):
        lenghtTotal += self.distanceTwoPoints(controlPointListSorted[i + 1], controlPointListSorted[i])
    #print "Polygon added to scene, length of tube: ", lenghtTotal

    # in case we want to extend the needle to the wanted length
    """
    The extension is currently done by adding a point to the control point list.
    """
    if lenghtTotal < realNeedleLength and extendNeedle:
      lastPoint = [controlPointListSorted[-1][0], controlPointListSorted[-1][1], controlPointListSorted[-1][2] - (realNeedleLength - lenghtTotal)]
      controlPointListSorted.append(lastPoint)

    scene = slicer.mrmlScene
    points = vtk.vtkPoints()
    polyData = vtk.vtkPolyData()
    lines = vtk.vtkCellArray()
    n = len(controlPointListSorted)
    lines.InsertNextCell(n)
    # start read out
    for i in range(n):
      pointIndex = points.InsertNextPoint(*(controlPointListSorted[i]))
      lines.InsertCellPoint(pointIndex)
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    # ## Create model node
    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetAndObservePolyData(polyData)
    # ## Create display node
    modelDisplay = slicer.vtkMRMLModelDisplayNode()

    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    # ## Add to scene
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    scene.AddNode(model)
    # ##Create Tube around the line
    tube = vtk.vtkTubeFilter()
    polyData = model.GetPolyData()
    tube.SetInputData(polyData)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(50)
    tube.Update()

    model.SetAndObservePolyData(tube.GetOutput())
    model.GetDisplayNode().Visibility2DOn()
    model.GetDisplayNode().SetOpacity(1-trans)
    if needleType == 'Validation':
      model.SetName('.manual-seg_' + str(colorVar))
    elif needleType == 'Obturator':
      model.SetName('.obturator-seg_' + str(colorVar))
    else:
      model.SetName('.auto-seg_' + str(self.round) + '-ID-' + str(model.GetID())+'-'+manualName)
    model.SetAttribute('type', needleType)

    if needleType == 'Validation' or needleType =='Debug':
      nth = int(colorVar) % MAXCOL
      modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
      model.SetAttribute("nth", str(nth))

    else:
      nth = int(model.GetID().strip('vtkMRMLModelNode')) % MAXCOL
      modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
      model.SetAttribute("nth", str(nth))

    if endMarker:
      fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      fiducial.SetName('.'+name+'_'+str(colorVar))
      fiducial.Initialize(slicer.mrmlScene)
      fiducial.SetFiducialCoordinates(controlPointListSorted[-1])
      fiducial.SetAttribute('TemporaryFiducial', '1')
      fiducial.SetLocked(True)
      displayNode = fiducial.GetDisplayNode()
      displayNode.SetGlyphScale(.5) # "arrow head"
      displayNode.SetColor(0, 0, 0)
      textNode = fiducial.GetAnnotationTextDisplayNode()
      textNode.SetTextScale(1)
      textNode.SetColor(0, 0, 0)
    if bPause: breakbox('Inspect this situation in the viewer!')


  def unique(self, a):
    '''
    Remove duplicates in 2D array
    '''
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


  def addNeedleToScene(self, controlPoint, colorVar, needleType='Detection', script=False, trans=0, manualName="", bPause=False):
    """Computes the Bezier's curve and adds visual representation of a needle to the scene

    :param controlPoint: array of RAS coordinates of points of a needle (used as control point for the Bezier's curve
    :param colorVar: color of the needle
    :param needleType: 'validation' for a manually segmentated needle, 'detection' for an automatically segmented needle,
    'obturator' for an obturator needle
    :return: visually, needle added to the scene
    """
    # productive
    profprint()
    """
    Create a model of the needle from its equation (Beziers curve fitting the control points)
    """
    widget = slicer.modules.NeedleFinderWidget

    realNeedleLength = widget.realNeedleLength.value
    extendNeedle = widget.extendNeedle.isChecked()
    # remove duplicates from list (bug with miccai2013)
    controlPoint = self.unique(np.array(controlPoint)).tolist()
    
    # sort the points in a decreasing order (from tip to bottom)
    controlPointListSorted = self.sortTableReverse(controlPoint, (2, 1, 0))

    # calculate the length of the list of ctr points
    lenghtTotal = 0
    for i in range(len(controlPoint) - 1):
        lenghtTotal += self.distanceTwoPoints(controlPointListSorted[i + 1], controlPointListSorted[i])
    print("Needle tube length: ",lenghtTotal)

    # in case we want to extend the needle to the wanted length
    """
    The extension is currently done by adding a point to the control point list.
    TODO: only append a straight tube, so it doesn't add a control point and modify
    the initial Bezier curve
    """
    if lenghtTotal < realNeedleLength and extendNeedle:
      lastPoint = [controlPointListSorted[-1][0], controlPointListSorted[-1][1], controlPointListSorted[-1][2] - (realNeedleLength - lenghtTotal)]
      controlPointListSorted.append(lastPoint)

    label = None
    scene = slicer.mrmlScene
    points = vtk.vtkPoints()
    polyData = vtk.vtkPolyData()
    lines = vtk.vtkCellArray()
    nbEvaluationPoints = 50
    n = len(controlPointListSorted) - 1
    Q = [[0, 0, 0] for t in range(nbEvaluationPoints + 1)]
    lines.InsertNextCell(nbEvaluationPoints + 1)
    # start calculation
    for t in range(nbEvaluationPoints+1): #+1):  #<<< lil bug    <<< we need +1, it's not a bug! otherwise needle is too short!
      tt = float(t) / (1 * nbEvaluationPoints)
      for j in range(3):
        for i in range(n + 1):
          Q[t][j] += self.binomial(n, i) * (1 - tt) ** (n - i) * tt ** i * controlPointListSorted[i][j]
      pointIndex = points.InsertNextPoint(*Q[t])
      lines.InsertCellPoint(pointIndex)
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    # ## Create model node
    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetAndObservePolyData(polyData)
    # ## Create display node
    modelDisplay = slicer.vtkMRMLModelDisplayNode()

    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    # ## Add to scene
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    scene.AddNode(model)
    # ##Create Tube around the line
    tube = vtk.vtkTubeFilter()
    polyData = model.GetPolyData()
    tube.SetInputData(polyData)
    tube.SetRadius(1)
    tube.SetNumberOfSides(50)
    tube.Update()

    model.SetAndObservePolyData(tube.GetOutput())
    model.GetDisplayNode().Visibility2DOn()
    model.GetDisplayNode().SetOpacity(1-trans)
    if needleType == 'Validation':
      model.SetName('manual-seg_' + str(colorVar))
    elif needleType == 'Obturator':
      model.SetName('obturator-seg_' + str(colorVar))
    else:
      model.SetName('auto-seg_' + str(self.round) + '-ID-' + str(model.GetID())+'-'+manualName)
    model.SetAttribute('type', needleType)

    self.lastNeedleNames.append(model.GetName())

    # evaluate and print the processing time
    processingTime = time.perf_counter() - self.t0
    # print processingTime

    if needleType == 'Validation':
      nth = int(colorVar) % MAXCOL
      modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
      model.SetAttribute("nth", str(nth))

    else:
      if widget.algoVersParameter.value<3: nth = int(model.GetID().strip('vtkMRMLModelNode')) % MAXCOL
      else: nth = int(colorVar) % MAXCOL
      modelDisplay.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
      model.SetAttribute("nth", str(nth))

    if needleType == 'Validation':
      1
      # self.addNeedleToTable(int(colorVar),label,'Validation') ### ??? why ID=color here
      # self.addNeedleToTable(int(model.GetID().strip('vtkMRMLModelNode')),label,'Validation')

    elif not script:
      self.addNeedleToTable(int(model.GetID().strip('vtkMRMLModelNode')), label)
