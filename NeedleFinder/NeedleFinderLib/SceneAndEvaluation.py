# -*- coding: utf-8 -*-
"""Scene management, UI table synchronization, parameter persistence, and Hausdorff evaluation."""

import configparser
import csv
import math
import operator
import os
import os.path
import random
import shutil
import time
import time as t
import numpy
import numpy as np
import SimpleITK as sitk
import sitkUtils
import vtk
import qt
import ctk
import slicer

from .Constants import COLORS_255, COLORS_RGB, HOLE_COORDINATES, HOLE_LABELS
from .GeometryUtils import extract_model_id_number
from .LegacyEditorCompat import EditorLib, Helper, LabelStatisticsLogic
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


class SceneAndEvaluationMixin(object):
  """Mixin providing scene cleanup, report table management, Hausdorff distance evaluation, and config I/O."""

  def deleteTempModels(self):
    """
    Delete control points from needle curves etc.
    """
    # research
    # remove old control pts and debug annotations from scene
    while slicer.util.getNodes('.*') != {}:
      nodes = slicer.util.getNodes('.*')
      for key,value in zip(list(nodes.keys()),list(nodes.values())):
        if key != ".temp": slicer.mrmlScene.RemoveNode(value)
    while slicer.util.getNodes('_*') != {}:
      nodes = slicer.util.getNodes('_*')
      for key,value in zip(list(nodes.keys()),list(nodes.values())):
        slicer.mrmlScene.RemoveNode(value)
    # ruler measurements
    while 0 and slicer.util.getNodes('M*') != {}:
      nodes = slicer.util.getNodes('M*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)
    # ROI annotations
    while 0 and slicer.util.getNodes('R*') != {}:
      nodes = slicer.util.getNodes('R*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)
    # fiducial annotations
    while 0 and slicer.util.getNodes('F*') != {}:
      nodes = slicer.util.getNodes('F*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)

  def deleteAllAutoNeedlesFromScene(self):
    """
    Delete every segmented needle of the current set
    """
    # productive #onButton
    profprint()
    while slicer.util.getNodes('auto-seg_' + str(self.round) + '*') != {}:
      nodes = slicer.util.getNodes('auto-seg_' + str(self.round) + '*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)

  def deleteAllModelsFromScene(self):
    """
    Delete every segmented needle, template slice markers and reset yellow segment
    """
    # research
    self.deleteNeedleDetectionModelsFromScene()
    self.deleteNeedleValidationModelsFromScene()

  def deleteNeedleValidationModelsFromScene(self):
    """
    Delete artifacts from validation from scene.
    """
    # producitve #onbutton
    profprint()
    while slicer.util.getNodes('manual-seg_*') != {}:
      nodes = slicer.util.getNodes('manual-seg_*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)
    while slicer.util.getNodes('obturator-seg_*') != {}:
      nodes = slicer.util.getNodes('obturator-seg_*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)
    # bezier control points
    self.deleteTempModels()
    while slicer.util.getNodes('template slice position*') != {}:
      nodes = slicer.util.getNodes('template slice position*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)

  def deleteNeedleDetectionModelsFromScene(self):
    """
    Delete artifacts from semi-auto segmentation from scene.
    """
    # productive #onButton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    # remove artifacts from segmentation editor, 3d models (from SE's model builder)
    if widget.currentLabel:
      for i in range(widget.currentLabel + 1):
        while slicer.util.getNodes(str(i)) != {}:
          nodes = slicer.util.getNodes(str(i))
          for node in list(nodes.values()):
            slicer.mrmlScene.RemoveNode(node)
    # remove artifacts from needle finder
    while slicer.util.getNodes('auto-seg_*') != {}:
      nodes = slicer.util.getNodes('auto-seg_*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)
    # while slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode') !={}:
      # nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode')
      # for node in nodes.values():
        # slicer.mrmlScene.RemoveNode(node)
    sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow")
    if sYellow == None :
        sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")
    sYellow.SetSliceVisible(0)
    reformatLogic = slicer.vtkSlicerReformatLogic()
    reformatLogic.SetSliceNormal(sYellow, 1, 0, 0)
    tempFidNodes = slicer.mrmlScene.GetNodesByName('.temp')
    for i in range(tempFidNodes.GetNumberOfItems()):
      node = tempFidNodes.GetItemAsObject(i)
      if node:
        slicer.mrmlScene.RemoveNode(node)
    # bezier control points
    self.deleteTempModels()
    sYellow.Modified()

  def deleteLastNeedle(self):
    """
    Delete last segmented needle of the current round
    """
    # productive #onButton
    profprint(self.getName())
    widget = slicer.modules.NeedleFinderWidget
    if widget.deleteNeedleButton.isEnabled() and self.lastNeedleNames:
      name = self.lastNeedleNames.pop()
      print("removing needle with name: ", name)
      while slicer.util.getNodes(name + '*') != {}:
        nodes = slicer.util.getNodes(name + '*')
        for node in list(nodes.values()):
          slicer.mrmlScene.RemoveNode(node)
      # rebuild report table
      model_id = extract_model_id_number(name)
      print("needle ID: <%s>" % model_id)
      if model_id is not None:
        self.deleteNeedleFromTable(int(model_id))

  def newInsertionNeedleSet(self):
    """
    Start a new round
    """
    # productive #onButton
    profbox()
    widget = slicer.modules.NeedleFinderWidget
    if widget.newInsertionButton:
      dialog = qt.QDialog()
      messageBox = qt.QMessageBox.information(dialog, 'Information', 'You are creating a new set of needles')
      self.round += 1
      widget.newInsertionButton.setText('Start a new set of needles - Round ' + str(self.round + 1) + '?')
      widget.deleteNeedleButton.setText('Delete Needles from round ' + str(self.round))

  def resetNeedleDetection(self, script=False):
    """
    Reset the needle detection to completely start over.
    """
    # productive #onButton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    ret = 0
    if script == False:
      dialog = qt.QDialog()
      ret = messageBox = qt.QMessageBox.question(dialog, 'Attention', """
          Are you sure that you want to reset the needle detection?
          It will delete every segmented needles...
          """, qt.QMessageBox.Ok, qt.QMessageBox.Cancel)
    if ret == qt.QMessageBox.Ok or script == True:
      self.deleteNeedleDetectionModelsFromScene()
      self.previousValues = [[0, 0, 0]]
      self.round = 1
      if widget.newInsertionButton:
        widget.newInsertionButton.setText('Start a new set of needles (' + str(self.round + 1) + ') ?')
        widget.deleteNeedleButton.setText('Delete Needles from set (' + str(self.round) + ")")
      # reset report table
      widget.table = None
      widget.row = 0
      widget.col = 0
      if not script:
        for i in widget.items:
          item = widget.items.pop()
          del item
        widget.items = None
        if widget.model.rowCount() > 0:
          for i in range(0, widget.model.rowCount()):
            ritem = widget.model.item(i)
            del ritem
          widget.model.removeRows(0, widget.model.rowCount())
        widget.model.clear()
        del widget.model
        widget.model = None
        widget.view.reset()
        slicer.modules.NeedleFinderWidget.analysisGroupBoxLayout.removeWidget(widget.view)
        del widget.view
        widget.view = None
        widget.initTableView()

      # ## Leave the needle detection mode
      widget.fiducialButton.checked = 0
      widget.stop()
      widget.fiducialButton.text = "2. Start Giving Needle Tips"

      # reset button and parameter states
      widget.templateSliceButton.setEnabled(1)
      widget.fiducialButton.setEnabled(0)
      widget.deleteNeedleButton.setEnabled(0)
      widget.resetDetectionButton.setEnabled(0)
      widget.labelMapNode = None
      widget.tempPointList = []
      widget.templateSliceButton.text = "1. Select Current Axial Slice as Seg. Limit (current: None)"
      if not script: widget.onResetParameters()

  def hideAnnotations(self):
    """
    This function shows/hides all text annotation in the scene.
    :return:
    """
    widget = slicer.modules.NeedleFinderWidget
    nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationTextDisplayNode')
    for i in range(nodes.GetNumberOfItems()):
        node = nodes.GetItemAsObject(i)
        if widget.hideAnnotationTextButton.checked:
          node.SetTextScale(0)
        else:
          node.SetTextScale(3)

  def resetNeedleValidation(self):
    """
    Reset the needle validation to completely start over.
    """
    # productive #onButton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    dialog = qt.QDialog()
    ret = messageBox = qt.QMessageBox.question(dialog, 'Attention', """
      Are you sure that you want to reset the needle validation?
      It will delete every segmented needles and the control points...
      """, qt.QMessageBox.Ok, qt.QMessageBox.Cancel)
    if ret == qt.QMessageBox.Ok:
      self.deleteNeedleValidationModelsFromScene()
      # while slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode') !={}:
        # nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode')
        # for node in nodes:
          # slicer.mrmlScene.RemoveNode(node)
      # self.deleteEvaluationNeedlesFromTable()

      widget.editNeedleTxtBox.value = 1
      widget.stepNeedle = 0
      self.tableValueCtrPt = [[[999, 999, 999] for i in range(100)] for j in range(100)]
      self.cleanTable()
      print("Manual needle validation segmentation reset!")

  def deleteEvaluationNeedlesFromTable(self):
    """
    Delete all manually segmented needles from report table.
    """
    # productive #onButton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.initTableView()
    for name in self.lastNeedleNames:
      print(name)
      ID = name.lstrip('manual-seg_')
      print("delete validation needle with ID: <%s>" % ID)
      try:
        ID = int(ID)
        self.deleteNeedleFromTable(ID)
      except ValueError:
        print("skipping invalid ID")

  def changeCursor(self, cursorNumber):
    """
    changes the cursor
    """
    # productive
    profprint()
    appLogic = slicer.app.applicationLogic()
    interactionNode = appLogic.GetInteractionNode()
    interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
    # baseImage = qt.QImage(":/Icons/AnnotationPointWithArrow.png")
    # width =  baseImage.width()
    # height = width
    # center= int(width/2)
    # cursorImage = qt.QImage(width, height, qt.QImage().Format_ARGB32)
    # cursorImage.fill(0)
    # cursorPixmap = qt.QPixmap()
    # cursorPixmap = cursorPixmap.fromImage(cursorImage)
    # cursor = qt.QCursor(cursorPixmap,center,0)

    layoutManager = slicer.app.layoutManager()
    sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    for nodeIndex in range(sliceNodeCount):
      # find the widget for each node in scene
      sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
      sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
      # sliceWidget.setCursor(cursor)    # doesn't work. Why?
      sliceWidget.setCursor(qt.QCursor(cursorNumber))

  def changeValue(self):
    """Read the value of the Qt widget and select this needle. It is then possible to display sequentially the points used as
     control points of this needle
    """
    # productive #onUpDnArrow
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    # widget.scrollPointButton.setText('Scroll Point for Needle ' + str(widget.editNeedleTxtBox.value) + ' (pt: ' + str(self.ptNumber) + ')')
    self.lockControlPoints(widget.editNeedleTxtBox.value)
    self.unlockControlPoints(widget.editNeedleTxtBox.value)
    widget.drawValidationNeedlesButton.text = "Render Manual Needle  " + str(widget.editNeedleTxtBox.value)


  def goToPoint(self, ID):
    """
    Focus the slices on the selected point
    :param ID: name of the fiducial point
    :return:
    """
    profprint()
    modelNode = getNode(ID)
    coord = [0, 0, 0]
    if modelNode != None:
      # if modelNode.GetAttribute("ValidationNeedle") == "1":
      modelNode.GetFiducialCoordinates(coord)
      X = coord[0]
      Y = coord[1]
      Z = coord[2]

      sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
      if sRed == None :
        sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode1")

      sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow")
      if sYellow == None :
        sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")

      sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeGreen")
      if sGreen == None :
        sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode3")

      mYellow = sYellow.GetSliceToRAS()
      mYellow.SetElement(0, 3, X)
      sYellow.Modified()
      sYellow.UpdateMatrices()

      mGreen = sGreen.GetSliceToRAS()
      mGreen.SetElement(1, 3, Y)
      sGreen.Modified()
      sGreen.UpdateMatrices()

      mRed = sRed.GetSliceToRAS()
      mRed.SetElement(2, 3, Z)
      sRed.Modified()
      sRed.UpdateMatrices()

  def renameObjects(self):
    nodes = slicer.util.getNodes('*\r')
    for node in list(nodes.values()):
      node.SetName(node.GetName()[:-1])


  def scrollPoint(self):
    """Reformat the axial view to display the slice containing the currently selected point.
    """
    # productive #onButton
    profprint()
    self.changeValue()
    widget = slicer.modules.NeedleFinderWidget
    needle = widget.editNeedleTxtBox.value
    # print self.ptNumber
    # print needle
    coord = [0, 0, 0]
    ptName = '.' + str(needle) + '-' + str(self.ptNumber)
    # print ptName
    modelNode = getNode(ptName)
    if modelNode != None:
        self.ptNumber = self.ptNumber + 1
        if modelNode.GetAttribute("ValidationNeedle") == "1":
          modelNode.GetFiducialCoordinates(coord)
          X = coord[0]
          Y = coord[1]
          Z = coord[2]

        sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
        if sRed == None :
          sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode1")

        sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow")
        if sYellow == None :
          sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")

        sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeGreen")
        if sGreen == None :
          sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode3")

        mYellow = sYellow.GetSliceToRAS()
        mYellow.SetElement(0, 3, X)
        sYellow.Modified()
        sYellow.UpdateMatrices()

        mGreen = sGreen.GetSliceToRAS()
        mGreen.SetElement(1, 3, Y)
        sGreen.Modified()
        sGreen.UpdateMatrices()

        mRed = sRed.GetSliceToRAS()
        mRed.SetElement(2, 3, Z)
        sRed.Modified()
        sRed.UpdateMatrices()
    elif self.ptNumber != 0:
        self.ptNumber = 0
        self.scrollPoint()

  def validationNeedle(self):
    """ When the button 'new validation needle' is pressed, the needle number is incremented, the needle step is reset to
     0

    """
    # productive #onButton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    # widget.editNeedleTxtBox.value += 1
    widget.validationNeedleButton.text = "New Validation Needle: (" + str(widget.editNeedleTxtBox.value) + ")->(" + str(widget.editNeedleTxtBox.value + 1) + ")"
    widget.editNeedleTxtBox.value += 1
    # self.tableValueCtrPt.append([])
    widget.stepNeedle = 0
    self.lockControlPoints(widget.editNeedleTxtBox.value)

  def filterWithSITK(self):
    """
    Demo method to filter image using SimpleITK.
    See https://github.com/Slicer/Slicer/blob/140b36b50877d85703c094a97fe13303dee570b5/Modules/Scripted/EditorLib/WatershedFromMarkerEffect.py
    """
    # research
    profbox()
    backgroundNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    backgroundNodeName = backgroundNode.GetName()
    backgroundImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(backgroundNodeName))
    filterImage = sitk.GradientMagnitudeRecursiveGaussian(backgroundImage, float(2));
    del backgroundImage
    sitk.WriteImage(filterImage, sitkUtils.GetSlicerITKReadWriteAddress(backgroundNodeName))

    # notify
    backgroundNode.GetImageData().Modified()
    backgroundNode.Modified()

  def writeTableHeader(self, fileName, variant=0):
    """
    Write table header to file.
    """
    # research
    w = slicer.modules.NeedleFinderWidget
    l = w.logic
    if not variant:
      l.exportEvaluation(['user','case','maxTipHD','maxHD', 'avgHD', 'stdHD', 'medHD',
                        'nNeedles','nOutliers','outliers',
                        'radiusNeedle',
                        'lenghtNeedle',
                        'radiusMax',
                        'numberOfPointsPerNeedle',
                        'nbRotatingIterations',
                        'stepSize',
                        'gradientPonderation',
                        'exponent',
                        'gaussianAttenuationButton',
                        'sigma',
                        'algoV',
                        'case',
                        t.strftime("%d/%m/%Y"), t.strftime("%H:%M:%S")
                        ], fileName)
    else:
      l.exportEvaluation(['user','case','tipHD','HD', 'man.-seg_', 'ID1', 'ID2',
                        'outlier?',
                        'radiusNeedle',
                        'lenghtNeedle',
                        'radiusMax',
                        'numberOfPointsPerNeedle',
                        'nbRotatingIterations',
                        'stepSize',
                        'gradientPonderation',
                        'exponent',
                        'gaussianAttenuationButton',
                        'sigma',
                        'algoV',
                        #'case',
                        t.strftime("%d/%m/%Y"), t.strftime("%H:%M:%S")
                        ], fileName)

  def parSearch(self, mode=False):
    """
    Parameter evaluation/optimization using no, grid brute-force or random-search algo.
    """
    # research
    profprint()
    w = slicer.modules.NeedleFinderWidget
    l = w.logic
    path = [ 0 for i in range(100)]
    
    if 0:
      path[24] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 24 NRRD/Manual/2013-02-25-Scene-without-CtrPt.mrml'
      path[29] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 29 NRRD/Manual/2013-02-26-Scene-without-CtrPts.mrml'
      path[30] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 30 NRRD/Manual/2013-02-26-Scene-without-CtrPt.mrml'
      path[31] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 31 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
      path[34] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 34 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
      path[35] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 35 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
      path[37] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 37 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
      path[38] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 38 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
      path[40] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD/Case 40 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'

    #Andre's file system (case copies from AMIGO share)
    # stripped OTHER cases
    if 0: path[33] = '/home/mastmeyer/Dropbox/GYN Cases/Case  033/NRRD/Auto-Eval-LB/2013-02-27-Scene.mrml'
    if 0:
      path[ 8] = '/home/mastmeyer/Dropbox/GYN Cases/Case  008/NRRD/Auto-Eval-LB/2013-05-07-Scene.mrml'
      path[12] = '/home/mastmeyer/Dropbox/GYN Cases/Case  012/NRRD/Auto-Eval-LB/2013-04-22-Scene.mrml'
      path[16] = '/home/mastmeyer/Dropbox/GYN Cases/Case  016/NRRD/Auto-Eval-LB/2013-04-21-Scene.mrml'
      path[21] = '/home/mastmeyer/Dropbox/GYN Cases/Case  021/NRRD/Auto-Eval-LB/2013-04-21-Scene.mrml'
      path[22] = '/home/mastmeyer/Dropbox/GYN Cases/Case  022/NRRD/Auto-Eval-LB/2013-04-21-Scene.mrml'
      path[25] = '/home/mastmeyer/Dropbox/GYN Cases/Case  025/NRRD/Auto-Eval-LB/2013-04-21-Scene.mrml'
      path[26] = '/home/mastmeyer/Dropbox/GYN Cases/Case  026/NRRD/Auto-Eval-LB/2013-04-17-Scene.mrml'
      path[27] = '/home/mastmeyer/Dropbox/GYN Cases/Case  027/NRRD/Auto-Eval-LB/2013-04-17-Scene.mrml'
    #stripped MICCAI13 cases (just manual seg. by LB/AM)
    if 1:
      path[24] = '/home/mastmeyer/Dropbox/GYN Cases/Case  024/NRRD/Auto-Eval-LB/2013-02-28-Scene.mrml'
      path[28] = '/home/mastmeyer/Dropbox/GYN Cases/Case  028/NRRD/Auto-Eval-LB/2013-02-28-Scene.mrml'
      path[29] = '/home/mastmeyer/Dropbox/GYN Cases/Case  029/NRRD/Auto-Eval-LB/2013-02-26-Scene.mrml'
      path[30] = '/home/mastmeyer/Dropbox/GYN Cases/Case  030/NRRD/Auto-Eval-LB/2013-02-26-Scene.mrml'
      path[31] = '/home/mastmeyer/Dropbox/GYN Cases/Case  031/NRRD/Auto-Eval-LB/2013-02-27-Scene.mrml'
      path[33] = '/home/mastmeyer/Dropbox/GYN Cases/Case  033/NRRD/Auto-Eval-LB/2013-02-27-Scene.mrml'
      path[34] = '/home/mastmeyer/Dropbox/GYN Cases/Case  034/NRRD/Auto-Eval-LB/2013-02-27-Scene.mrml'
      path[37] = '/home/mastmeyer/Dropbox/GYN Cases/Case  037/NRRD/Manual Alireza/2013-02-27-Scene.mrml'
      path[38] = '/home/mastmeyer/Dropbox/GYN Cases/Case  038/NRRD/Manual Alireza/2013-02-27-Scene.mrml'
      path[40] = '/home/mastmeyer/Dropbox/GYN Cases/Case  040/NRRD/Manual Alireza/2013-02-27-Scene.mrml'
    #show a directory selector for saving the results
    self.dirDialog = qt.QFileDialog(w.parent)
    self.dirDialog.setDirectory('/tmp')
    self.dirDialog.options = self.dirDialog.ShowDirsOnly
    self.dirDialog.acceptMode = self.dirDialog.AcceptSave
    #self.dirDialog.show()
    dir=self.dirDialog.getExistingDirectory()
    w.logDir=dir
    print("saving results to ", dir)
    try: shutil.copyfile('/home/amast/WualaDrive/mastmeyer/Homes/NeedleFinder/NeedleFinder/NeedleFinder.py',dir+'/NeedleFinder_ref.py')
    except: breakbox("/!\\ reference source NeedleFinder.py not found!")
    if mode == 0:
      #save a copy of the source file as reference
      # simple run with current parameters/algo over several patients
      self.writeTableHeader(dir+'/AP-All_stats.csv')
      filLog=open(dir+'/allog.tsv', 'w')
      #filLog.write("case\tman.-seg_\tiStep\tcrit\treject\tvalue\tlimit\n")
      filLog.close()
      nUsers=1 #CONST
      for user in range(nUsers): 
        w.userNr=user
        print("simulated user (offset): ",user)
        for id in range(100): #<o> range(100)
          if path[id]:
            w.caseNr=id
            print("processing ", path[id])
            self.writeTableHeader(dir+'/User-'+str(user)+'_AP-' + str(id) + '.csv', 1)
            slicer.mrmlScene.Clear(0)
            slicer.util.loadScene(path[id])
            #TODO implement random tips in  a sphere (d=2mm) from tube center 
            l.startValidation(script=True, offset=user*50/nUsers)
            results, outliers = l.evaluate(script=True)  # calculate HD distances
            for result in results:
              result[0:0]=[user,id]
            l.exportEvaluation(results, dir+'/User-'+str(user)+'_AP-' + str(id) + '.csv')
            #slicer.util.saveScene(dir+'/AP-' + str(id) + '.mrb') # may use lots of disk space
            # stats
            HD = np.array(results)
            # HD.shape = (int(len(results)/float(3)),3)
            maxTipHD = HD[:, 2].max()
            maxHD = HD[:, 3].max()
            avgHD = HD[:, 3].mean()
            stdHD = HD[:, 3].std()
            sl = np.sort(HD[:, 3])
            medHD = sl[sl.size // 2]
            resultsEval = [user,id,maxTipHD, maxHD, avgHD, stdHD, medHD]+[len(results)]+[len(outliers)] +[str(outliers)]+ l.valuesExperience + [id]
            l.exportEvaluation(resultsEval, dir+'/AP-All_stats.csv')
            #pause()
      msgbox("parSearch mode 0 done, results in "+dir)
    elif mode == 1:
      id = 'Current'
      # simple brute force search in the dimensions (Guillaumes parameterSearch.py)
      self.writeTableHeader(dir+'/BF-' + str(id) + '.csv', 1)
      self.writeTableHeader(dir+'/BF-' + str(id) + '_stats.csv')
      for i in range(3, 12):
        # l.resetNeedleDetection(script=True) # ??? this resets the parameters to default
        w.numberOfPointsPerNeedle.setValue(i)  # change parameter control points
        l.startValidation(script=True)
        results, outliers = l.evaluate(script=True)  # calculate HD distances
        for result in results:
          result[0:0]=[user,id]
        l.exportEvaluation(results, dir+'/BF-' + str(id) + '.csv')
        slicer.util.saveScene(dir+'/BF-' + str(id) + '.mrb') # may use lots of disk space
        # stats
        HD = np.array(results)
        # HD.shape = (int(len(results)/float(3)),3)
        maxTipHD = HD[:, 2].max()
        maxHD = HD[:, 3].max()
        avgHD = HD[:, 3].mean()
        stdHD = HD[:, 3].std()
        sl = np.sort(HD[:, 3])
        medHD = sl[sl.size // 2]
        resultsEval = [user,id,maxTipHD,maxHD, avgHD, stdHD, medHD] +[len(results)]+[len(outliers)] +[str(outliers)]+ l.valuesExperience + [id]
        l.exportEvaluation(resultsEval, dir+'/BF-' + str(id) + '_stats.csv')
        #pause()
      msgbox("parSearch mode 1 done, results in "+dir)
    elif mode == 2:
      # code piece from Guillaumes (bruteForce.py) multi patient mode search
      for id in range(100):
        if path[id]:
          w.caseNr=id
          print("processing ", path[id])
          slicer.mrmlScene.Clear(0)
          slicer.util.loadScene(path[id])
          self.writeTableHeader(dir+'/RS-' + str(id) + '.csv', 1)
          self.writeTableHeader(dir+'/RS-' + str(id) + '_stats.csv')
          for i in range(1, 10000):
            # l.resetNeedleDetection(script=True) # ??? this resets the parameters to default
            w.radiusNeedleParameter.setValue(np.random.randint(1, 6))
            w.stepsize.setValue(np.random.randint(1, 40))
            w.sigmaValue.setValue(np.random.randint(1, 40))  # change parameter sigma
            w.gradientPonderation.setValue(np.random.randint(1, 20))
            w.exponent.setValue(np.random.randint(1, 20))
            w.numberOfPointsPerNeedle.setValue(np.random.randint(3, 11))
            l.startValidation(script=True)
            results, outliers = l.evaluate(script=True)  # calculate HD distances
            for result in results:
              result[0:0]=[user,id]
            l.exportEvaluation(results, dir+'/RS-' + str(id) + '.csv')
            slicer.util.saveScene(dir+'/RS-' + str(id) + '.mrb') # may use lots of disk space
            # stats
            HD = np.array(results)
            maxTipHD = HD[:, 2].max()
            maxHD = HD[:, 3].max()
            avgHD = HD[:, 3].mean()
            stdHD = HD[:, 3].std()
            sl = np.sort(HD[:, 3])
            medHD = sl[sl.size // 2]
            resultsEval = [user,id,maxTipHD,maxHD, avgHD, stdHD, medHD] +[len(results)]+[len(outliers)] +[str(outliers)]+ l.valuesExperience + [id]
            l.exportEvaluation(resultsEval, dir+'/RS-' + str(id) + '_stats.csv')
            # end = time.time()
            # print 'processing time: ', end-start
            # start = time.time()
            #pause()
        msgbox("parSearch mode 2 done, results in "+dir)
      #rof id
    #file mode 2
    slicer.mrmlScene.Clear(0) #clean up to save memory

  #----------------------------------------------------------------------------------------------
  """ Needle segmentation report"""
  #----------------------------------------------------------------------------------------------

  def addNeedleToTable(self, ID, label=None, needleType=None):
    """
    Add last segmented needle to the table
    The color icon corresponds to the color of the needle, which corresponds to its label (color code)
    """
    # productive
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.initTableViewControlPoints()
    widget.initTableView()
    if label != None:
      ref = int(label[0]) % MAXNEEDLES
      needleLabel = self.option[ref]
      # reliability = label[1]
    else:
      needleLabel = str(ID)
      ref = ID % MAXNEEDLES
      # reliability = '-'
    # ref = int(modelNode.GetAttribute("nth"))

    widget.labelStats["Labels"].append(ref)
    widget.labelStats[ref, "#"] = needleLabel
    # self.labelStats[ref,"Round"] = str(self.round)
    # self.labelStats[ref,"Reliability"] = str(reliability)

    ################################################
    # Column 0
    color = qt.QColor()
    color.setRgb(self.color255[ref][0], self.color255[ref][1], self.color255[ref][2])
    item = qt.QStandardItem()
    item.setData(color, 1)
    # self.model.appendRow(item)
    widget.model.setItem(widget.row, 0, item)
    widget.items.append(item)
    ################################################
    # Column 1
    widget.col = 1
    for k in widget.keys:
      item = qt.QStandardItem()
      item.setText(widget.labelStats[ref, k])
      widget.model.setItem(widget.row, widget.col, item)
      widget.items.append(item)
      widget.col += 1
    ################################################
    # Column 2
    displayButton = qt.QPushButton("Display")
    displayButton.checked = True
    displayButton.checkable = True
    if needleType == 'Validation':
      ID = int(getNode('manual-seg_' + str(ID)).GetID().strip('vtkMRMLModelNode'))
    displayButton.connect("clicked()", lambda who=ID: self.displayNeedleTube(who))
    index = widget.model.index(widget.row, 2)

    widget.items.append(displayButton)
    widget.col += 1
    widget.view.setIndexWidget(index, displayButton)
    ################################################
    # Column 3
    reformatButton = qt.QPushButton("Reformat")
    reformatButton.connect("clicked()", lambda who=ID: self.reformatSagittalView4Needle(who))
    index = widget.model.index(widget.row, 3)
    widget.items.append(reformatButton)
    widget.col += 1
    widget.view.setIndexWidget(index, reformatButton)
    ################################################
    # Column 4
    editField = qt.QTextEdit("")
    index = widget.model.index(widget.row, 4)
    widget.items.append(editField)
    widget.col += 1
    widget.view.setIndexWidget(index, editField)

    widget.row += 1

  def deleteNeedleFromTable(self, ID):
    """
    Delete last needle from model
    """
    profprint()
    # productive #onButton
    widget = slicer.modules.NeedleFinderWidget
    print("len(items): ", len(widget.items))
    if widget.row and ID in widget.labelStats.get("Labels", []):
      pos = widget.labelStats["Labels"].index(ID)
      ref = ID % MAXNEEDLES
      widget.labelStats["Labels"].pop(pos)
      widget.labelStats[ref, "Label"] = None
      pos += 1
      for i in range(1, widget.col + 1):
        item = widget.items.pop(pos * widget.col - i)
        del item
      pos -= 1
      ritem = widget.model.item(pos)
      del ritem
      widget.model.removeRow(pos)
      widget.row -= 1

  def addPointToTable(self, ID, needleNumber, pointNumber, label=None, needleType=None, sortTable = 1):
    """
    Add control point to the table
    The color icon corresponds to the color of the needle, which corresponds to its label (color code)
    """
    # productive
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.initTableViewControlPoints()

    widget.labelStatsCTL["Labels"].append(ID)
    widget.labelStatsCTL["ID"].append(1000*needleNumber + pointNumber)
    # self.labelStatsCTL[needleNumber, "#"] = needleLabel

    ################################################
    # Column 0
    color = qt.QColor()
    color.setRgb(self.color255[needleNumber][0], self.color255[needleNumber][1], self.color255[needleNumber][2])
    item = qt.QStandardItem()
    item.setData(color, 1)
    widget.modelCTL.setItem(widget.rowCTL, 0, item)
    widget.itemsCTL.append(item)
    ################################################
    # Column 1
    colCTL = 1
    item = qt.QStandardItem()
    item.setText(str(needleNumber).zfill(2))
    widget.modelCTL.setItem(widget.rowCTL, colCTL, item)
    widget.itemsCTL.append(item)
    ################################################
    # Column 1
    colCTL = 2
    item = qt.QStandardItem()
    item.setText(str(pointNumber).zfill(2))
    widget.modelCTL.setItem(widget.rowCTL, colCTL, item)
    widget.itemsCTL.append(item)
    # ################################################
    # Column 3
    displayButton = qt.QPushButton("Delete")
    displayButton.checked = True
    displayButton.checkable = True
    if needleType == 'Validation':
      ID = int(getNode('.' + str(ID)).GetID().strip('vtkMRMLModelNode'))
    listArgs = [ID, needleNumber, pointNumber]
    displayButton.connect("clicked()", lambda who=listArgs: self.removeNeedleShaftEvalMarker(who))
    index = widget.modelCTL.index(widget.rowCTL, 3)
    widget.itemsCTL.append(displayButton)
    widget.viewCTL.setIndexWidget(index, displayButton)
    # ################################################
    # Column 3
    reformatButton = qt.QPushButton("Focus")
    reformatButton.connect("clicked()", lambda who=ID: self.goToPoint(who))
    index = widget.modelCTL.index(widget.rowCTL, 4)
    widget.itemsCTL.append(reformatButton)
    widget.viewCTL.setIndexWidget(index, reformatButton)
    # ################################################
    # # Column 4
    # editField = qt.QTextEdit("")

    # self.items.append(editField)
    # self.col += 1
    # self.view.setIndexWidget(index, editField)
    widget.rowCTL += 1
    if sortTable:
      widget.viewCTL.sortByColumn(2)
      widget.viewCTL.sortByColumn(1)

  def deletePointFromTable(self, ID):
    """
    Delete point from table
    """
    profprint()
    # productive #onButton
    widget = slicer.modules.NeedleFinderWidget
    sortedIndex = widget.labelStatsCTL['ID']
    sortedIndex.sort()
    sortedIndex.reverse()
    # print sortedIndex
    # print ID
    label = ID.split('.')
    labelval = label[1].split('-')
    # print labelval
    val = 1000 * int(labelval[0]) + int(labelval[1])
    # print "val: ", val
    # print "len(items): ", len(widget.itemsCTL)
    if widget.rowCTL:
      pos = sortedIndex.index(val)
      widget.labelStatsCTL["Labels"].pop(pos)
      widget.labelStatsCTL["ID"].pop(pos)
      pos += 1
      for i in range(1, widget.colCTL + 1):
        item = widget.itemsCTL.pop(pos * widget.colCTL - i)
        del item
      pos -= 1
      ritem = widget.modelCTL.item(pos)
      del ritem
      widget.modelCTL.removeRow(pos)
      widget.rowCTL -= 1

  def cleanTable(self):
    '''
    Clean report table and internal variables
    :return:
    '''
    profprint()
    # productive #onButton
    widget = slicer.modules.NeedleFinderWidget
    widget.labelStatsCTL = {}
    widget.labelStats = {}

    # table report
    widget.table = None
    widget.tableCTL = None
    if widget.modelCTL:
      widget.modelCTL.clear()
    if widget.model:
      widget.model.clear()
    # widget.view = None
    # widget.viewCTL = None
    # widget.model = None
    # widget.modelCTL = None
    # look for duplicate list and delete it if necessary
    nodes = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode','Fiducials List')
    for i in range(nodes.GetNumberOfItems()):
      slicer.mrmlScene.RemoveNode(nodes.GetItemAsObject(i))

    nodes = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode','All Annotations')
    for i in range(nodes.GetNumberOfItems()):
      slicer.mrmlScene.RemoveNode(nodes.GetItemAsObject(i))

    self.observeManualNeedles()

  #-----------------------------------------------------------
  # Radiation

  def AddRadiation(self, i, needleID):
    """
    Goal of this function is to draw quadrics simulating the dose radiation.
    Currently, ellipse is a too naive model.
    This project might be continued later
    """
    # obsolete
    profbox()
    pass
    # needleNode = slicer.mrmlScene.GetNodeByID(needleID)
    # polyData = needleNode.GetPolyData()
    # nb = polyData.GetNumberOfPoints()
    # base = [0,0,0]
    # tip = [[0,0,0] for k in range(11)]
    # if nb>100:

      # polyData.GetPoint(nb-1,tip[10])
      # polyData.GetPoint(0,base)

    # a = tip[10][0]-base[0]
    # b = tip[10][1]-base[1]
    # c = tip[10][2]-base[2]

    # for l in range(7):
      # tip[9-l][0] = tip[10][0]-0.1*a*(l+1)
      # tip[9-l][1] = tip[10][1]-0.1*b*(l+1)
      # tip[9-l][2] = tip[10][2]-0.1*c*(l+1)
    # for l in range(1,3):
      # tip[l][0] = tip[10][0]+0.1*a*l
      # tip[l][1] = tip[10][1]+0.1*b*l
      # tip[l][2] = tip[10][2]+0.1*c*l

    # rad = vtk.vtkAppendPolyData()

    # for l in range(1,11):
      # TransformPolyDataFilter=vtk.vtkTransformPolyDataFilter()
      # Transform=vtk.vtkTransform()
      # TransformPolyDataFilter.SetInput(self.m_polyRadiation)

      # vtkmat = Transform.GetMatrix()
      # vtkmat.SetElement(0,3,tip[l][0])
      # vtkmat.SetElement(1,3,tip[l][1])
      # vtkmat.SetElement(2,3,tip[l][2])
      # TransformPolyDataFilter.SetTransform(Transform)
      # TransformPolyDataFilter.Update()

      # rad.AddInput(TransformPolyDataFilter.GetOutput())

    # modelNode = slicer.vtkMRMLModelNode()
    # displayNode = slicer.vtkMRMLModelDisplayNode()
    # storageNode = slicer.vtkMRMLModelStorageNode()

    # fileName = 'Rad'+self.option[i]+'.vtk'

    # mrmlScene = slicer.mrmlScene
    # modelNode.SetName(fileName)
    # modelNode.SetAttribute("radiation","segmented")
    # modelNode.SetAttribute("needleID",str(needleID))
    # modelNode.SetAndObservePolyData(rad.GetOutput())

    # modelNode.SetScene(mrmlScene)
    # storageNode.SetScene(mrmlScene)
    # storageNode.SetFileName(fileName)
    # displayNode.SetScene(mrmlScene)
    # displayNode.SetVisibility(0)
    # mrmlScene.AddNode(storageNode)
    # mrmlScene.AddNode(displayNode)
    # mrmlScene.AddNode(modelNode)
    # modelNode.SetAndObserveStorageNodeID(storageNode.GetID())
    # modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

    # displayNode.SetPolyData(modelNode.GetPolyData())

    # displayNode.SetVisibility2D(0)
    # displayNode.SetScalarVisibility(1)
    # displayNode.SetActiveScalarName('scalars')
    # displayNode.SetScalarRange(0,230)
    # displayNode.SetOpacity(0.06)
    # displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileHotToColdRainbow.txt')
    # displayNode.SetBackfaceCulling(0)
    # pNode= self.parameterNode()
    # pNode.SetParameter(fileName,modelNode.GetID())
    # mrmlScene.AddNode(modelNode)

  #----------------------------------------------------------------------------------------------
  """
  The purpose of the following functions is to process the results of the needle segmentation from
  Yi Gao's CLI module.
  Currently, another solution has been chosen but this could be usefull again later.
  To use it, simply uncommented the corresponding buttons in "createUserInterface"
  """
  #----------------------------------------------------------------------------------------------

  def needleSegmentationCLIDEMO(self):
    """
    Function used for the CLI module from Yi Gao using hessian filter (C++)
    """
    # research
    profbox()
    widget = slicer.modules.NeedleFinderWidget
    scene = slicer.mrmlScene
    pNode = self.parameterNode()
    if slicer.mrmlScene.GetNodeByID(pNode.GetParameter("baselineVolumeID")) == None:
      inputVolume = self.__volumeSelector.currentNode()
      inputVolumeID = self.__volumeSelector.currentNode().GetID()
    else:
      inputVolume = slicer.mrmlScene.GetNodeByID(pNode.GetParameter("baselineVolumeID"))
      inputVolumeID = slicer.mrmlScene.GetNodeByID(pNode.GetParameter("baselineVolumeID")).GetID()
    inputLabelID = self.__needleLabelSelector.currentNode().GetID()

    datetime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

    inputVolume.SetAttribute("foldername", datetime)
    self.outputVolumeNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLModelNode')
    self.outputVolumeNode.SetName("Output Needle Model")
    outputVolumeStorageNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLModelStorageNode')
    slicer.mrmlScene.AddNode(self.outputVolumeNode)
    slicer.mrmlScene.AddNode(outputVolumeStorageNode)
    self.outputVolumeNode.AddAndObserveStorageNodeID(outputVolumeStorageNode.GetID())
    outputVolumeStorageNode.WriteData(self.outputVolumeNode)

    outputID = self.outputVolumeNode.GetID()

    self.foldername = '/NeedleModels/' + datetime

    # Set the parameters for the CLI module
    parameters = {}
    parameters['inputVolume'] = inputVolumeID
    parameters['inputLabel'] = inputLabelID
    parameters['outputVtk'] = outputID
    parameters['outputFolderName'] = self.foldername
    parameters['nbPointsPerLine'] = self.nbPointsPerLine.value
    parameters['nbRadiusIterations'] = self.nbRadiusIterations.value
    parameters['radiusMax'] = self.radiusMax.value
    parameters['numberOfPointsPerNeedle'] = self.numberOfPointsPerNeedle.value
    parameters['nbRotatingIterations'] = self.nbRotatingIterations.value

    module = slicer.modules.mainlabelneedletrackingcli
    self.__cliNode = None
    self.__cliNode = slicer.cli.run(module, None, parameters, wait_for_completion=True)

    ##### match the needles ######

    self.setNeedleCoordinates()
    self.computerPolydataAndMatrix()
    xmin = min(self.p[0])
    xmax = max(self.p[0])
    ymin = min(self.p[1])
    ymax = max(self.p[1])
    xdelta = xmax - xmin
    ydelta = ymax - ymin
    k = 0

    self.base = [[0 for j in range(3)] for j in range(63)]
    self.tip = [[0 for j in range(3)] for j in range(63)]
    self.needlenode = [[0 for j in range(2)] for j in range(63)]
    self.bentNeedleNode = [[0 for j in range(2)] for j in range(63)]
    self.displaynode = [0 for j in range(63)]
    self.displaynodeB = [0 for j in range(63)]
    self.fiducialnode = [0 for j in range(63)]

    for i in range(63):

      pathneedle = self.foldername + '/' + str(i) + '.vtp'
      pathBentNeedle = self.foldername + '/' + str(i) + '_bent.vtp'
      self.needlenode[i] = slicer.util.loadModel(pathneedle, True)
      self.bentNeedleNode[i] = slicer.util.loadModel(pathBentNeedle, True)

      if self.needlenode[i][0] == True and self.needlenode[i][1] != None:
        self.displaynode[i] = self.needlenode[i][1].GetDisplayNode()
        self.displaynodeB[i] = self.bentNeedleNode[i][1].GetDisplayNode()


        polydata = self.needlenode[i][1].GetPolyData()
        polydata.GetPoint(0, self.base[i])

        self.displaynode[i].Visibility2DOn()
        self.displaynodeB[i].Visibility2DOn()
        bestmatch = None
        mindist = None
        for j in range(63):
          delta = ((self.p[0][j] - (self.base[i][0])) ** 2 + (self.p[1][j] - self.base[i][1]) ** 2) ** (0.5)
          if delta < mindist or mindist == None:
            bestmatch = j
            mindist = delta

        bestmatch = k
        k += 1
        self.displaynode[i].SetColor(self.color[bestmatch])
        self.displaynodeB[i].SetColor(self.color[bestmatch])
        self.needlenode[i][1].SetName(self.option[bestmatch] + "_segmented")
        self.bentNeedleNode[i][1].SetName(self.option[bestmatch] + "_optimized")
        self.needlenode[i][1].SetAttribute("segmented", "1")
        self.bentNeedleNode[i][1].SetAttribute("optimized", "1")
        self.needlenode[i][1].SetAttribute("nth", str(bestmatch))
        self.bentNeedleNode[i][1].SetAttribute("nth", str(bestmatch))
        self.needlenode[i][1].SetAttribute("needleID", self.needlenode[i][1].GetID())
        self.bentNeedleNode[i][1].SetAttribute("needleID", self.bentNeedleNode[i][1].GetID())

    if widget.removeDuplicates.isChecked():
      self.positionFilteringNeedles()

    d = slicer.mrmlScene.GetNodeByID(outputID).GetDisplayNode()
    d.SetVisibility(0)

    self.__editorFrame.collapsed = 1

    self.addButtons()

  def addButtons(self):
    """Buttons for the reporting widget, displaying information of the segmented needles.
    Used in conjunction with Yi Gaos CLI stuff.
    """
    profbox()
    if self.buttonsGroupBox != None:
      self.layout.removeWidget(self.buttonsGroupBox)
      self.buttonsGroupBox.deleteLater()
      self.buttonsGroupBox = None
    self.buttonsGroupBox = qt.QGroupBox()
    self.buttonsGroupBox.setTitle('Manage Needles')
    self.layout.addRow(self.buttonsGroupBox)
    self.buttonsGroupBoxLayout = qt.QFormLayout(self.buttonsGroupBox)

    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("segmented") == "1":
        i = int(modelNode.GetAttribute("nth"))
        buttonDisplay = qt.QPushButton("Hide " + self.option[i])
        buttonBentDisplay = qt.QPushButton("Hide Bent " + self.option[i])
        buttonDisplay.checkable = True
        buttonBentDisplay.checkable = True

        if modelNode.GetDisplayVisibility() == 0:
          buttonDisplay.setChecked(1)

        buttonDisplay.connect("clicked()", lambda who=i: self.displayNeedle(who))
        buttonBentDisplay.connect("clicked()", lambda who=i: self.displayBentNeedle(who))
        buttonReformat = qt.QPushButton("Reformat " + self.option[i])
        buttonReformat.connect("clicked()", lambda who=i: self.reformatSagittalView4Needle(who))
        widgets = qt.QWidget()
        hlay = qt.QHBoxLayout(widgets)
        hlay.addWidget(buttonDisplay)
        hlay.addWidget(buttonBentDisplay)
        hlay.addWidget(buttonReformat)
        self.buttonsGroupBoxLayout.addRow(widgets)

  def displayBentNeedle(self, i):
    """
    not actively used anymore. works with Yi Gao CLI module for straight needle detection + bending post-computed
    """
    profbox()
    # obsolete
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("nth") == str(i) and modelNode.GetAttribute("optimized") == '1' :
        displayNode = modelNode.GetModelDisplayNode()
        nVisibility = displayNode.GetVisibility()
        # print nVisibility
        if nVisibility:
          displayNode.Visibility2DOff()
          displayNode.SetVisibility(0)
        else:
          displayNode.Visibility2DOn()
          displayNode.SetVisibility(1)

  def displayNeedle(self, i):
    """
    ??? not used anymore. works with Yi Gao CLI module for straight needle detection + bending post-computed
    """
    # obsolete
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("nth") == str(i) and modelNode.GetAttribute("segmented") == '1' :
        displayNode = modelNode.GetModelDisplayNode()
        nVisibility = displayNode.GetVisibility()

        if nVisibility:
          displayNode.Visibility2DOff()
          displayNode.SetVisibility(0)
        else:
          displayNode.Visibility2DOn()
          displayNode.SetVisibility(1)

  def showOneNeedle(self, i, visibility):
    """
    ??? Not used anymore. But can be usefull later
    """
    # obsolete
    profbox()
    fidname = "fid" + self.option[i]
    pNode = self.parameterNode()
    needleID = pNode.GetParameter(self.option[i] + '.vtp')
    fidID = pNode.GetParameter(fidname)
    NeedleNode = slicer.mrmlScene.GetNodeByID(needleID)
    fiducialNode = slicer.mrmlScene.GetNodeByID(fidID)

    if NeedleNode != None:
      displayNode = NeedleNode.GetModelDisplayNode()
      nVisibility = displayNode.GetVisibility()

      if fiducialNode == None:
        displayNode.SetVisibility(1)
        displayNode.SetOpacity(0.9)
        polyData = NeedleNode.GetPolyData()
        polyData.Update()
        nb = int(polyData.GetNumberOfPoints() - 1)
        coord = [0, 0, 0]
        if nb > 100:
          fiducialNode = slicer.vtkMRMLAnnotationFiducialNode()
          polyData.GetPoint(nb, coord)
          fiducialNode.SetName(self.option[i])
          fiducialNode.SetFiducialCoordinates(coord)
          fiducialNode.Initialize(slicer.mrmlScene)
          fiducialNode.SetLocked(1)
          fiducialNode.SetSelectable(0)
          fidDN = fiducialNode.GetDisplayNode()
          fidDN.SetColor(NeedleNode.GetDisplayNode().GetColor())
          fidDN.SetGlyphScale(0)
          fidTN = fiducialNode.GetAnnotationTextDisplayNode()
          fidTN.SetTextScale(3)
          fidTN.SetColor(NeedleNode.GetDisplayNode().GetColor())
          fiducialNode.SetDisplayVisibility(0)
          pNode.SetParameter(fidname, fiducialNode.GetID())
          fiducialNode.SetDisplayVisibility(1)

      if visibility == 0:

        displayNode.SetVisibility(0)
        displayNode.SetVisibility2D(0)
        if fiducialNode != None:
          fiducialNode.SetDisplayVisibility(0)

      else:

        displayNode.SetVisibility(1)
        displayNode.SetVisibility2D(1)
        if fiducialNode != None:
          fiducialNode.SetDisplayVisibility(1)

    else:
      vtkmat = vtk.vtkMatrix4x4()
      vtkmat.DeepCopy(self.m_vtkmat)
      vtkmat.SetElement(0, 3, self.m_vtkmat.GetElement(0, 3) + self.p[0][i])
      vtkmat.SetElement(1, 3, self.m_vtkmat.GetElement(1, 3) + self.p[1][i])
      vtkmat.SetElement(2, 3, self.m_vtkmat.GetElement(2, 3) + (30.0 - 150.0) / 2.0)

      TransformPolyDataFilter = vtk.vtkTransformPolyDataFilter()
      Transform = vtk.vtkTransform()
      TransformPolyDataFilter.SetInputData(self.m_polyCylinder)
      Transform.SetMatrix(vtkmat)
      TransformPolyDataFilter.SetTransform(Transform)
      TransformPolyDataFilter.Update()

      triangles = vtk.vtkTriangleFilter()
      triangles.SetInputConnection(TransformPolyDataFilter.GetOutputPort())
      self.AddModel(i, triangles.GetOutput())
      self.showOneNeedle(i, visibility)

  def AddModel(self, i, polyData):
    """
    Not used. Check if can be removed
    """
    # obsolete
    profbox()
    modelNode = slicer.vtkMRMLModelNode()
    displayNode = slicer.vtkMRMLModelDisplayNode()
    storageNode = slicer.vtkMRMLModelStorageNode()

    fileName = self.option[i] + '.vtp'

    mrmlScene = slicer.mrmlScene
    modelNode.SetName(fileName)
    modelNode.SetAndObservePolyData(polyData)
    modelNode.SetAttribute("planned", "1")

    mrmlScene.SaveStateForUndo()
    modelNode.SetScene(mrmlScene)
    storageNode.SetScene(mrmlScene)
    storageNode.SetFileName(fileName)
    displayNode.SetScene(mrmlScene)
    displayNode.SetVisibility(1)
    mrmlScene.AddNode(storageNode)
    mrmlScene.AddNode(displayNode)
    mrmlScene.AddNode(modelNode)
    modelNode.SetAndObserveStorageNodeID(storageNode.GetID())
    modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

    displayNode.SetColor(self.color[i])
    displayNode.SetVisibility2D(0)
    pNode = self.parameterNode()
    pNode.SetParameter(fileName, modelNode.GetID())
    mrmlScene.AddNode(modelNode)
    displayNode.SetVisibility(1)

  def displayRadPlanned(self):
    """
    Display 'radiation' of planned needle -> cf iGyne / not used anymore
    """
    # obsolete?
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      displayNode = modelNode.GetDisplayNode()
      if modelNode.GetAttribute("radiation") == "planned":
        needleNode = slicer.mrmlScene.GetNodeByID(modelNode.GetAttribute("needleID"))
        if needleNode.GetDisplayVisibility() == 1:
          modelNode.SetDisplayVisibility(abs(int(slicer.modules.NeedleFinderWidget.displayRadPlannedButton.checked) - 1))

  def displayRadSegmented(self):
    """
    Display 'radiation' of segmented needles
    ??? used?
    """
    # obsolete?
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("radiation") == "segmented":
        needleNode = slicer.mrmlScene.GetNodeByID(modelNode.GetAttribute("needleID"))
        if needleNode != None:
          if needleNode.GetDisplayVisibility() == 1:
            modelNode.SetDisplayVisibility(abs(int(slicer.modules.NeedleFinderWidget.displayRadSegmentedButton.checked) - 1))
            d = modelNode.GetDisplayNode()
            d.SetVisibility2D(abs(int(slicer.modules.NeedleFinderWidget.displayRadSegmentedButton.checked) - 1))

  def displayContour(self, i, visibility):
    """
    Display the iso-contour of needle i

    :param i: nth value of needle
    :param visibility: boolean to set the visibility state of the needle
    ??? used?
    """
    # obsolete?
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("contour") == "1" and modelNode.GetAttribute("nth") == str(i) :
        needleNode = slicer.mrmlScene.GetNodeByID(modelNode.GetAttribute("needleID"))
        if needleNode != None:
          if needleNode.GetDisplayVisibility() == 1:
            modelNode.SetDisplayVisibility(visibility)
            d = modelNode.GetDisplayNode()
            d.SetVisibility2D(visibility)

  def displayContours(self):
    """
    Display or hide the iso-contours of every needles
    ??? used?
    """
    # obsolete?
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("contour") == "1":
        needleNode = slicer.mrmlScene.GetNodeByID(modelNode.GetAttribute("needleID"))
        if needleNode != None:
          if needleNode.GetDisplayVisibility() == 1:
            modelNode.SetDisplayVisibility(abs(int(slicer.modules.NeedleFinderWidget.displayContourButton.checked) - 1))
            d = modelNode.GetDisplayNode()
            d.SetVisibility2D(abs(int(slicer.modules.NeedleFinderWidget.displayContourButton.checked) - 1))

  def displayFiducial(self):
    """
    ??? used? show labels of the needles by adding a fiducial point at the tip
    """
    # obsolete?
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      displayNode = modelNode.GetDisplayNode()
      if modelNode.GetAttribute("segmented") == "1" and modelNode.GetAttribute("nth") != None:
        if 1:
          i = int(modelNode.GetAttribute("nth"))
          if self.fiducialnode[i] == 0:
            polyData = modelNode.GetPolyData()
            nb = int(polyData.GetNumberOfPoints() - 1)
            coord = [0, 0, 0]
            if nb > 10:
              self.fiducialnode[i] = slicer.vtkMRMLAnnotationFiducialNode()
              polyData.GetPoint(nb, coord)
              self.fiducialnode[i].SetName(self.option[i])
              self.fiducialnode[i].SetFiducialCoordinates(coord)
              self.fiducialnode[i].Initialize(slicer.mrmlScene)
              self.fiducialnode[i].SetLocked(1)
              self.fiducialnode[i].SetSelectable(0)
              fidDN = self.fiducialnode[i].GetDisplayNode()
              fidDN.SetColor(modelNode.GetDisplayNode().GetColor())
              fidDN.SetGlyphScale(0)
              fidTN = self.fiducialnode[i].GetAnnotationTextDisplayNode()
              fidTN.SetTextScale(3)
              fidTN.SetColor(modelNode.GetDisplayNode().GetColor())

              self.fiducialnode[i].SetDisplayVisibility(modelNode.GetDisplayNode().GetVisibility())
          else:
            if modelNode.GetDisplayNode().GetVisibility():
               self.fiducialnode[i].SetDisplayVisibility(abs(self.fiducialnode[i].GetDisplayVisibility() - 1))
            if self.fiducialnode[i].GetDisplayVisibility() == 1:
              self.displayFiducialButton.text = "Hide Labels on Needles"
            else:
              self.displayFiducialButton.text = "Display Labels on Needles"

  def resetCoronalSegment(self):
    """
    Reset coronal segment orientation.
    """
    #research
    profprint()
    sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeGreen")
    if sGreen == None :
      sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode3")
    reformatLogic = slicer.vtkSlicerReformatLogic()
    #sGreen.SetSliceVisible(0)
    sGreen.SetOrientationToCoronal()
    #sw = slicer.app.layoutManager().sliceWidget("Green")
    #sw.fitSliceToBackground()
    sGreen.Modified()

  def resetSagittalSegment(self):
    """
    Reset sagittal segment orientation.
    """
    #research
    profprint()
    sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow")
    if sYellow == None :
      sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")
    reformatLogic = slicer.vtkSlicerReformatLogic()
    sYellow.SetSliceVisible(0)
    sYellow.SetOrientationToSagittal()
    sw = slicer.app.layoutManager().sliceWidget("Yellow")
    sw.fitSliceToBackground()
    sYellow.Modified()

  def reformatCoronalView4NeedleSegment(self, base, tip, ID=-1):
    """
    Reformat the coronal view to be tangent to the needle.
    """
    #research
    profprint()
    for i in range(2):  # workaround update problem
      if ID >=0:
        modelNode = getNode('vtkMRMLModelNode' + str(ID))
        polyData = modelNode.GetPolyData()
        nb = polyData.GetNumberOfPoints()
        base = [0, 0, 0]
        tip = [0, 0, 0]
        polyData.GetPoint(nb - 1, tip)
        polyData.GetPoint(0, base)
      a, b, c = tip[0] - base[0], tip[1] - base[1], tip[2] - base[2]
    
      sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeGreen")
      if sGreen == None :
        sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode3")
      reformatLogic = slicer.vtkSlicerReformatLogic()
      #sGreen.SetSliceVisible(1)
      reformatLogic.SetSliceNormal(sGreen, 1, -a / b, 0)
      #reformatLogic.SetSliceOrigin(sGreen, base[0],base[1],base[2])#crashes
      m = sGreen.GetSliceToRAS()
      m.SetElement(0, 3, base[0])
      m.SetElement(1, 3, base[1])
      m.SetElement(2, 3, base[2])
      sGreen.Modified()

  def drawIsoSurfaces0(self):
    """
    ??? used? for development purposes.
    This shall indicate radiation influence zones.
    Ellipsoid at tip of the needel.
    """
    # research
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    v = vtk.vtkAppendPolyData()

    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("nth") != None and modelNode.GetDisplayVisibility() == 1 :
        v.AddInputData(modelNode.GetPolyData())

    modeller = vtk.vtkImplicitModeller()
    modeller.SetInputConnection(v.GetOutputPort())
    modeller.SetSampleDimensions(self.dim.value, self.dim.value, self.dim.value)
    modeller.SetCapping(0)
    modeller.SetAdjustBounds(self.abonds.value)
    modeller.SetProcessModeToPerVoxel()
    modeller.SetAdjustDistance(self.adist.value / 100)
    modeller.SetMaximumDistance(self.maxdist.value / 100)

    contourFilter = vtk.vtkContourFilter()
    contourFilter.SetNumberOfContours(self.nb.value)
    contourFilter.SetInputConnection(modeller.GetOutputPort())
    contourFilter.ComputeNormalsOn()
    contourFilter.ComputeScalarsOn()
    contourFilter.UseScalarTreeOn()
    contourFilter.SetValue(self.contour.value, self.contourValue.value)
    contourFilter.SetValue(self.contour2.value, self.contourValue2.value)
    contourFilter.SetValue(self.contour3.value, self.contourValue3.value)
    contourFilter.SetValue(self.contour4.value, self.contourValue4.value)
    contourFilter.SetValue(self.contour5.value, self.contourValue5.value)

    isoSurface = contourFilter.GetOutput()
    self.AddContour(isoSurface)

  #----------------------------------------------------------------------------------------------
  """
  End of the functions used with the Yi's CLI module
  """
  #----------------------------------------------------------------------------------------------
  #----------------------------------------------------------------------------------------------
  """
  Functions for validation study
  """
  #----------------------------------------------------------------------------------------------

  def returnTipsFromNeedleModels(self, type="Validation", offset=0, center=False):
    """ Returns the IJK coordinates of the tips of manually segmented needle polygon models
    :return: array of IJK coordinates of validation needle tips
    """
    # productive
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    returnTips = []
    names = []
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    nbNode = modelNodes.GetNumberOfItems()
    for nthNode in range(nbNode):
        # print nthNode
        node = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLModelNode')
        if node.GetAttribute('type') == type:
            polydata = node.GetPolyData()
            p, pOpp, pbis = [0, 0, 0], [0, 0, 0], [0, 0, 0]
            #if not polydata: breakbox("/!!!\ needle tube not found as polydata in scene/vtk file missing: "+widget.caseNr+" "+node.GetName())
            if polydata and polydata.GetNumberOfPoints() > 100:  # ??? this is risky when u have other models in the scene (not only neeedles(
                if not widget.autoStopTip.isChecked():
                  polydata.GetPoint(0+offset, p)
                  if center: 
                    polydata.GetPoint(0+offset+25, pOpp)
                    p = (np.array(p) + np.array(pOpp))/2
                  polydata.GetPoint(int(polydata.GetNumberOfPoints() - 1 - offset), pbis)
                  if center: 
                    polydata.GetPoint(0+offset+25, pOpp)
                    pbis = (np.array(pbis) + np.array(pOpp))/2
                  if pbis[2] > p[2]:
                      p = pbis
                else:
                  # get a point from the middle (low=/4, high=/4*3) of the needle shaft polygon model
                  polydata.GetPoint(int(polydata.GetNumberOfPoints() / 2)+ offset, p) #CONST
                  if center: 
                    polydata.GetPoint(int(polydata.GetNumberOfPoints() / 2)+offset+25, pOpp)
                    p = (np.array(p) + np.array(pOpp))/2
                returnTips.append(self.ras2ijk(p))
                names.append(node.GetName())
    return returnTips, names

  def returnBasesFromNeedleModels(self, type="Validation", offset=0):
    """ Returns the IJK coordinates of the base of manually segmented needle polygon models
    :return: array of IJK coordinates of validation needle bases
    """
    #research
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    returnBases = []
    names = []
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    nbNode = modelNodes.GetNumberOfItems()
    for nthNode in range(nbNode):
        # print nthNode
        node = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLModelNode')
        if node.GetAttribute('type') == type:
            polydata = node.GetPolyData()
            p, pbis = [0, 0, 0], [0, 0, 0]
            #if not polydata: breakbox("/!!!\ needle tube not found as polydata in scene: "+widget.caseNr+" "+node.GetName())
            if polydata and polydata.GetNumberOfPoints() > 100:  # ??? this is risky when u have other models in the scene (not only neeedles(
                if not widget.autoStopTip.isChecked():
                  polydata.GetPoint(0+offset, p)
                  polydata.GetPoint(int(polydata.GetNumberOfPoints() - 1 - offset), pbis)
                  if pbis[2] < p[2]:
                      p = pbis
                returnBases.append(self.ras2ijk(p))
                names.append(node.GetName())
    return returnBases, names
  
  def returnMidsFromNeedleModels(self, type="Validation", offset=0):
    """ Returns the IJK coordinates of the middles of manually segmented needle polygon models
    :return: array of IJK coordinates of validation needle tips
    """
    # productive
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    returnTips = []
    names = []
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    nbNode = modelNodes.GetNumberOfItems()
    for nthNode in range(nbNode):
        # print nthNode
        node = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLModelNode')
        if node.GetAttribute('type') == type:
            polydata = node.GetPolyData()
            pmid=[0, 0, 0]
            #if not polydata: breakbox("/!!!\ needle tube not found as polydata in scene/vtk file missing: "+widget.caseNr+" "+node.GetName())
            if polydata and polydata.GetNumberOfPoints() > 100:  # ??? this is risky when u have other models in the scene (not only neeedles(
              polydata.GetPoint(int((polydata.GetNumberOfPoints() - 1)/2)+offset, pmid)
              returnTips.append(self.ras2ijk(pmid))
              names.append(node.GetName())
    return returnTips, names

  def startValidation(self, script=False, offset=0, needleNr=None):
    """Start the evaluation process:
    * Calls returnTipsFromNeedleModels() to build an array of the tip of manually segmented needles
    * Use theses tips to generate auto segmented needles
    * Calls evaluate() to compute the Hausdorff's distance between every pair of needles

    :return: print the results in the python interactor (CMD+3 or CTRL+3)
    """
    # productive #button
    print("\n"*10)
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    if needleNr==None: self.deleteNeedleDetectionModelsFromScene()
    tips, names = self.returnTipsFromNeedleModels(offset=offset, center=not script)
    if needleNr!=None:
      needleName='manual-seg_'+str(needleNr)
      for tip,name in zip(tips,names):
        if name!=needleName:
          try:
            names.remove(name)
            tips.remove(tip)
          except:
            print("not found: ", needleName) 
    # delete old needles as they will be recalculated
    self.deleteAllAutoNeedlesFromScene()
    # select the image node from the Red slice viewer
    m = vtk.vtkMatrix4x4()
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    volumeNode.GetIJKToRASMatrix(m)
    imageData = volumeNode.GetImageData()
    spacing = volumeNode.GetSpacing()
    # chrono starts
    self.t0 = time.perf_counter()
    print('spacing: ', spacing)
    self.needleDetectionThread(tips, imageData, spacing=spacing, script=script, names=names) # <<< Ruibin
          
    ''' old due to Ruibins code:
    for i in range(len(tips)):
      A = tips[i]
      name = names[i]
      colorVar = i  # ??? /(len(tips))
      self.needleDetectionThread(A, imageData, colorVar, spacing, script=script,strName=name)
      if widget.autoStopTip.isChecked():
        self.needleDetectionUPThread(A, imageData, colorVar, spacing, script=script,strName=name,tipOnly=False)
    '''
    # print tips
    nOutliers=0
    if script == False:
        t = self.evaluate(needleNr=needleNr)
        print('----------Alg.#%d---------'%widget.algoVersParameter.value)
        print('New HD Validation Results:')
        print('i\tman.-seg_\ttipHD [mm]\tHD [mm]')
        for i in range(len(t)):
          try: strI=str(int(t[i][2]))
          except: strI=str(t[i][2])
          print(i,'\t', strI,'\t',t[i][0],'\t',t[i][1])
          if t[i][1]>outlierThresh_mm: nOutliers+=1
        print('nOutliers_'+str(outlierThresh_mm)+'mm=',nOutliers)
        print('==========================')
    if not script: msgbox('Validation ready!')

  def placeAxialLimitMarker(self, assign=True):
    """
    Get the K (of IJK) value of the current axial slice.
    Used to define the slice containing the template
    """
    # productive #onButton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    if assign:
      # remove old nodes from scene
      while slicer.util.getNodes('template slice position*') != {}:
        nodes = slicer.util.getNodes('template slice position*')
        for node in list(nodes.values()):
          slicer.mrmlScene.RemoveNode(node)

      s = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetSliceNode()
      offSet = s.GetSliceOffset()
      rasVector = [0, 0, offSet]
      widget.templateSliceButton.text = "1. Select Current Axial Slice as Seg. Limit (current: " + str(offSet) + ")"

      widget.axialSegmentationLimit = int(round(self.ras2ijk(rasVector)[2]))

      self.fiducialNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
      self.fiducialNode.Initialize(slicer.mrmlScene)
      self.fiducialNode.SetName('template slice position')
      self.fiducialNode.SetFiducialCoordinates(rasVector)
      fd = self.fiducialNode.GetDisplayNode()
      fd.SetVisibility(1)
      fd.SetColor([0, 1, 0])
    widget.fiducialButton.setEnabled(1)
    # here we are set and can create a first label volume from volume data
    widget.createAddOrSelectLabelMapNode()

    # ## Add crosshair
    crosshairNode = getNode("Crosshair")
    crosshairNode.SetCrosshairMode(slicer.vtkMRMLCrosshairNode().ShowIntersection)

  def exportEvaluation(self, results, url):
    """ Export evaluation results to a CSV file (e.g. when doing parameter optimizations)

    :param results: array containing the results of the evaluation
    :param url: url for saving the file
    """
    # research
    profprint()
    if not os.path.exists(url):
      print("creating new results file: ",url)
      open(url, 'w').close()
    myfile = open(url, 'a')

    wr = csv.writer(myfile)
    r = numpy.array(results)
    if len(r.shape) == 1:
      wr.writerow(results)
    else:
      wr.writerows(results)

  def hausdorffDistance(self, id1, id2):
    r"""
    Calculates the Hausdorff's distance [HD]_ of two needles. Both needles are truncated to start and end at the same slices.

    :param id1: vtkMRMLModelNodeID of Needle 1
    :param id2: vtkMRMLModelNodeID of Needle 2
    :return: Hausdorff's distance in millimeters

    .. math::  d_{\mathrm H}(X,Y) = \max\{\,\sup_{x \in X} \inf_{y \in Y} d(x,y),\, \sup_{y \in Y} \inf_{x \in X} d(x,y)\,\}

    .. image:: http://upload.wikimedia.org/wikipedia/commons/2/21/Hausdorff_distance_sample.svg
           :height: 300px
           :width: 500 px
           :scale: 100 %
           :alt: alternate text
           :align: center

    .. [HD] http://en.wikipedia.org/wiki/Hausdorff_distance
    """
    # productive #math
    if frequent: profprint()
    node1 = slicer.mrmlScene.GetNodeByID(id1)
    polydata1 = node1.GetPolyData()
    node2 = slicer.mrmlScene.GetNodeByID(id2)
    polydata2 = node2.GetPolyData()
    nb1 = polydata1.GetNumberOfPoints()
    nb2 = polydata2.GetNumberOfPoints()
    minimum = None
    maximum = None
    JJ, jj = None, None
    II, ii = None, None
    pt1 = [0, 0, 0]
    pt2 = [0, 0, 0]
    polydata1.GetPoint(1, pt1)
    polydata1.GetPoint(nb1 - 1, pt2)
    minVal1 = min(pt1[2], pt2[2])
    maxVal1 = max(pt1[2], pt2[2])
    pt1 = [0, 0, 0]
    pt2 = [0, 0, 0]
    pt1b, pt2b = None, None
    polydata2.GetPoint(1, pt1)
    polydata2.GetPoint(nb2 - 1, pt2)
    minVal2 = min(pt1[2], pt2[2])
    maxVal2 = max(pt1[2], pt2[2])
    valueBase = max(minVal1, minVal2)
    valueTip = min(maxVal1, maxVal2)

    # truncate polydatas
    truncatedPolydata1 = self.clipPolyData(node1, valueBase)
    truncatedPolydata2 = self.clipPolyData(node2, valueBase)

    cellId = vtk.mutable(1)
    subid = vtk.mutable(1)
    dist = vtk.mutable(1)
    cl2 = vtk.vtkCellLocator()
    cl2.SetDataSet(truncatedPolydata2)
    cl2.BuildLocator()
    # Hausforff 1 -> 2
    minima = []
    for i in range(int(nb1 / float(10))):
      pt = [0, 0, 0]
      polydata1.GetPoint(10 * i, pt)
      closest = [0, 0, 0]
      cl2.FindClosestPoint(pt, closest, cellId, subid, dist)
      if abs(closest[2] - pt[2]) <= 1:
        minima.append(self.distance(pt, closest))
      else:
          minima.append(0)
    hausdorff12 = max(minima)

    # Hausforff 2 -> 1
    minima = []
    cl1 = vtk.vtkCellLocator()
    cl1.SetDataSet(truncatedPolydata1)
    cl1.BuildLocator()
    for i in range(int(nb2 / float(10))):
      pt = [0, 0, 0]
      polydata2.GetPoint(10 * i, pt)
      closest = [0, 0, 0]
      cl1.FindClosestPoint(pt, closest, cellId, subid, dist)
      if abs(closest[2] - pt[2]) <= 1:
        minima.append(self.distance(pt, closest))
      else:
          minima.append(0)
    hausdorff21 = max(minima)
    return max(hausdorff12, hausdorff21)

  def hausdorffDistance13(self, id1, id2):
    """MICCAI13 Version
    iGyne_old 4450bbcb543e7432122f06c1905aab4eb8b446e6
    """
    # productive #math
    if frequent: profprint()
    node1 = slicer.mrmlScene.GetNodeByID(id1)
    polydata1 = node1.GetPolyData()
    node2 = slicer.mrmlScene.GetNodeByID(id2)
    polydata2 = node2.GetPolyData()
    nb1 = polydata1.GetNumberOfPoints()
    nb2 = polydata2.GetNumberOfPoints()
    minimum = None
    maximum = None
    JJ, jj = None, None
    II, ii = None, None
    pt1 = [0, 0, 0]
    pt2 = [0, 0, 0]
    polydata1.GetPoint(1, pt1)
    polydata1.GetPoint(nb1 - 1, pt2)
    minVal1 = min(pt1[2], pt2[2])
    maxVal1 = max(pt1[2], pt2[2])
    pt1 = [0, 0, 0]
    pt2 = [0, 0, 0]
    pt1b, pt2b = None, None
    polydata2.GetPoint(1, pt1)
    polydata2.GetPoint(nb2 - 1, pt2)
    minVal2 = min(pt1[2], pt2[2])
    maxVal2 = max(pt1[2], pt2[2])
    valueBase = max(minVal1, minVal2)
    valueTip = min(maxVal1, maxVal2)
    cellId = vtk.mutable(1)
    subid = vtk.mutable(1)
    dist = vtk.mutable(1)
    cl2 = vtk.vtkCellLocator()
    cl2.SetDataSet(polydata2)
    cl2.BuildLocator()
    # Hausforff 1 -> 2
    minima = []
    for i in range(int(nb1 / float(100))):
      pt = [0, 0, 0]
      polydata1.GetPoint(100 * i, pt)
      closest = [0, 0, 0]
      cl2.FindClosestPoint(pt, closest, cellId, subid, dist)
      if abs(closest[2] - pt[2]) <= 1:
        minima.append(self.distance(pt, closest))
      else:
          minima.append(0)
    hausdorff12 = max(minima)

    # Hausforff 2 -> 1
    minima = []
    cl1 = vtk.vtkCellLocator()
    cl1.SetDataSet(polydata1)
    cl1.BuildLocator()
    for i in range(int(nb2 / float(10))):
      pt = [0, 0, 0]
      polydata2.GetPoint(10 * i, pt)
      closest = [0, 0, 0]
      cl1.FindClosestPoint(pt, closest, cellId, subid, dist)
      if abs(closest[2] - pt[2]) <= 1:
        minima.append(self.distance(pt, closest))
      else:
          minima.append(0)
    hausdorff21 = max(minima)
    return max(hausdorff12, hausdorff21)

  def evaluate(self, script=False, needleNr=None):
    """
    This function first invokes needleMatching() with, for each automatically segmented needle in the vtkMRMLScene,
    associates it with its manually segmented version.
    Then, the HD is computed between each pairs of needle and the results are reported in a numpy array

    :return numpy array of [ value , ID needle 1, ID needle 2 ]
    """
    # productive
    profprint()
    result = self.needleMatching()
    HD = []
    outliers=[]
    widget = slicer.modules.NeedleFinderWidget
    self.valuesExperience = [ widget.radiusNeedleParameter.value,
                            widget.lenghtNeedleParameter.value,
                            widget.radiusMax.value,
                            widget.numberOfPointsPerNeedle.value,
                            widget.nbRotatingIterations.value,
                            widget.stepsize.value,
                            widget.gradientPonderation.value,
                            widget.exponent.value,
                            widget.gaussianAttenuationButton.isChecked() * 1,
                            widget.sigmaValue.value,
                            widget.algoVersParameter.value]

    for i in range(len(result)):
      if widget.algoVersParameter.value == 0:
        val = self.hausdorffDistance(result[i][1], result[i][2])
      else:
        val = self.hausdorffDistance13(result[i][1], result[i][2])
      try: needleNrFromFilename=int(result[i][3].strip('manual-seg_'))
      except: needleNrFromFilename=-1
      if needleNr!=None and needleNr!=needleNrFromFilename:
        continue
      if script == True:
        results = [result[i][0], float(val), int(needleNrFromFilename), int(result[i][1].strip('vtkMRMLModelNode')), int(result[i][2].strip('vtkMRMLModelNode'))]+[val>outlierThresh_mm] + self.valuesExperience
      else:
        results = [result[i][0], float(val), int(needleNrFromFilename), int(result[i][1].strip('vtkMRMLModelNode')), int(result[i][2].strip('vtkMRMLModelNode'))]+[val>outlierThresh_mm]
      HD.append(results)
      if val>outlierThresh_mm: outliers.append(needleNrFromFilename) #CONST
    if script == False:
      return numpy.array(HD).astype(numpy.double)
    else:
      return HD, outliers

  def distTip(self, id1, id2):
    """ Returns the distance between the tip of two needles

    :param id1: ID number for the needle 1 (vtkMRMLModelNode)
    :param id2: ID number for the needle 2 (vtkMRMLModelNode)
    :return: distance in millimeters between the tip of both needles
    """
    # productive #math
    if frequent: profprint()
    node = slicer.mrmlScene.GetNodeByID('vtkMRMLModelNode' + str(id1))
    polydata = node.GetPolyData()
    node2 = slicer.mrmlScene.GetNodeByID('vtkMRMLModelNode' + str(id2))
    polydata2 = node2.GetPolyData()
    p, pbis = [0, 0, 0], [0, 0, 0]
    p2 = [0, 0, 0]
    p2bis = [0, 0, 0]
    tipDistance = []
    for i in range(100):
        polydata.GetPoint(i, p)
        polydata.GetPoint(polydata.GetNumberOfPoints()-1 - i, pbis)
        if pbis[2] > p[2]:
          p = pbis
        polydata2.GetPoint(i, p2)
        polydata2.GetPoint(polydata2.GetNumberOfPoints()-1 - i, p2bis)
        if p2bis[2] > p2[2]:
          p2 = p2bis
        tipDistance.append(((p2[0] - p[0]) ** 2 + (p2[1] - p[1]) ** 2 + (p2[2] - p[2]) ** 2) ** 0.5)
    return min(tipDistance)
  
  def distBase(self, id1, id2):
    """ Returns the distance between the base of two needles

    :param id1: ID number for the needle 1 (vtkMRMLModelNode)
    :param id2: ID number for the needle 2 (vtkMRMLModelNode)
    :return: distance in millimeters between the tip of both needles
    """
    # productive #math
    if frequent: profprint()
    node = slicer.mrmlScene.GetNodeByID('vtkMRMLModelNode' + str(id1))
    polydata = node.GetPolyData()
    node2 = slicer.mrmlScene.GetNodeByID('vtkMRMLModelNode' + str(id2))
    polydata2 = node2.GetPolyData()
    p, pbis = [0, 0, 0], [0, 0, 0]
    p2 = [0, 0, 0]
    p2bis = [0, 0, 0]
    baseDistance = []
    for i in range(100):
        polydata.GetPoint(i, p)
        polydata.GetPoint(polydata.GetNumberOfPoints()-1 - i, pbis)
        if pbis[2] > p[2]:
          pbis = p
        polydata2.GetPoint(i, p2)
        polydata2.GetPoint(polydata2.GetNumberOfPoints()-1 - i, p2bis)
        if p2bis[2] > p2[2]:
          p2bis = p2
        baseDistance.append(((p2bis[0] - pbis[0]) ** 2 + (p2bis[1] - pbis[1]) ** 2 + (p2bis[2] - pbis[2]) ** 2) ** 0.5)
    return min(baseDistance)

  def needleMatching(self):
    """This functions associates manually segmented needles to their automatically segmented version. To do so,
    each manually segmented needle has a 'type' attribute named 'validation'. For each manually segmented needle, an axial
    distance between the tip of this needle, and the tip of every auto segmented needle is computed. The auto needle that
    offers the minimal tip axial distance is considered as corresponding needle

    :return: array of tuple of corresponding needles
    """
    # productive
    profprint()
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    nbNode = modelNodes.GetNumberOfItems()
    result = []
    found = []
    # print nbNode
    for nthNode in range(nbNode):
      node = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLModelNode')
      if node.GetID() not in found and node.GetAttribute('type') != 'Validation':
        dist = []
        polydata = node.GetPolyData()
        if polydata != None:
          bounds = polydata.GetBounds()
          for nthNode2 in range(nbNode):
            node2 = slicer.mrmlScene.GetNthNodeByClass(nthNode2, 'vtkMRMLModelNode')
            if node2.GetID() not in found and node2.GetAttribute('type') == 'Validation':
              polydata2 = node2.GetPolyData()
              if polydata2 != None and polydata2.GetNumberOfPoints() > 100 and polydata.GetNumberOfPoints() > 100:
                tipDistance = self.distTip(int(node.GetID().strip('vtkMRMLModelNode')) , int(node2.GetID().strip('vtkMRMLModelNode')))
                baseDistance = self.distBase(int(node.GetID().strip('vtkMRMLModelNode')) , int(node2.GetID().strip('vtkMRMLModelNode')))
                name = node.GetName()
                manualName = name.lstrip('auto-seg_').lstrip('manual-seg_').lstrip('obturator-seg_').lstrip('0123456789').lstrip('-ID-vtkMRMLModelNode').lstrip('0123456789-')
                if manualName==node2.GetName(): dist.append([tipDistance, node2.GetID(), node2.GetName()])
                # print tipDistance
          if dist != []:
            match = [min(dist)[0], min(dist)[1], node.GetID(), min(dist)[2]]
            result.append(match)
            found.append(min(dist)[1])
            found.append(node.GetID())
            node.GetDisplayNode().SetVisibility2D(1)
    # print result
    return result

  def setAllNeedleTubesAsValidationNeedles(self):
    """
    This is used for testing, R&D. It sets the needles to validation type.
    E.g. you load vtk needle models and want to use them as validation needles...
    """
    # #test #research #button
    profprint()
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    nbNode = modelNodes.GetNumberOfItems()
    for nthNode in range(nbNode):
      node = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLModelNode')
      if node.GetID() and node.GetAttribute('type') != 'Validation':
        node.SetAttribute('type', 'Validation')
        displayNode = node.GetDisplayNode()
        colorVar = random.randrange(50, 100, 1)  # ??? /(100.)
        nth = int(colorVar) % MAXCOL
        displayNode.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
        displayNode.SetVisibility2D(True)
        displayNode.SetSliceIntersectionThickness(2)
        #displayNode.SetOpacity(0.7)

  def distance(self, pt1, pt2):
    """3D distance between two points

    :param pt1: point 1 [x,y,z]
    :param pt2: point 2 [x,y,z
    :return: Euclidian's distance
    """
    # productive #frequent
    if frequent: profprint()
    d = ((float(pt1[0]) - float(pt2[0])) ** 2 + (float(pt1[1]) - float(pt2[1])) ** 2 + (float(pt1[2]) - float(pt2[2])) ** 2) ** 0.5
    return d

  def addManualTip(self, A):
    """Add fiducial node to the scene (called by processEventAddManualTips). Used as starting point for needle segmentation

    :param A: RAS coordinates
    """
    # obsolete?
    profbox()
    self.fiducialNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
    self.fiducialNode.Initialize(slicer.mrmlScene)
    self.fiducialNode.SetName('tip')
    self.fiducialNode.SetFiducialCoordinates(A)
    fd = self.fiducialNode.GetDisplayNode()
    fd.SetVisibility(1)
    fd.SetColor([0, 1, 0])

  def distanceTwoPoints(self, A, B):
    """3D distance between two points

    :param pt1: point 1 [x,y,z]
    :param pt2: point 2 [x,y,z
    :return: Euclidian's distance
    """
    # productive
    # used by addNeedleToScene
    if frequent: profprint()
    length = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2) ** 0.5
    return length

  def clipPolyData(self, node, value, visible=0):
    """ Function used to truncate needle models to evaluate the Hausdorff's distance.

    :param node: vtkMRMLModelNode
    :param value: axial slice where to cut the model
    :return: vtkPolyData of the cutted model
    """
    # productive
    profprint()
    # We clip with an implicit function. Here we use a plane positioned near
    # the center of the cow model and oriented at an arbitrary angle.
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, 0)
    plane.SetNormal(0, 0, 1)
    # vtkClipPolyData requires an implicit function to define what it is to
    # clip with. Any implicit function, including complex boolean combinations
    # can be used. Notice that we can specify the value of the implicit function
    # with the SetValue method.
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(node.GetPolyData())
    clipper.SetClipFunction(plane)
    clipper.GenerateClipScalarsOn()
    clipper.GenerateClippedOutputOn()
    clipper.SetValue(value)
    clipper.Update()
    polyData = clipper.GetOutput()
    if 1 == 1:
        scene = slicer.mrmlScene
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
    if visible != 1:
        scene.RemoveNode(model)

    return polyData

  def setWL(self, dn, w, l):
    """
    Set window/level in mpr slice view
    """
    # productive #frequent #onDrag
    if frequent: profprint();
    dn.SetWindow(w)
    dn.SetLevel(l)

  def setColors255(self):
    """Set color map with colors encoded in RGB between 0 and 255
    """
    # productive
    profprint()
    self.color255 = [list(c) for c in COLORS_255]
    return self.color255

  def setColors(self):
    """Sets color map with colors encoded in RGB between 0 and 1
    """
    # productive
    profprint()
    self.color255 = self.setColors255()
    self.color = [list(c) for c in COLORS_RGB]
    return self.color

  def setHolesCoordinates(self):
    """Coordinates of the 63 holes in the obturator
    """
    # productive
    profprint()
    self.p = [list(row) for row in HOLE_COORDINATES]
    return self.p

  def setLabels(self):
    """
    Set the list of labels corresponding of the 63 holes of the obturator
    """
    # productive
    profprint()
    self.option = dict(HOLE_LABELS)
    return self.option

  def setParameterNode(self, parameterNode):
    """Set Parameter Node (to save parameters from one step to the other)
    ??? used?
    """
    # framework
    profbox()
    self.parameterNode = parameterNode

  def parameterNode(self):
    """Returns parameter node
    ??? used?
    """
    # framework
    profbox()
    return self.parameterNode

  def getBoldFont(self):
    """Get a Qt bold font
    ??? Used?
    """
    # obsolete?
    profbox()
    boldFont = qt.QFont("Sans Serif", 12, qt.QFont.Bold)
    return boldFont

  def saveParameters (self , filePath):
    """
    save the current needle detection parameters to file
    """
    # productive #onButton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    config = configparser.RawConfigParser()
    config.add_section('NeedleFinder Parameters')
    config.add_section('BooleanSection')
    config.add_section('IntegerSection')

    config.set('BooleanSection', 'autoCorrectTip', widget.autoCorrectTip.isChecked())
    config.set('BooleanSection', 'invertedContrast', widget.invertedContrast.isChecked())
    config.set('BooleanSection', 'gradient', widget.gradient.isChecked())
    config.set('BooleanSection', 'filterControlPoints', widget.filterControlPoints.isChecked())
    config.set('BooleanSection', 'drawFiducialPoints', widget.drawFiducialPoints.isChecked())
    config.set('BooleanSection', 'autoStopTip', widget.autoStopTip.isChecked())
    config.set('BooleanSection', 'extendNeedle', widget.extendNeedle.isChecked())
    config.set('BooleanSection', 'maxLength', widget.maxLength.isChecked())
    config.set('BooleanSection', 'gaussianAttenuationButton', widget.gaussianAttenuationButton.isChecked())

    config.set('IntegerSection', 'realNeedleLength', widget.realNeedleLength.value)
    config.set('IntegerSection', 'sigmaValue', widget.sigmaValue.value)
    config.set('IntegerSection', 'gradientPonderation', widget.gradientPonderation.value)
    config.set('IntegerSection', 'exponent', widget.exponent.value)
    config.set('IntegerSection', 'distanceMax', widget.radiusMax.value)
    config.set('IntegerSection', 'nbRotatingIterations', widget.nbRotatingIterations.value)
    config.set('IntegerSection', 'numberOfPointsPerNeedle', widget.numberOfPointsPerNeedle.value)
    config.set('IntegerSection', 'lenghtNeedleParameter', widget.lenghtNeedleParameter.value)
    config.set('IntegerSection', 'radiusNeedleParameter', widget.radiusNeedleParameter.value)
    config.set('IntegerSection', 'algoVersParameter', widget.algoVersParameter.value)

    # Writing our configuration file to 'example.cfg'
    with open(filePath, 'w') as configfile:
        config.write(configfile)

  def loadParameters (self, filePath):
    """
    load the parameters from parameter text file
    """
    # productive #onButton
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    config = configparser.RawConfigParser()
    config.read(filePath)

    autoCorrectTip = config.getboolean('BooleanSection', 'autoCorrectTip')
    invertedContrast = config.getboolean('BooleanSection', 'invertedContrast')
    gradient = config.getboolean('BooleanSection', 'gradient')
    filterControlPoints = config.getboolean('BooleanSection', 'filterControlPoints')
    drawFiducialPoints = config.getboolean('BooleanSection', 'drawFiducialPoints')
    autoStopTip = config.getboolean('BooleanSection', 'autoStopTip')
    extendNeedle = config.getboolean('BooleanSection', 'extendNeedle')
    maxLength = config.getboolean('BooleanSection', 'maxLength')
    gaussianAttenuationButton = config.getboolean('BooleanSection', 'gaussianAttenuationButton')

    realNeedleLength = config.getint('IntegerSection', 'realNeedleLength')
    sigmaValue = config.getint('IntegerSection', 'sigmaValue')
    gradientPonderation = config.getint('IntegerSection', 'gradientPonderation')
    exponent = config.getint('IntegerSection', 'exponent')
    try:
      radiusMax = config.getint('IntegerSection', 'distanceMax') # try deprecated parameter name (old parameter files)
    except:
      radiusMax = config.getint('IntegerSection', 'radiusMax')
    nbRotatingIterations = config.getint('IntegerSection', 'nbRotatingIterations')
    numberOfPointsPerNeedle = config.getint('IntegerSection', 'numberOfPointsPerNeedle')
    lenghtNeedleParameter = config.getint('IntegerSection', 'lenghtNeedleParameter')
    radiusNeedleParameter = config.getint('IntegerSection', 'radiusNeedleParameter')
    algoVersParameter = config.getint('IntegerSection', 'algoVersParameter')

    widget.autoCorrectTip.checked = autoCorrectTip
    widget.invertedContrast.checked = invertedContrast
    widget.gradient.checked = gradient
    widget.filterControlPoints.checked = filterControlPoints
    widget.drawFiducialPoints.checked = drawFiducialPoints
    widget.autoStopTip.checked = autoStopTip
    widget.extendNeedle.checked = extendNeedle
    widget.maxLength.checked = maxLength
    widget.gaussianAttenuationButton.checked = gaussianAttenuationButton

    widget.realNeedleLength.value = realNeedleLength
    widget.sigmaValue.value = sigmaValue
    widget.gradientPonderation.value = gradientPonderation
    widget.exponent.value = exponent
    widget.radiusMax.value = radiusMax
    widget.nbRotatingIterations.value = nbRotatingIterations
    widget.numberOfPointsPerNeedle.value = numberOfPointsPerNeedle
    widget.lenghtNeedleParameter.value = lenghtNeedleParameter
    widget.radiusNeedleParameter.value = radiusNeedleParameter
    widget.algoVersParameter.value = algoVersParameter
    print("#############")
    print("algoVers: ", algoVersParameter)
    print("Parameters successfully loaded!")
