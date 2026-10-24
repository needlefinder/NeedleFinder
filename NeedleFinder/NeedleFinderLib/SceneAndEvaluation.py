"""Scene management, UI table synchronization, parameter persistence, and Hausdorff evaluation."""

import configparser
import csv
import os
import random
import shutil
import time
import numpy as np
import SimpleITK as sitk
import sitkUtils
import vtk
import qt
import slicer

from .Constants import COLORS_255, COLORS_RGB, HOLE_COORDINATES, HOLE_LABELS
from .GeometryUtils import (
    clip_polydata_axial,
    compute_base_distance,
    compute_hausdorff_distance,
    compute_tip_distance,
    distance_3d,
    extract_model_id_number,
)
from .LegacyEditorCompat import (
    create_fiducial_node,
    get_fiducial_coords,
    set_all_fiducial_text_scale,
)
from .ProfilingUtils import (
    MAXCOL,
    MAXNEEDLES,
    breakbox,
    frequent,
    getNode,
    msgbox,
    outlierThresh_mm,
    profbox,
    profprint,
)


class SceneAndEvaluationMixin:
  """Mixin providing scene cleanup, report table management, Hausdorff distance evaluation, and config I/O."""

  def deleteTempModels(self):
    """
    Delete control points from needle curves etc.
    """
    while slicer.util.getNodes('.*') != {}:
      nodes = slicer.util.getNodes('.*')
      for key, value in zip(list(nodes.keys()), list(nodes.values())):
        if key != ".temp":
          slicer.mrmlScene.RemoveNode(value)
    while slicer.util.getNodes('_*') != {}:
      nodes = slicer.util.getNodes('_*')
      for key, value in zip(list(nodes.keys()), list(nodes.values())):
        slicer.mrmlScene.RemoveNode(value)

  def deleteAllAutoNeedlesFromScene(self):
    """
    Delete every segmented needle of the current set
    """
    profprint()
    while slicer.util.getNodes('auto-seg_' + str(self.round) + '*') != {}:
      nodes = slicer.util.getNodes('auto-seg_' + str(self.round) + '*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)

  def deleteAllModelsFromScene(self):
    """
    Delete every segmented needle, template slice markers and reset yellow segment
    """
    self.deleteNeedleDetectionModelsFromScene()
    self.deleteNeedleValidationModelsFromScene()

  def deleteNeedleValidationModelsFromScene(self):
    """
    Delete artifacts from validation from scene.
    """
    profprint()
    while slicer.util.getNodes('manual-seg_*') != {}:
      nodes = slicer.util.getNodes('manual-seg_*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)
    while slicer.util.getNodes('obturator-seg_*') != {}:
      nodes = slicer.util.getNodes('obturator-seg_*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)
    self.deleteTempModels()
    while slicer.util.getNodes('template slice position*') != {}:
      nodes = slicer.util.getNodes('template slice position*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)

  def deleteNeedleDetectionModelsFromScene(self):
    """
    Delete artifacts from semi-auto segmentation from scene.
    """
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    if widget.currentLabel:
      for i in range(widget.currentLabel + 1):
        while slicer.util.getNodes(str(i)) != {}:
          nodes = slicer.util.getNodes(str(i))
          for node in list(nodes.values()):
            slicer.mrmlScene.RemoveNode(node)
    while slicer.util.getNodes('auto-seg_*') != {}:
      nodes = slicer.util.getNodes('auto-seg_*')
      for node in list(nodes.values()):
        slicer.mrmlScene.RemoveNode(node)
    sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")
    sYellow.SetSliceVisible(0)
    reformatLogic = slicer.vtkSlicerReformatLogic()
    reformatLogic.SetSliceNormal(sYellow, 1, 0, 0)
    tempFidNodes = slicer.mrmlScene.GetNodesByName('.temp')
    for i in range(tempFidNodes.GetNumberOfItems()):
      node = tempFidNodes.GetItemAsObject(i)
      if node:
        slicer.mrmlScene.RemoveNode(node)
    self.deleteTempModels()
    sYellow.Modified()

  def deleteLastNeedle(self):
    """
    Delete last segmented needle of the current round
    """
    profprint(self.getName())
    widget = slicer.modules.NeedleFinderWidget
    if widget.deleteNeedleButton.isEnabled() and self.lastNeedleNames:
      name = self.lastNeedleNames.pop()
      print("removing needle with name: ", name)
      while slicer.util.getNodes(name + '*') != {}:
        nodes = slicer.util.getNodes(name + '*')
        for node in list(nodes.values()):
          slicer.mrmlScene.RemoveNode(node)
      model_id = extract_model_id_number(name)
      print("needle ID: <%s>" % model_id)
      if model_id is not None:
        self.deleteNeedleFromTable(int(model_id))

  def newInsertionNeedleSet(self):
    """
    Start a new round
    """
    profbox()
    widget = slicer.modules.NeedleFinderWidget
    if widget.newInsertionButton:
      dialog = qt.QDialog()
      qt.QMessageBox.information(dialog, 'Information', 'You are creating a new set of needles')
      self.round += 1
      widget.newInsertionButton.setText('Start a new set of needles - Round ' + str(self.round + 1) + '?')
      widget.deleteNeedleButton.setText('Delete Needles from round ' + str(self.round))

  def resetNeedleDetection(self, script=False):
    """
    Reset the needle detection to completely start over.
    """
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    ret = 0
    if not script:
      dialog = qt.QDialog()
      ret = qt.QMessageBox.question(dialog, 'Attention', """
          Are you sure that you want to reset the needle detection?
          It will delete every segmented needles...
          """, qt.QMessageBox.Ok, qt.QMessageBox.Cancel)
    if ret == qt.QMessageBox.Ok or script:
      self.deleteNeedleDetectionModelsFromScene()
      self.lastDetectionDecisions = None
      if hasattr(widget, "populateRejectedCandidatesTable"):
        try:
          widget.populateRejectedCandidatesTable([])
        except Exception:
          pass
      self.previousValues = [[0, 0, 0]]
      self.round = 1
      if widget.newInsertionButton:
        widget.newInsertionButton.setText('Start a new set of needles (' + str(self.round + 1) + ') ?')
        widget.deleteNeedleButton.setText('Delete Needles from set (' + str(self.round) + ")")
      widget.table = None
      widget.row = 0
      widget.col = 0
      if not script:
        for _ in widget.items:
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

      widget.fiducialButton.checked = 0
      widget.stop()
      widget.fiducialButton.text = "2. Start Giving Needle Tips"

      widget.templateSliceButton.setEnabled(1)
      widget.fiducialButton.setEnabled(0)
      widget.deleteNeedleButton.setEnabled(0)
      widget.resetDetectionButton.setEnabled(0)
      widget.labelMapNode = None
      widget.tempPointList = []
      widget.axialSegmentationLimit = None
      widget.axialSegmentationLimitRAS = None
      widget.templateSliceButton.text = "1. Select Current Axial Slice as Seg. Limit (current: None)"
      if not script:
        widget.onResetParameters()

  def hideAnnotations(self):
    """
    This function shows/hides all text annotation in the scene.
    """
    widget = slicer.modules.NeedleFinderWidget
    set_all_fiducial_text_scale(0.0 if widget.hideAnnotationTextButton.checked else 3.0)

  def resetNeedleValidation(self):
    """
    Reset the needle validation to completely start over.
    """
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    dialog = qt.QDialog()
    ret = qt.QMessageBox.question(dialog, 'Attention', """
      Are you sure that you want to reset the needle validation?
      It will delete every segmented needles and the control points...
      """, qt.QMessageBox.Ok, qt.QMessageBox.Cancel)
    if ret == qt.QMessageBox.Ok:
      self.deleteNeedleValidationModelsFromScene()
      widget.editNeedleTxtBox.value = 1
      widget.stepNeedle = 0
      self.tableValueCtrPt = [[[999, 999, 999] for _ in range(100)] for _ in range(100)]
      self.cleanTable()
      print("Manual needle validation segmentation reset!")

  def deleteEvaluationNeedlesFromTable(self):
    """
    Delete all manually segmented needles from report table.
    """
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
    Changes the cursor in all slice widgets.
    """
    profprint()
    appLogic = slicer.app.applicationLogic()
    interactionNode = appLogic.GetInteractionNode()
    interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    layoutManager = slicer.app.layoutManager()
    sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    for nodeIndex in range(sliceNodeCount):
      sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
      sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
      if sliceWidget:
        sliceWidget.setCursor(qt.QCursor(cursorNumber))

  def changeValue(self, *_args):
    """Read the value of the Qt widget and select this needle."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    self.lockControlPoints(widget.editNeedleTxtBox.value)
    self.unlockControlPoints(widget.editNeedleTxtBox.value)
    widget.drawValidationNeedlesButton.text = "Render Manual Needle  " + str(widget.editNeedleTxtBox.value)

  def goToPoint(self, ID):
    """
    Focus the slices on the selected point
    :param ID: name of the fiducial point
    """
    profprint()
    modelNode = getNode(ID)
    coord = [0.0, 0.0, 0.0]
    if modelNode is not None:
      get_fiducial_coords(modelNode, coord)
      X, Y, Z = coord[0], coord[1], coord[2]

      sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode1")
      sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")
      sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeGreen") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode3")

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
    """Reformat the axial view to display the slice containing the currently selected point."""
    profprint()
    self.changeValue()
    widget = slicer.modules.NeedleFinderWidget
    needle = widget.editNeedleTxtBox.value
    coord = [0.0, 0.0, 0.0]
    ptName = '.' + str(needle) + '-' + str(self.ptNumber)
    modelNode = getNode(ptName)
    if modelNode is not None:
        self.ptNumber = self.ptNumber + 1
        X = Y = Z = 0.0
        if modelNode.GetAttribute("ValidationNeedle") == "1":
          get_fiducial_coords(modelNode, coord)
          X, Y, Z = coord[0], coord[1], coord[2]

        sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode1")
        sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")
        sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeGreen") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode3")

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
    """When the button 'new validation needle' is pressed, increment the needle number and reset step to 0."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.validationNeedleButton.text = "New Validation Needle: (" + str(widget.editNeedleTxtBox.value) + ")->(" + str(widget.editNeedleTxtBox.value + 1) + ")"
    widget.editNeedleTxtBox.value += 1
    widget.stepNeedle = 0
    self.lockControlPoints(widget.editNeedleTxtBox.value)

  def filterWithSITK(self):
    """
    Demo method to filter image using SimpleITK.
    """
    profbox()
    backgroundNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    backgroundImage = sitkUtils.PullVolumeFromSlicer(backgroundNode)
    filterImage = sitk.GradientMagnitudeRecursiveGaussian(backgroundImage, 2.0)
    sitkUtils.PushVolumeToSlicer(filterImage, backgroundNode)
    backgroundNode.GetImageData().Modified()
    backgroundNode.Modified()

  def writeTableHeader(self, fileName, variant=0):
    """
    Write table header to file.
    """
    w = slicer.modules.NeedleFinderWidget
    l = w.logic
    if not variant:
      l.exportEvaluation(['user', 'case', 'maxTipHD', 'maxHD', 'avgHD', 'stdHD', 'medHD',
                        'nNeedles', 'nOutliers', 'outliers',
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
                        time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S")
                        ], fileName)
    else:
      l.exportEvaluation(['user', 'case', 'tipHD', 'HD', 'man.-seg_', 'ID1', 'ID2',
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
                        time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S")
                        ], fileName)

  def parSearch(self, mode=False):
    """
    Parameter evaluation/optimization using no, grid brute-force or random-search algo.
    """
    profprint()
    w = slicer.modules.NeedleFinderWidget
    l = w.logic
    path = [0 for _ in range(100)]
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

    self.dirDialog = qt.QFileDialog(w.parent)
    self.dirDialog.setDirectory('/tmp')
    self.dirDialog.options = self.dirDialog.ShowDirsOnly
    self.dirDialog.acceptMode = self.dirDialog.AcceptSave
    out_dir = self.dirDialog.getExistingDirectory()
    w.logDir = out_dir
    print("saving results to ", out_dir)
    try:
      shutil.copyfile(slicer.modules.needlefinder.path, out_dir + '/NeedleFinder_ref.py')
    except Exception:
      breakbox("/!\\ reference source NeedleFinder.py not found!")
    user = 0
    if mode == 0:
      self.writeTableHeader(out_dir + '/AP-All_stats.csv')
      open(out_dir + '/allog.tsv', 'w').close()
      nUsers = 1
      for user in range(nUsers):
        w.userNr = user
        print("simulated user (offset): ", user)
        for case_id in range(100):
          if path[case_id]:
            w.caseNr = case_id
            print("processing ", path[case_id])
            self.writeTableHeader(out_dir + '/User-' + str(user) + '_AP-' + str(case_id) + '.csv', 1)
            slicer.mrmlScene.Clear(0)
            slicer.util.loadScene(path[case_id])
            l.startValidation(script=True, offset=user * 50 / nUsers)
            results, outliers = l.evaluate(script=True)
            for result in results:
              result[0:0] = [user, case_id]
            l.exportEvaluation(results, out_dir + '/User-' + str(user) + '_AP-' + str(case_id) + '.csv')
            HD = np.array(results)
            maxTipHD = HD[:, 2].max()
            maxHD = HD[:, 3].max()
            avgHD = HD[:, 3].mean()
            stdHD = HD[:, 3].std()
            sl = np.sort(HD[:, 3])
            medHD = sl[sl.size // 2]
            resultsEval = [user, case_id, maxTipHD, maxHD, avgHD, stdHD, medHD, len(results), len(outliers), str(outliers)] + l.valuesExperience + [case_id]
            l.exportEvaluation(resultsEval, out_dir + '/AP-All_stats.csv')
      msgbox("parSearch mode 0 done, results in " + out_dir)
    elif mode == 1:
      case_id = 'Current'
      self.writeTableHeader(out_dir + '/BF-' + str(case_id) + '.csv', 1)
      self.writeTableHeader(out_dir + '/BF-' + str(case_id) + '_stats.csv')
      for i in range(3, 12):
        w.numberOfPointsPerNeedle.setValue(i)
        l.startValidation(script=True)
        results, outliers = l.evaluate(script=True)
        for result in results:
          result[0:0] = [user, case_id]
        l.exportEvaluation(results, out_dir + '/BF-' + str(case_id) + '.csv')
        slicer.util.saveScene(out_dir + '/BF-' + str(case_id) + '.mrb')
        HD = np.array(results)
        maxTipHD = HD[:, 2].max()
        maxHD = HD[:, 3].max()
        avgHD = HD[:, 3].mean()
        stdHD = HD[:, 3].std()
        sl = np.sort(HD[:, 3])
        medHD = sl[sl.size // 2]
        resultsEval = [user, case_id, maxTipHD, maxHD, avgHD, stdHD, medHD, len(results), len(outliers), str(outliers)] + l.valuesExperience + [case_id]
        l.exportEvaluation(resultsEval, out_dir + '/BF-' + str(case_id) + '_stats.csv')
      msgbox("parSearch mode 1 done, results in " + out_dir)
    elif mode == 2:
      for case_id in range(100):
        if path[case_id]:
          w.caseNr = case_id
          print("processing ", path[case_id])
          slicer.mrmlScene.Clear(0)
          slicer.util.loadScene(path[case_id])
          self.writeTableHeader(out_dir + '/RS-' + str(case_id) + '.csv', 1)
          self.writeTableHeader(out_dir + '/RS-' + str(case_id) + '_stats.csv')
          for _ in range(1, 10000):
            w.radiusNeedleParameter.setValue(np.random.randint(1, 6))
            w.stepsize.setValue(np.random.randint(1, 40))
            w.sigmaValue.setValue(np.random.randint(1, 40))
            w.gradientPonderation.setValue(np.random.randint(1, 20))
            w.exponent.setValue(np.random.randint(1, 20))
            w.numberOfPointsPerNeedle.setValue(np.random.randint(3, 11))
            l.startValidation(script=True)
            results, outliers = l.evaluate(script=True)
            for result in results:
              result[0:0] = [user, case_id]
            l.exportEvaluation(results, out_dir + '/RS-' + str(case_id) + '.csv')
            slicer.util.saveScene(out_dir + '/RS-' + str(case_id) + '.mrb')
            HD = np.array(results)
            maxTipHD = HD[:, 2].max()
            maxHD = HD[:, 3].max()
            avgHD = HD[:, 3].mean()
            stdHD = HD[:, 3].std()
            sl = np.sort(HD[:, 3])
            medHD = sl[sl.size // 2]
            resultsEval = [user, case_id, maxTipHD, maxHD, avgHD, stdHD, medHD, len(results), len(outliers), str(outliers)] + l.valuesExperience + [case_id]
            l.exportEvaluation(resultsEval, out_dir + '/RS-' + str(case_id) + '_stats.csv')
        msgbox("parSearch mode 2 done, results in " + out_dir)
    slicer.mrmlScene.Clear(0)

  #----------------------------------------------------------------------------------------------
  """ Needle segmentation report"""
  #----------------------------------------------------------------------------------------------

  def confidenceToRGB(self, confidence):
    """Map confidence score (0..100%) to RGB in [0.0, 1.0] (Red -> Orange -> Amber -> Emerald Green)."""
    c = float(np.clip(confidence, 0.0, 100.0))
    if c >= 85.0:
      t = min(1.0, (c - 85.0) / 15.0)
      return (0.16 * (1.0 - t) + 0.09 * t, 0.78 * (1.0 - t) + 0.85 * t, 0.36 * (1.0 - t) + 0.42 * t)
    elif c >= 70.0:
      t = (c - 70.0) / 15.0
      return (0.95 * (1.0 - t) + 0.16 * t, 0.76 * (1.0 - t) + 0.78 * t, 0.14 * (1.0 - t) + 0.36 * t)
    elif c >= 50.0:
      t = (c - 50.0) / 20.0
      return (0.96 * (1.0 - t) + 0.95 * t, 0.45 * (1.0 - t) + 0.76 * t, 0.12 * (1.0 - t) + 0.14 * t)
    else:
      t = max(0.0, (c - 25.0) / 25.0)
      return (0.90 * (1.0 - t) + 0.96 * t, 0.18 * (1.0 - t) + 0.45 * t, 0.18 * (1.0 - t) + 0.12 * t)

  def updateNeedleColorsByConfidence(self, colorByConf=None):
    """Update 3D model colors and Segmentation Report table swatches when toggling confidence coloring."""
    widget = getattr(slicer.modules, "NeedleFinderWidget", None)
    if colorByConf is None and widget is not None and getattr(widget, "colorByConfidenceCheckBox", None) is not None:
      colorByConf = bool(widget.colorByConfidenceCheckBox.isChecked())
    else:
      colorByConf = bool(colorByConf)

    modelNodes = slicer.util.getNodes("vtkMRMLModelNode*")
    for modelNode in list(modelNodes.values()):
      if modelNode is None or modelNode.GetDisplayNode() is None:
        continue
      if modelNode.GetAttribute("type") != "Detection" and not str(modelNode.GetName()).startswith("auto-seg_"):
        continue
      nth_str = modelNode.GetAttribute("nth")
      nth = int(nth_str) % MAXCOL if nth_str is not None else 0
      conf_str = modelNode.GetAttribute("confidence")
      if colorByConf and conf_str is not None:
        try:
          r_f, g_f, b_f = self.confidenceToRGB(float(conf_str))
        except Exception:
          r_f, g_f, b_f = self.color[nth][0], self.color[nth][1], self.color[nth][2]
      else:
        r_f, g_f, b_f = self.color[nth][0], self.color[nth][1], self.color[nth][2]
      modelNode.GetDisplayNode().SetColor(r_f, g_f, b_f)

      row_str = modelNode.GetAttribute("table_row")
      if widget is not None and getattr(widget, "model", None) is not None and row_str is not None:
        try:
          r_idx = int(row_str)
          if 0 <= r_idx < widget.model.rowCount():
            item0 = widget.model.item(r_idx, 0)
            if item0 is not None:
              qcol = qt.QColor(int(round(r_f * 255)), int(round(g_f * 255)), int(round(b_f * 255)))
              item0.setData(qcol, 1)
        except Exception:
          pass

  def addNeedleToTable(self, ID, label=None, needleType=None, comment="", confidence=None):
    """
    Add last segmented needle to the table
    The color icon corresponds to the color of the needle, which corresponds to its label (color code) or confidence score.
    """
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.initTableViewControlPoints()
    widget.initTableView()
    if label is not None:
      ref = int(label[0]) % MAXNEEDLES
      needleLabel = self.option[ref]
    else:
      needleLabel = str(ID)
      ref = ID % MAXNEEDLES

    widget.row = widget.model.rowCount()
    widget.labelStats["Labels"].append(ID)
    widget.labelStats[ref, "#"] = needleLabel

    ################################################
    # Column 0
    color = qt.QColor()
    use_conf_color = (
        confidence is not None
        and getattr(widget, "colorByConfidenceCheckBox", None) is not None
        and widget.colorByConfidenceCheckBox.isChecked()
    )
    if use_conf_color:
      r_f, g_f, b_f = self.confidenceToRGB(confidence)
      color.setRgb(int(round(r_f * 255)), int(round(g_f * 255)), int(round(b_f * 255)))
    else:
      color.setRgb(self.color255[ref][0], self.color255[ref][1], self.color255[ref][2])
    item = qt.QStandardItem()
    item.setData(color, 1)
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
    displayButton.clicked.connect(lambda _=False, who=ID: self.displayNeedleTube(who))
    index = widget.model.index(widget.row, 2)

    widget.items.append(displayButton)
    widget.col += 1
    widget.view.setIndexWidget(index, displayButton)
    ################################################
    # Column 3
    reformatButton = qt.QPushButton("Reformat")
    reformatButton.clicked.connect(lambda _=False, who=ID: self.reformatSagittalView4Needle(who))
    index = widget.model.index(widget.row, 3)
    widget.items.append(reformatButton)
    widget.col += 1
    widget.view.setIndexWidget(index, reformatButton)
    ################################################
    # Column 4
    editField = qt.QTextEdit(str(comment) if comment else "")
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
    widget = slicer.modules.NeedleFinderWidget
    print("len(items): ", len(widget.items))
    labels = widget.labelStats.get("Labels", [])
    if widget.row and ID in labels:
      pos = labels.index(ID)
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

  def addPointToTable(self, ID, needleNumber, pointNumber, label=None, needleType=None, sortTable=1):
    """
    Add control point to the table
    The color icon corresponds to the color of the needle, which corresponds to its label (color code)
    """
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.initTableViewControlPoints()

    widget.labelStatsCTL["Labels"].append(ID)
    widget.labelStatsCTL["ID"].append(1000 * needleNumber + pointNumber)

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
    # Column 2
    colCTL = 2
    item = qt.QStandardItem()
    item.setText(str(pointNumber).zfill(2))
    widget.modelCTL.setItem(widget.rowCTL, colCTL, item)
    widget.itemsCTL.append(item)
    ################################################
    # Column 3
    displayButton = qt.QPushButton("Delete")
    if needleType == 'Validation':
      ID = int(getNode('.' + str(ID)).GetID().strip('vtkMRMLModelNode'))
    listArgs = [ID, needleNumber, pointNumber]
    displayButton.clicked.connect(lambda _=False, who=listArgs: self.removeNeedleShaftEvalMarker(who))
    index = widget.modelCTL.index(widget.rowCTL, 3)
    widget.itemsCTL.append(displayButton)
    widget.viewCTL.setIndexWidget(index, displayButton)
    ################################################
    # Column 4
    reformatButton = qt.QPushButton("Focus")
    reformatButton.clicked.connect(lambda _=False, who=ID: self.goToPoint(who))
    index = widget.modelCTL.index(widget.rowCTL, 4)
    widget.itemsCTL.append(reformatButton)
    widget.viewCTL.setIndexWidget(index, reformatButton)
    widget.rowCTL += 1
    if sortTable:
      widget.viewCTL.sortByColumn(2, qt.Qt.AscendingOrder)
      widget.viewCTL.sortByColumn(1, qt.Qt.AscendingOrder)

  def deletePointFromTable(self, ID):
    """
    Delete point from table
    """
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    sortedIndex = widget.labelStatsCTL['ID']
    sortedIndex.sort(reverse=True)
    label = ID.split('.')
    labelval = label[1].split('-')
    val = 1000 * int(labelval[0]) + int(labelval[1])
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
    '''
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    widget.labelStatsCTL = {}
    widget.labelStats = {}

    widget.table = None
    widget.tableCTL = None
    if widget.modelCTL:
      widget.modelCTL.clear()
    if widget.model:
      widget.model.clear()
    for list_name in ('Fiducials List', 'All Annotations'):
      nodes = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode', list_name)
      for i in range(nodes.GetNumberOfItems()):
        slicer.mrmlScene.RemoveNode(nodes.GetItemAsObject(i))

    self.observeManualNeedles()

  def AddRadiation(self, i, needleID):
    """
    Placeholder for radiation quadrics model.
    """
    profbox()

  def needleSegmentationCLIDEMO(self):
    """
    Function used for the CLI module from Yi Gao using hessian filter (C++)
    """
    profbox()
    widget = slicer.modules.NeedleFinderWidget
    pNode = self.parameterNode()
    if slicer.mrmlScene.GetNodeByID(pNode.GetParameter("baselineVolumeID")) is None:
      inputVolume = self.__volumeSelector.currentNode()
      inputVolumeID = self.__volumeSelector.currentNode().GetID()
    else:
      inputVolume = slicer.mrmlScene.GetNodeByID(pNode.GetParameter("baselineVolumeID"))
      inputVolumeID = inputVolume.GetID()
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

    parameters = {
        'inputVolume': inputVolumeID,
        'inputLabel': inputLabelID,
        'outputVtk': outputID,
        'outputFolderName': self.foldername,
        'nbPointsPerLine': self.nbPointsPerLine.value,
        'nbRadiusIterations': self.nbRadiusIterations.value,
        'radiusMax': self.radiusMax.value,
        'numberOfPointsPerNeedle': self.numberOfPointsPerNeedle.value,
        'nbRotatingIterations': self.nbRotatingIterations.value,
    }

    module = slicer.modules.mainlabelneedletrackingcli
    self.__cliNode = slicer.cli.run(module, None, parameters, wait_for_completion=True)

    self.setNeedleCoordinates()
    self.computerPolydataAndMatrix()
    k = 0

    self.base = [[0 for _ in range(3)] for _ in range(63)]
    self.tip = [[0 for _ in range(3)] for _ in range(63)]
    self.needlenode = [[0 for _ in range(2)] for _ in range(63)]
    self.bentNeedleNode = [[0 for _ in range(2)] for _ in range(63)]
    self.displaynode = [0 for _ in range(63)]
    self.displaynodeB = [0 for _ in range(63)]
    self.fiducialnode = [0 for _ in range(63)]

    for i in range(63):
      pathneedle = self.foldername + '/' + str(i) + '.vtp'
      pathBentNeedle = self.foldername + '/' + str(i) + '_bent.vtp'
      self.needlenode[i] = slicer.util.loadModel(pathneedle, True)
      self.bentNeedleNode[i] = slicer.util.loadModel(pathBentNeedle, True)

      if self.needlenode[i][0] and self.needlenode[i][1] is not None:
        self.displaynode[i] = self.needlenode[i][1].GetDisplayNode()
        self.displaynodeB[i] = self.bentNeedleNode[i][1].GetDisplayNode()
        polydata = self.needlenode[i][1].GetPolyData()
        polydata.GetPoint(0, self.base[i])
        self.displaynode[i].Visibility2DOn()
        self.displaynodeB[i].Visibility2DOn()
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
    """Buttons for the reporting widget, displaying information of the segmented needles."""
    profbox()
    if self.buttonsGroupBox is not None:
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

        buttonDisplay.clicked.connect(lambda _=False, who=i: self.displayNeedle(who))
        buttonBentDisplay.clicked.connect(lambda _=False, who=i: self.displayBentNeedle(who))
        buttonReformat = qt.QPushButton("Reformat " + self.option[i])
        buttonReformat.clicked.connect(lambda _=False, who=i: self.reformatSagittalView4Needle(who))
        widgets = qt.QWidget()
        hlay = qt.QHBoxLayout(widgets)
        hlay.addWidget(buttonDisplay)
        hlay.addWidget(buttonBentDisplay)
        hlay.addWidget(buttonReformat)
        self.buttonsGroupBoxLayout.addRow(widgets)

  def displayBentNeedle(self, i):
    """Toggle visibility of bent needle model."""
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("nth") == str(i) and modelNode.GetAttribute("optimized") == '1':
        displayNode = modelNode.GetModelDisplayNode()
        if displayNode.GetVisibility():
          displayNode.Visibility2DOff()
          displayNode.SetVisibility(0)
        else:
          displayNode.Visibility2DOn()
          displayNode.SetVisibility(1)

  def displayNeedle(self, i):
    """Toggle visibility of straight needle model."""
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("nth") == str(i) and modelNode.GetAttribute("segmented") == '1':
        displayNode = modelNode.GetModelDisplayNode()
        if displayNode.GetVisibility():
          displayNode.Visibility2DOff()
          displayNode.SetVisibility(0)
        else:
          displayNode.Visibility2DOn()
          displayNode.SetVisibility(1)

  def showOneNeedle(self, i, visibility):
    """Show or hide a single needle and its label fiducial."""
    profbox()
    fidname = "fid" + self.option[i]
    pNode = self.parameterNode()
    needleID = pNode.GetParameter(self.option[i] + '.vtp')
    fidID = pNode.GetParameter(fidname)
    NeedleNode = slicer.mrmlScene.GetNodeByID(needleID)
    fiducialNode = slicer.mrmlScene.GetNodeByID(fidID)

    if NeedleNode is not None:
      displayNode = NeedleNode.GetModelDisplayNode()
      if fiducialNode is None:
        displayNode.SetVisibility(1)
        displayNode.SetOpacity(0.9)
        polyData = NeedleNode.GetPolyData()
        nb = int(polyData.GetNumberOfPoints() - 1)
        coord = [0.0, 0.0, 0.0]
        if nb > 100:
          polyData.GetPoint(nb, coord)
          fiducialNode = create_fiducial_node(
              name=self.option[i],
              ras=coord,
              color=NeedleNode.GetDisplayNode().GetColor(),
              glyph_scale=0.0,
              text_scale=3.0,
              locked=True,
              selectable=False,
              visibility=True,
          )
          pNode.SetParameter(fidname, fiducialNode.GetID())

      vis_flag = 1 if visibility else 0
      displayNode.SetVisibility(vis_flag)
      displayNode.SetVisibility2D(vis_flag)
      if fiducialNode is not None:
        fiducialNode.SetDisplayVisibility(vis_flag)
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
    """Add planned needle model to scene."""
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
    displayNode.SetVisibility(1)

  def displayRadPlanned(self):
    """Display 'radiation' of planned needle."""
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("radiation") == "planned":
        needleNode = slicer.mrmlScene.GetNodeByID(modelNode.GetAttribute("needleID"))
        if needleNode and needleNode.GetDisplayVisibility() == 1:
          modelNode.SetDisplayVisibility(abs(int(slicer.modules.NeedleFinderWidget.displayRadPlannedButton.checked) - 1))

  def displayRadSegmented(self):
    """Display 'radiation' of segmented needles."""
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("radiation") == "segmented":
        needleNode = slicer.mrmlScene.GetNodeByID(modelNode.GetAttribute("needleID"))
        if needleNode is not None and needleNode.GetDisplayVisibility() == 1:
          vis = abs(int(slicer.modules.NeedleFinderWidget.displayRadSegmentedButton.checked) - 1)
          modelNode.SetDisplayVisibility(vis)
          modelNode.GetDisplayNode().SetVisibility2D(vis)

  def displayContour(self, i, visibility):
    """Display the iso-contour of needle i."""
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("contour") == "1" and modelNode.GetAttribute("nth") == str(i):
        needleNode = slicer.mrmlScene.GetNodeByID(modelNode.GetAttribute("needleID"))
        if needleNode is not None and needleNode.GetDisplayVisibility() == 1:
          modelNode.SetDisplayVisibility(visibility)
          modelNode.GetDisplayNode().SetVisibility2D(visibility)

  def displayContours(self):
    """Display or hide the iso-contours of every needle."""
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("contour") == "1":
        needleNode = slicer.mrmlScene.GetNodeByID(modelNode.GetAttribute("needleID"))
        if needleNode is not None and needleNode.GetDisplayVisibility() == 1:
          vis = abs(int(slicer.modules.NeedleFinderWidget.displayContourButton.checked) - 1)
          modelNode.SetDisplayVisibility(vis)
          modelNode.GetDisplayNode().SetVisibility2D(vis)

  def displayFiducial(self):
    """Show labels of the needles by adding a fiducial point at the tip."""
    profbox()
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    for modelNode in list(modelNodes.values()):
      if modelNode.GetAttribute("segmented") == "1" and modelNode.GetAttribute("nth") is not None:
        i = int(modelNode.GetAttribute("nth"))
        if self.fiducialnode[i] == 0:
          polyData = modelNode.GetPolyData()
          nb = int(polyData.GetNumberOfPoints() - 1)
          coord = [0.0, 0.0, 0.0]
          if nb > 10:
            polyData.GetPoint(nb, coord)
            self.fiducialnode[i] = create_fiducial_node(
                name=self.option[i],
                ras=coord,
                color=modelNode.GetDisplayNode().GetColor(),
                glyph_scale=0.0,
                text_scale=3.0,
                locked=True,
                selectable=False,
                visibility=bool(modelNode.GetDisplayNode().GetVisibility()),
            )
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
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    if assign:
      while slicer.util.getNodes('template slice position*') != {}:
        nodes = slicer.util.getNodes('template slice position*')
        for node in list(nodes.values()):
          slicer.mrmlScene.RemoveNode(node)

      s = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetSliceNode()
      offSet = s.GetSliceOffset()
      rasVector = [0, 0, offSet]
      widget.axialSegmentationLimit = int(round(self.ras2ijk(rasVector)[2]))
      widget.axialSegmentationLimitRAS = float(offSet)
      widget.templateSliceButton.text = (
          f"1. Select Current Axial Slice as Seg. Limit (current: {offSet:.1f} mm / slice {widget.axialSegmentationLimit})"
      )

      self.fiducialNode = create_fiducial_node(
          name='template slice position',
          ras=rasVector,
          color=(0.0, 1.0, 0.0),
          visibility=True,
      )
    else:
      k_lim, z_ras = self.findAxialSegmentationLimitFromMarker(bForceFallback=False)
      widget.axialSegmentationLimit = int(k_lim)
      widget.axialSegmentationLimitRAS = float(z_ras)
      if widget.templateSliceButton is not None:
        widget.templateSliceButton.text = (
            f"1. Select Current Axial Slice as Seg. Limit (current: {z_ras:.1f} mm / slice {k_lim})"
        )
    widget.fiducialButton.setEnabled(1)
    widget.createAddOrSelectLabelMapNode()

    crosshairNode = getNode("Crosshair")
    crosshairNode.SetCrosshairMode(slicer.vtkMRMLCrosshairNode().ShowIntersection)

  def exportEvaluation(self, results, url):
    """Export evaluation results to a CSV file (e.g. when doing parameter optimizations)."""
    profprint()
    if not os.path.exists(url):
      print("creating new results file: ", url)
      open(url, 'w').close()
    with open(url, 'a', newline='') as myfile:
      wr = csv.writer(myfile)
      r = np.array(results)
      if len(r.shape) == 1:
        wr.writerow(results)
      else:
        wr.writerows(results)

  def hausdorffDistance(self, id1, id2):
    r"""Calculates the Hausdorff's distance [HD]_ of two needles."""
    if frequent:
      profprint()
    node1 = slicer.mrmlScene.GetNodeByID(id1)
    node2 = slicer.mrmlScene.GetNodeByID(id2)
    return compute_hausdorff_distance(node1.GetPolyData(), node2.GetPolyData(), step1=10, step2=10, truncate=True)

  def hausdorffDistance13(self, id1, id2):
    """MICCAI13 Version."""
    if frequent:
      profprint()
    node1 = slicer.mrmlScene.GetNodeByID(id1)
    node2 = slicer.mrmlScene.GetNodeByID(id2)
    return compute_hausdorff_distance(node1.GetPolyData(), node2.GetPolyData(), step1=100, step2=10, truncate=False)

  def evaluate(self, script=False, needleNr=None):
    """
    This function first invokes needleMatching() with, for each automatically segmented needle in the vtkMRMLScene,
    associates it with its manually segmented version.
    Then, the HD is computed between each pairs of needle and the results are reported in a numpy array

    :return numpy array of [ value , ID needle 1, ID needle 2 ]
    """
    profprint()
    result = self.needleMatching()
    HD = []
    outliers = []
    widget = slicer.modules.NeedleFinderWidget
    self.valuesExperience = [
        widget.radiusNeedleParameter.value,
        widget.lenghtNeedleParameter.value,
        widget.radiusMax.value,
        widget.numberOfPointsPerNeedle.value,
        widget.nbRotatingIterations.value,
        widget.stepsize.value,
        widget.gradientPonderation.value,
        widget.exponent.value,
        widget.gaussianAttenuationButton.isChecked() * 1,
        widget.sigmaValue.value,
        widget.algoVersParameter.value,
    ]

    for i in range(len(result)):
      if widget.algoVersParameter.value == 0:
        val = self.hausdorffDistance(result[i][1], result[i][2])
      else:
        val = self.hausdorffDistance13(result[i][1], result[i][2])
      try:
        needleNrFromFilename = int(result[i][3].strip('manual-seg_'))
      except Exception:
        needleNrFromFilename = -1
      if needleNr is not None and needleNr != needleNrFromFilename:
        continue
      base_row = [
          result[i][0],
          float(val),
          int(needleNrFromFilename),
          extract_model_id_number(result[i][1]),
          extract_model_id_number(result[i][2]),
          val > outlierThresh_mm,
      ]
      if script:
        results = base_row + self.valuesExperience
      else:
        results = base_row
      HD.append(results)
      if val > outlierThresh_mm:
        outliers.append(needleNrFromFilename)
    if not script:
      return np.array(HD).astype(np.double)
    else:
      return HD, outliers

  def distTip(self, id1, id2):
    """Returns the distance between the tip of two needles."""
    if frequent:
      profprint()
    node = slicer.mrmlScene.GetNodeByID('vtkMRMLModelNode' + str(id1))
    node2 = slicer.mrmlScene.GetNodeByID('vtkMRMLModelNode' + str(id2))
    return compute_tip_distance(node.GetPolyData(), node2.GetPolyData())

  def distBase(self, id1, id2):
    """Returns the distance between the base of two needles."""
    if frequent:
      profprint()
    node = slicer.mrmlScene.GetNodeByID('vtkMRMLModelNode' + str(id1))
    node2 = slicer.mrmlScene.GetNodeByID('vtkMRMLModelNode' + str(id2))
    return compute_base_distance(node.GetPolyData(), node2.GetPolyData())

  def needleMatching(self):
    """Associates manually segmented needles to their automatically segmented version."""
    profprint()
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    nbNode = modelNodes.GetNumberOfItems()
    result = []
    found = []
    for nthNode in range(nbNode):
      node = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLModelNode')
      if node.GetID() not in found and node.GetAttribute('type') != 'Validation':
        dist = []
        polydata = node.GetPolyData()
        if polydata is not None:
          for nthNode2 in range(nbNode):
            node2 = slicer.mrmlScene.GetNthNodeByClass(nthNode2, 'vtkMRMLModelNode')
            if node2.GetID() not in found and node2.GetAttribute('type') == 'Validation':
              polydata2 = node2.GetPolyData()
              if polydata2 is not None and polydata2.GetNumberOfPoints() > 100 and polydata.GetNumberOfPoints() > 100:
                tipDistance = self.distTip(extract_model_id_number(node.GetID()), extract_model_id_number(node2.GetID()))
                name = node.GetName()
                manualName = (
                    name.lstrip('auto-seg_')
                    .lstrip('manual-seg_')
                    .lstrip('obturator-seg_')
                    .lstrip('0123456789')
                    .lstrip('-ID-vtkMRMLModelNode')
                    .lstrip('0123456789-')
                )
                if manualName == node2.GetName():
                  dist.append([tipDistance, node2.GetID(), node2.GetName()])
          if dist != []:
            match = [min(dist)[0], min(dist)[1], node.GetID(), min(dist)[2]]
            result.append(match)
            found.append(min(dist)[1])
            found.append(node.GetID())
            node.GetDisplayNode().SetVisibility2D(1)
    return result

  def setAllNeedleTubesAsValidationNeedles(self):
    """Set all non-validation needle models to validation type."""
    profprint()
    modelNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')
    nbNode = modelNodes.GetNumberOfItems()
    for nthNode in range(nbNode):
      node = slicer.mrmlScene.GetNthNodeByClass(nthNode, 'vtkMRMLModelNode')
      if node.GetID() and node.GetAttribute('type') != 'Validation':
        node.SetAttribute('type', 'Validation')
        displayNode = node.GetDisplayNode()
        colorVar = random.randrange(50, 100, 1)
        nth = int(colorVar) % MAXCOL
        displayNode.SetColor(self.color[int(nth)][0], self.color[int(nth)][1], self.color[int(nth)][2])
        displayNode.SetVisibility2D(True)
        displayNode.SetSliceIntersectionThickness(2)

  def distance(self, pt1, pt2):
    """3D distance between two points."""
    if frequent:
      profprint()
    return distance_3d(pt1, pt2)

  def addManualTip(self, A):
    """Add fiducial node to the scene as starting point for needle segmentation."""
    profbox()
    self.fiducialNode = create_fiducial_node(
        name='tip',
        ras=A,
        color=(0.0, 1.0, 0.0),
        visibility=True,
    )

  def distanceTwoPoints(self, A, B):
    """3D distance between two points."""
    if frequent:
      profprint()
    return distance_3d(A, B)

  def clipPolyData(self, node, value, visible=0):
    """Function used to truncate needle models to evaluate the Hausdorff's distance."""
    profprint()
    polyData = clip_polydata_axial(node.GetPolyData(), value)
    if visible == 1:
      scene = slicer.mrmlScene
      model = slicer.vtkMRMLModelNode()
      model.SetScene(scene)
      model.SetAndObservePolyData(polyData)
      modelDisplay = slicer.vtkMRMLModelDisplayNode()
      modelDisplay.SetScene(scene)
      scene.AddNode(modelDisplay)
      model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
      modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
      scene.AddNode(model)
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
