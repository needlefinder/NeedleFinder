"""**NeedleFinder Documentation**

Guillaume Pernelle,  Andre Mastmeyer, Ruibin Ma

.. moduleauthor:: gpernelle <gpernelle@gmail.com>

.. image:: https://raw.githubusercontent.com/gpernelle/NeedleFinder/amastm/docs/_static/algo-figure-high_res.png
           :height: 250px
           :width: 700 px
           :scale: 100 %
           :alt: alternate text
           :align: center

Links
-----
    * Validation of Catheter Segmentation for MR-guided Gynecologic Cancer Brachytherapy [1]_

    * Labeled Needle Rendering Solution for Image Guided Brachytherapy [2]_

  .. [1] https://www.spl.harvard.edu/publications/item/view/2459
  .. [2] https://www.spl.harvard.edu/publications/item/view/2316
"""

import fnmatch
from functools import partial
import os
import random
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

from NeedleFinderLib import (
    AssistedSegmentationMixin,
    EditorLib,
    EditorWidget,
    Helper,
    LabelStatisticsLogic,
    MAXCOL,
    MAXNEEDLES,
    SceneAndEvaluationMixin,
    TrackingAlgorithmsMixin,
    add_fiducial_modified_observer,
    assertbox,
    binomial,
    breakbox,
    cmp,
    conesColor,
    create_fiducial_node,
    factorial,
    fibonacci,
    frequent,
    get_fiducial_coords,
    getClassName,
    getNode,
    ijk_to_ras,
    is_fiducial_node,
    is_rust_available,
    lineno,
    loggingInfos,
    logprint,
    msgbox,
    outlierThresh_mm,
    pause,
    profbox,
    profiling,
    profprint,
    ras_to_ijk,
    set_fiducial_coords,
    sort_table,
    step_size,
    step_size_13,
    step_size_andre,
    whoami,
    whosdaddy,
    whosgranny,
)


class NeedleFinder(ScriptedLoadableModule):
  def __init__(self, parent):
    """Initialize the module."""
    profprint()
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "NeedleFinder"
    parent.categories = ["IGT"]
    parent.dependencies = []
    parent.contributors = [
        "Guillaume Pernelle",
        "Andre Mastmeyer",
        "Ruibin Ma",
        "Alireza Mehrtash",
        "Lauren Barber",
        "Nabgha Fahrat",
        "Sandy Wells",
        "Yi Gao",
        "Antonio Damato",
        "Tina Kapur",
        "Akila Viswanathan",
    ]
    parent.helpText = "https://github.com/gpernelle/NeedleFinder/wiki"
    parent.acknowledgementText = " Version : NeedleFinder 2015 v1.0."
    self.NeedleFinderWidget = 0
    self.parent = parent
    self.loaded = 0
    self.logic = NeedleFinderLogic()
    if not hasattr(slicer, 'selfTests'):
      slicer.selfTests = {}
    slicer.selfTests['NeedleFinder'] = self.runTest

    def __onNodeAdded__(self, caller, eventId, callData=None):
      """If fiducial node, observe movement for undo function."""
      self.logic.observeSingleFiducial(callData, eventId)

    def __onNodeRemoved__(self, caller, eventId, callData=None):
      """Delete observer if fiducial node removed."""
      self.logic.removeNodeObserver(caller, eventId)

    def __onSceneLoaded__(self, caller, eventId, callData=None):
      """Load CTRL points AFTER scene finished loading."""
      if hasattr(slicer.modules, 'NeedleFinderWidget') and slicer.modules.NeedleFinderWidget:
        self.logic.loadCTLPointsInTable()

    def __onSceneClosed__(self, caller, eventId, callData=None):
      """Clean report table and internal variables."""
      if hasattr(slicer.modules, 'NeedleFinderWidget') and slicer.modules.NeedleFinderWidget:
        self.logic.cleanTable()
        w = slicer.modules.NeedleFinderWidget
        w.initObturatorNeedles()

    self.__onSceneLoaded__ = partial(__onSceneLoaded__, self)
    self.__onSceneClosed__ = partial(__onSceneClosed__, self)
    self.__onNodeAdded__ = partial(__onNodeAdded__, self)
    self.__onNodeAdded__.CallDataType = vtk.VTK_OBJECT

    slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.EndImportEvent, self.__onSceneLoaded__)
    slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.EndCloseEvent, self.__onSceneClosed__)
    slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.__onNodeAdded__)

  def __del__(self):
    if self.NeedleFinderWidget:
      self.NeedleFinderWidget.removeObservers()

  def getName(self):
    """Return class name."""
    return self.__class__.__name__

  def runTest(self):
    """Run unit tests."""
    profprint()
    tester = NeedleFinderTest()
    tester.runTest()


class NeedleFinderWidget(ScriptedLoadableModuleWidget):

  def __init__(self, parent=None):
    """Initialize the widget."""
    profprint()
    ScriptedLoadableModuleWidget.__init__(self, parent)
    if not parent:
      self.parent = slicer.qMRMLWidget()
      self.parent.setLayout(qt.QVBoxLayout())
      self.parent.setMRMLScene(slicer.mrmlScene)
    else:
      self.parent = parent
    self.layout2 = self.parent.layout()
    self.layout = qt.QFormLayout()
    slicer.modules.NeedleFinderWidget = self

    self.analysisGroupBox = None
    self.buttonsGroupBox = None
    self.interactorObserverTags = []
    self.styleObserverTags = []
    self.observerTagsFid = {}
    self.axialSegmentationLimit = None
    self.templateSliceButton = None
    self.fiducialButton = None
    self.newInsertionButton = None
    self.deleteNeedleButton = None
    self.resetDetectionButton = None
    self.resetValidationButton = None
    self.validationNeedleNumber = 1
    self.stepNeedle = 0
    self.fileName = None
    self.dirDialog = None
    self.logic = NeedleFinderLogic()
    self.needleValidationClicks = 1
    self.addManualTipClicks = 2
    self.addCTLPoints = 14
    self.obturatorNeedleTipClicks = 3
    self.caseNr = 0
    self.userNr = 0
    self.logDir = tempfile.gettempdir()

    self.table = None
    self.tableCTL = None
    self.view = None
    self.viewCTL = None
    self.model = None
    self.modelCTL = None

    self.initObturatorNeedles()

    self.styleObserverTags = []
    self.sliceWidgetsPerStyle = {}

    self.CrosshairNode = None
    self.CrosshairNodeObserverTag = None

    self.CrosshairNode = slicer.mrmlScene.GetNthNodeByClass(0, 'vtkMRMLCrosshairNode')
    if self.CrosshairNode:
      self.CrosshairNodeObserverTag = self.CrosshairNode.AddObserver(
          slicer.vtkMRMLCrosshairNode.CursorPositionModifiedEvent, self.processEvent
      )

    self.editorWidget = None
    self.editUtil = None
    self.undoRedo = None
    self.wandLogics = {}
    self.labelMapNode = None
    self.currentLabel = None
    self.tempPointList = []
    self.undoListFid = [{} for i in range(30)]
    if not parent:
      self.setup()
      self.parent.show()

  def __del__(self):
    self.removeObservers()

  def getName(self):
    """Return class name."""
    return self.__class__.__name__

  def initObturatorNeedles(self):
    self.obtuNeedle = 0
    self.obtuNeedleValueCtrPt = [[[999, 999, 999] for i in range(10)] for j in range(10)]
    self.obtuNeedlePt = [[[999, 999, 999] for i in range(10)] for j in range(10)]

  def initTableView(self):
    """Initialize a table gathering information on segmented needles."""
    profprint()
    if self.table is None:
      self.keys = ("#")
      self.labelStats = {}
      self.labelStats['Labels'] = []
      self.items = []
      if self.model is None:
        self.model = qt.QStandardItemModel()
      self.model.setColumnCount(5)
      self.model.setHeaderData(0, 1, "")
      self.model.setHeaderData(1, 1, "#")
      self.model.setHeaderData(2, 1, "Display")
      self.model.setHeaderData(3, 1, "Reformat")
      self.model.setHeaderData(4, 1, "Comments")
      if self.view is None:
        self.view = qt.QTableView()
        self.view.setMinimumHeight(240)
        self.view.sortingEnabled = False
        self.view.verticalHeader().visible = False
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.alternatingRowColors = True
        self.analysisGroupBoxLayout.addRow(self.view)

      self.view.setModel(self.model)
      self.view.setColumnWidth(0, 24)
      self.view.setColumnWidth(1, 64)
      self.view.setColumnWidth(2, 76)
      self.view.setColumnWidth(3, 84)
      self.table = 1
      self.row = 0
      self.col = 0

  def initTableViewControlPoints(self):
    """Initialize a table gathering control points from manual segmentation."""
    profprint()
    if self.tableCTL is None:
      self.keysCTL = ("#")
      self.labelStatsCTL = {}
      self.labelStatsCTL['Labels'] = []
      self.labelStatsCTL['ID'] = []
      self.itemsCTL = []
      if self.modelCTL is None:
        self.modelCTL = qt.QStandardItemModel()
      self.modelCTL.setColumnCount(5)
      self.modelCTL.setHeaderData(0, 1, "")
      self.modelCTL.setHeaderData(1, 1, "# Needle")
      self.modelCTL.setHeaderData(2, 1, "# Point")
      self.modelCTL.setHeaderData(3, 1, "Delete")
      self.modelCTL.setHeaderData(4, 1, "Focus")
      if self.viewCTL is None:
        self.viewCTL = qt.QTableView()
        self.viewCTL.setMinimumHeight(240)
        self.viewCTL.sortingEnabled = True
        self.viewCTL.verticalHeader().visible = False
        self.viewCTL.horizontalHeader().setStretchLastSection(True)
        self.viewCTL.alternatingRowColors = True
        self.analysisGroupBoxLayoutCTL.addRow(self.viewCTL)

      self.viewCTL.setModel(self.modelCTL)
      self.viewCTL.setColumnWidth(0, 24)
      self.viewCTL.setColumnWidth(1, 72)
      self.viewCTL.setColumnWidth(2, 72)
      self.viewCTL.setColumnWidth(3, 76)
      self.tableCTL = 1
      self.rowCTL = 0
      self.colCTL = 0

  def createAddOrSelectLabelMapNode(self, script=False):
    """Create label map node for Segmentation editor and needle finder."""
    profprint()
    print("creating label map for working intensity volume")
    volLogic = slicer.modules.volumes.logic()
    sliceLogic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
    vn = sliceLogic.GetBackgroundLayer().GetVolumeNode()
    self.labelMapNode = getNode(vn.GetName() + "-label")
    if not self.labelMapNode:
      self.labelMapNode = volLogic.CreateAndAddLabelVolume(slicer.mrmlScene, vn, vn.GetName() + "-label")
    if not script:
      selectionNode = slicer.app.applicationLogic().GetSelectionNode()
      selectionNode.SetReferenceActiveLabelVolumeID(self.labelMapNode.GetID())
      for view_name in ("Red", "Yellow", "Green"):
        sc = slicer.app.layoutManager().sliceWidget(view_name).sliceController()
        sc.setLabelMapOpacity(0.5)
      for node_id, fallback_id in (
          ("vtkMRMLSliceNodeRed", "vtkMRMLSliceNode1"),
          ("vtkMRMLSliceNodeYellow", "vtkMRMLSliceNode2"),
          ("vtkMRMLSliceNodeGreen", "vtkMRMLSliceNode3"),
      ):
        s_node = slicer.mrmlScene.GetNodeByID(node_id) or slicer.mrmlScene.GetNodeByID(fallback_id)
        if s_node:
          s_node.SetUseLabelOutline(1)
    self.editorWidget.setMasterNode(vn)
    self.editorWidget.setMergeNode(self.labelMapNode)

  def onEditorCollapsed(self, collapsed):
    """When segmentation editor is used before needle finder, prepare a label volume."""
    profprint()
    if not collapsed:
      self.createAddOrSelectLabelMapNode()

  def setup(self):
    """Instantiate and connect widgets."""
    profprint()
    logic = self.logic

    # Dark-mode & light-mode compatible high-contrast stylesheets
    toggle_ready_style = (
        "QPushButton { background-color: #238636; color: #ffffff; font-weight: 600; padding: 5px 8px; border-radius: 4px; }"
        "QPushButton:hover { background-color: #2ea043; }"
        "QPushButton:disabled { background-color: rgba(128, 128, 128, 0.25); color: rgba(160, 160, 160, 0.7); }"
        "QPushButton:checked { background-color: #da3633; color: #ffffff; font-weight: 700; }"
    )
    primary_action_style = (
        "QPushButton { background-color: #1f6feb; color: #ffffff; font-weight: 600; padding: 5px 8px; border-radius: 4px; }"
        "QPushButton:hover { background-color: #388bfd; }"
        "QPushButton:disabled { background-color: rgba(128, 128, 128, 0.25); color: rgba(160, 160, 160, 0.7); }"
    )
    secondary_toggle_style = (
        "QPushButton { padding: 5px 8px; }"
        "QPushButton:checked { background-color: #da3633; color: #ffffff; font-weight: 700; }"
    )

    self.__reportFrame = ctk.ctkCollapsibleButton()
    self.__reportFrame.text = "Segmentation Report"
    self.__reportFrame.collapsed = 1
    reportFrame = qt.QFormLayout(self.__reportFrame)

    self.analysisGroupBox = qt.QWidget()
    reportFrame.addRow(self.analysisGroupBox)
    self.analysisGroupBoxLayout = qt.QFormLayout(self.analysisGroupBox)
    self.analysisGroupBoxLayout.setContentsMargins(0, 0, 0, 0)

    self.__reportFrameCTL = ctk.ctkCollapsibleButton()
    self.__reportFrameCTL.text = "Manual Segmentation Points"
    self.__reportFrameCTL.collapsed = 1
    reportFrameCTL = qt.QFormLayout(self.__reportFrameCTL)

    self.analysisGroupBoxCTL = qt.QWidget()
    reportFrameCTL.addRow(self.analysisGroupBoxCTL)
    self.analysisGroupBoxLayoutCTL = qt.QFormLayout(self.analysisGroupBoxCTL)
    self.analysisGroupBoxLayoutCTL.setContentsMargins(0, 0, 0, 0)

    # ------------------------------------------------------------------
    # 1. Automated Needle Segmentation
    # ------------------------------------------------------------------
    self.__segmentationFrame = ctk.ctkCollapsibleButton()
    self.__segmentationFrame.text = "1. Automated Needle Segmentation"
    self.__segmentationFrame.collapsed = 0
    segmentationFrame = qt.QFormLayout(self.__segmentationFrame)

    self.algoComboBox = qt.QComboBox()
    self.algoComboBox.addItems([
        "0 — 3x3 Neighborhood Cone Search (Dev)",
        "1 — MICCAI 2013 v1 (Cone Ray Search)",
        "2 — MICCAI 2013 v2 (Spacing-Corrected Cone)",
        "3 — MICCAI 2015 (Mechanical Deflection Model) [Recommended]",
        "4 — MICCAI 2013 + Multi-Needle Rationality Check",
        "5 — MICCAI 2015 + Multi-Needle Rationality Check",
    ])
    self.algoComboBox.setCurrentIndex(3)
    self.algoComboBox.toolTip = "Select the needle tracking algorithm version."
    segmentationFrame.addRow("Tracking Algorithm:", self.algoComboBox)

    self.templateSliceButton = qt.QPushButton('1. Select Current Axial Slice as Seg. Limit (current: None)')
    self.templateSliceButton.toolTip = "Set the axial base limit slice (template plane) from the current Red slice position."
    self.templateSliceButton.clicked.connect(lambda _=False: logic.placeAxialLimitMarker(assign=True))
    self.templateSliceButton.setEnabled(1)

    self.skipSegLimitButton = qt.QPushButton('Skip Limit')
    self.skipSegLimitButton.checkable = False
    self.skipSegLimitButton.toolTip = "Use an existing axial segmentation limit marker in the scene."
    self.skipSegLimitButton.clicked.connect(lambda _=False: self.onSkipSegLimit())

    step1Row = qt.QFrame()
    step1Layout = qt.QHBoxLayout(step1Row)
    step1Layout.setContentsMargins(0, 0, 0, 0)
    step1Layout.addWidget(self.templateSliceButton, 4)
    step1Layout.addWidget(self.skipSegLimitButton, 1)
    segmentationFrame.addRow(step1Row)

    self.fiducialButton = qt.QPushButton('2. Start Giving Needle Tips [CTRL + ENTER]')
    self.fiducialButton.checkable = True
    self.fiducialButton.setStyleSheet(toggle_ready_style)
    self.fiducialButton.toggled.connect(self.onStartStopGivingNeedleTipsToggled)
    self.fiducialButton.setEnabled(0)
    segmentationFrame.addRow(self.fiducialButton)

    self.autoDetectAllButton = qt.QPushButton('Auto-Detect All Needles')
    self.autoDetectAllButton.toolTip = (
        "Automatically detect and segment all needle trajectories in the volume "
        "using 3D symmetric tubular filtering and multi-slice momentum tracking "
        "(no manual tip clicks required)."
    )
    self.autoDetectAllButton.setStyleSheet(primary_action_style)
    self.autoDetectAllButton.clicked.connect(self.onAutoDetectAllNeedlesClicked)

    self.minConfidenceSpinBox = qt.QDoubleSpinBox()
    self.minConfidenceSpinBox.setRange(5.0, 95.0)
    self.minConfidenceSpinBox.setSingleStep(5.0)
    self.minConfidenceSpinBox.setDecimals(0)
    self.minConfidenceSpinBox.setValue(35.0)
    self.minConfidenceSpinBox.setSuffix("%")
    self.minConfidenceSpinBox.toolTip = (
        "Minimum confidence score (%) for automatically detected needles."
    )

    autoDetectRow = qt.QFrame()
    autoDetectLayout = qt.QHBoxLayout(autoDetectRow)
    autoDetectLayout.setContentsMargins(0, 0, 0, 0)
    autoDetectLayout.addWidget(self.autoDetectAllButton, 4)
    minConfLabel = qt.QLabel("Min Conf:")
    autoDetectLayout.addWidget(minConfLabel, 0)
    autoDetectLayout.addWidget(self.minConfidenceSpinBox, 1)
    segmentationFrame.addRow(autoDetectRow)

    self.colorByConfidenceCheckBox = qt.QCheckBox("Color Needles by Confidence Score")
    self.colorByConfidenceCheckBox.checked = False
    self.colorByConfidenceCheckBox.toolTip = (
        "Color detected 3D needle tubes and Segmentation Report swatches by calibrated confidence score "
        "(Green >= 85%, Amber 70-85%, Orange 50-70%, Red < 50%) instead of distinct per-needle index colors."
    )
    self.colorByConfidenceCheckBox.toggled.connect(self.onColorByConfidenceToggled)

    self.colorByConfidenceLegendLabel = qt.QLabel(
        "<span style='color:#17c964;font-weight:600;'>&#9632; &ge;85%</span> &nbsp;"
        "<span style='color:#f5a524;font-weight:600;'>&#9632; 70-85%</span> &nbsp;"
        "<span style='color:#f3722c;font-weight:600;'>&#9632; 50-70%</span> &nbsp;"
        "<span style='color:#e5383b;font-weight:600;'>&#9632; &lt;50%</span>"
    )
    self.colorByConfidenceLegendLabel.setAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)

    colorByConfRow = qt.QFrame()
    colorByConfLayout = qt.QHBoxLayout(colorByConfRow)
    colorByConfLayout.setContentsMargins(0, 0, 0, 0)
    colorByConfLayout.addWidget(self.colorByConfidenceCheckBox, 1)
    colorByConfLayout.addWidget(self.colorByConfidenceLegendLabel, 0)
    segmentationFrame.addRow(colorByConfRow)

    self.newInsertionButton = None

    self.deleteNeedleButton = qt.QPushButton('Delete Last Needle [Ctrl + Z]')
    self.deleteNeedleButton.clicked.connect(lambda _=False: logic.deleteLastNeedle())
    self.deleteNeedleButton.setEnabled(0)

    self.resetDetectionButton = qt.QPushButton('Reset Detection')
    self.resetDetectionButton.clicked.connect(lambda _=False: logic.resetNeedleDetection())
    self.resetDetectionButton.setEnabled(0)

    segActionsRow = qt.QFrame()
    segActionsLayout = qt.QHBoxLayout(segActionsRow)
    segActionsLayout.setContentsMargins(0, 0, 0, 0)
    segActionsLayout.addWidget(self.deleteNeedleButton)
    segActionsLayout.addWidget(self.resetDetectionButton)
    segmentationFrame.addRow(segActionsRow)

    segmentationFrame.addRow(self.__reportFrame)

    # ------------------------------------------------------------------
    # 2. Manual Ground Truth & Validation
    # ------------------------------------------------------------------
    self.__validationFrame = ctk.ctkCollapsibleButton()
    self.__validationFrame.text = "2. Manual Segmentation && Evaluation"
    self.__validationFrame.collapsed = 0
    validationFrame = qt.QFormLayout(self.__validationFrame)

    self.startGivingControlPointsButton = qt.QPushButton('Start Giving Control Points')
    self.startGivingControlPointsButton.checkable = True
    self.startGivingControlPointsButton.setStyleSheet(toggle_ready_style)
    self.startGivingControlPointsButton.toggled.connect(self.onStartStopGivingValidationControlPointsToggled)

    self.startAssistModeButton = qt.QPushButton('Assisted Manual Segmentation')
    self.startAssistModeButton.checkable = True
    self.startAssistModeButton.setStyleSheet(secondary_toggle_style)
    self.startAssistModeButton.toggled.connect(self.onStartAssistModeToggled)

    self.validationNeedleButton = qt.QPushButton('Next Validation Needle: (0)->(1)')
    self.validationNeedleButton.toolTip = (
        "Increment the active manual needle index so subsequent clicks belong to the next needle."
    )
    self.validationNeedleButton.clicked.connect(lambda _=False: logic.validationNeedle())

    self.drawValidationNeedlesButton = qt.QPushButton('Render Manual Needle 0')
    self.drawValidationNeedlesButton.toolTip = (
        "Redraw manually segmented needles after adding or moving control points."
    )
    self.drawValidationNeedlesButton.clicked.connect(lambda _=False: logic.drawValidationNeedles())

    self.startValidationButton = qt.QPushButton('Start Evaluation (Auto vs. Manual)')
    self.startValidationButton.toolTip = "Launch tracking algorithm from the tips of all manually segmented needles and compute Hausdorff/tip/base accuracy."
    self.startValidationButton.clicked.connect(lambda _=False: logic.startValidation())
    self.startValidationButton.setStyleSheet(primary_action_style)

    self.resetValidationButton = qt.QPushButton('Reset Manual Seg.')
    self.templateRegistrationButton = qt.QPushButton('[Beta] Template Registration')

    self.hideAnnotationTextButton = qt.QPushButton('Hide Marker Texts')
    self.hideAnnotationTextButton.checkable = True
    self.hideAnnotationTextButton.setStyleSheet(secondary_toggle_style)

    self.undoButton = qt.QPushButton('Undo Point Move')
    self.undoButton.checkable = False

    self.resetValidationButton.clicked.connect(lambda _=False: logic.resetNeedleValidation())
    self.templateRegistrationButton.clicked.connect(lambda _=False: logic.autoregistration())
    self.hideAnnotationTextButton.clicked.connect(lambda _=False: logic.hideAnnotations())
    self.undoButton.clicked.connect(lambda _=False: logic.undoFid())

    self.editNeedleTxtBox = qt.QSpinBox()
    self.editNeedleTxtBox.valueChanged.connect(logic.changeValue)
    editLabel = qt.QLabel('Needle #:')

    self.configFrameCTL = qt.QFrame()
    self.configFrameCTL.setLayout(qt.QHBoxLayout())
    self.configFrameCTL.layout().setContentsMargins(0, 0, 0, 0)
    self.configFrameCTL.layout().addWidget(editLabel)
    self.configFrameCTL.layout().addWidget(self.editNeedleTxtBox)
    self.configFrameCTL.layout().addWidget(self.validationNeedleButton)
    self.configFrameCTL.layout().addWidget(self.drawValidationNeedlesButton)

    valModesRow = qt.QFrame()
    valModesLayout = qt.QHBoxLayout(valModesRow)
    valModesLayout.setContentsMargins(0, 0, 0, 0)
    valModesLayout.addWidget(self.startGivingControlPointsButton)
    valModesLayout.addWidget(self.startAssistModeButton)

    valToolsRow = qt.QFrame()
    valToolsLayout = qt.QHBoxLayout(valToolsRow)
    valToolsLayout.setContentsMargins(0, 0, 0, 0)
    valToolsLayout.addWidget(self.undoButton)
    valToolsLayout.addWidget(self.hideAnnotationTextButton)
    valToolsLayout.addWidget(self.resetValidationButton)

    validationFrame.layout().addRow(self.configFrameCTL)
    validationFrame.addRow(valModesRow)
    validationFrame.addRow(valToolsRow)
    validationFrame.addRow(self.startValidationButton)
    validationFrame.addRow(self.__reportFrameCTL)

    # ------------------------------------------------------------------
    # 3. Advanced Parameters & R&D (Collapsed by default for clean UX)
    # ------------------------------------------------------------------
    self.__parameterFrame = ctk.ctkCollapsibleButton()
    self.__parameterFrame.text = "Advanced Detection Parameters"
    self.__parameterFrame.collapsed = 1
    parameterFrame = qt.QFormLayout(self.__parameterFrame)

    self.configFrame = qt.QFrame()
    self.configFrame.setLayout(qt.QHBoxLayout())
    self.configFrame.layout().setContentsMargins(0, 0, 0, 0)
    parameterFrame.layout().addRow(self.configFrame)
    self.loadButton = qt.QPushButton()
    self.loadButton.text = "Load Parameters"
    self.loadButton.checkable = False
    self.loadButton.toolTip = "Click to load parameters from a configuration file."
    self.loadButton.clicked.connect(lambda _=False: self.onLoad())
    self.saveButton = qt.QPushButton()
    self.saveButton.checkable = False
    self.saveButton.text = "Save Parameters"
    self.saveButton.toolTip = "Click to save the parameters in a configuration file."
    self.saveButton.clicked.connect(lambda _=False: self.onSave())
    self.resetParametersButton = qt.QPushButton()
    self.resetParametersButton.checkable = False
    self.resetParametersButton.text = "Reset Default Parameters"
    self.resetParametersButton.toolTip = "Click to reset the default parameters from default.cfg"
    self.resetParametersButton.clicked.connect(lambda _=False: self.onResetParameters())
    self.configFrame.layout().addWidget(self.loadButton)
    self.configFrame.layout().addWidget(self.saveButton)
    self.configFrame.layout().addWidget(self.resetParametersButton)

    self.autoCorrectTip = qt.QCheckBox('Auto correct tip position?')
    parameterFrame.addRow(self.autoCorrectTip)
    self.autoCorrectTip.setChecked(0)

    self.invertedContrast = qt.QCheckBox('Search for bright needles (CT)?')
    parameterFrame.addRow(self.invertedContrast)
    self.gradient = qt.QCheckBox('Compute gradient?')
    self.gradient.setChecked(1)
    parameterFrame.addRow(self.gradient)

    self.filterControlPoints = qt.QCheckBox('Filter Control Points?')
    self.filterControlPoints.setChecked(0)

    self.drawFiducialPoints = qt.QCheckBox('Draw Control Points?')
    self.drawFiducialPoints.setChecked(0)
    parameterFrame.addRow(self.drawFiducialPoints)

    self.autoStopTip = qt.QCheckBox('Tracking in both directions')
    self.autoStopTip.setChecked(0)
    parameterFrame.addRow(self.autoStopTip)

    self.extendNeedle = qt.QCheckBox('Extend Needle')
    self.extendNeedle.setChecked(0)
    parameterFrame.addRow(self.extendNeedle)

    realNeedleLengthLabel = qt.QLabel('Real Needle Length (mm):')
    self.realNeedleLength = qt.QSpinBox()
    self.realNeedleLength.setMinimum(1)
    self.realNeedleLength.setMaximum(1500)
    self.realNeedleLength.setValue(240)
    parameterFrame.addRow(realNeedleLengthLabel, self.realNeedleLength)

    self.maxLength = qt.QCheckBox('Max Needle Length?')
    self.maxLength.setChecked(1)
    parameterFrame.addRow(self.maxLength)

    self.gaussianAttenuationButton = qt.QCheckBox('Add Gaussian Prob. Attenuation?')
    self.gaussianAttenuationButton.setChecked(1)
    parameterFrame.addRow(self.gaussianAttenuationButton)

    self.sigmaValue = qt.QSpinBox()
    self.sigmaValue.setMinimum(1)
    self.sigmaValue.setMaximum(500)
    self.sigmaValue.setValue(20)
    sigmaValueLabel = qt.QLabel("Sigma Value (exp(-x^2/(2*(sigma/10)^2))): ")
    parameterFrame.addRow(sigmaValueLabel, self.sigmaValue)

    self.gradientPonderation = qt.QSpinBox()
    self.gradientPonderation.setMinimum(1)
    self.gradientPonderation.setMaximum(500)
    self.gradientPonderation.setValue(5)
    gradientPonderationLabel = qt.QLabel("Gradient Ponderation: ")
    parameterFrame.addRow(gradientPonderationLabel, self.gradientPonderation)

    self.exponent = qt.QSpinBox()
    self.exponent.setMinimum(1)
    self.exponent.setMaximum(500)
    self.exponent.setValue(2)
    exponentLabel = qt.QLabel("Center Ponderation: ")
    parameterFrame.addRow(exponentLabel, self.exponent)

    self.nbPointsPerLine = qt.QSpinBox()
    self.nbPointsPerLine.setMinimum(2)
    self.nbPointsPerLine.setMaximum(500)
    self.nbPointsPerLine.setValue(20)

    self.nbRadiusIterations = qt.QSpinBox()
    self.nbRadiusIterations.setMinimum(2)
    self.nbRadiusIterations.setMaximum(1000)
    self.nbRadiusIterations.setValue(13)

    self.radiusMax = qt.QSpinBox()
    self.radiusMax.setMinimum(0)
    self.radiusMax.setMaximum(1000)
    self.radiusMax.setValue(5)
    distanceMaxLabel = qt.QLabel("Radius of cone base (mm): ")
    parameterFrame.addRow(distanceMaxLabel, self.radiusMax)

    self.nbRotatingIterations = qt.QSpinBox()
    self.nbRotatingIterations.setMinimum(2)
    self.nbRotatingIterations.setMaximum(1000)
    self.nbRotatingIterations.setValue(35)
    nbRotatingIterationsLabel = qt.QLabel("Number of rotating steps: ")
    parameterFrame.addRow(nbRotatingIterationsLabel, self.nbRotatingIterations)

    self.numberOfPointsPerNeedle = qt.QSpinBox()
    self.numberOfPointsPerNeedle.setMinimum(1)
    self.numberOfPointsPerNeedle.setMaximum(50)
    self.numberOfPointsPerNeedle.setValue(6)
    numberOfPointsPerNeedleLabel = qt.QLabel("Number of Control Points: ")
    parameterFrame.addRow(numberOfPointsPerNeedleLabel, self.numberOfPointsPerNeedle)

    self.stepsize = qt.QSpinBox()
    self.stepsize.setMinimum(1)
    self.stepsize.setMaximum(500)
    self.stepsize.setValue(5)

    self.lenghtNeedleParameter = qt.QSpinBox()
    self.lenghtNeedleParameter.setMinimum(1)
    self.lenghtNeedleParameter.setMaximum(10000)
    self.lenghtNeedleParameter.setValue(100)
    stepsizeLabel = qt.QLabel("Length of the needles (mm): ")
    parameterFrame.addRow(stepsizeLabel, self.lenghtNeedleParameter)

    self.radiusNeedleParameter = qt.QSpinBox()
    self.radiusNeedleParameter.setMinimum(1)
    self.radiusNeedleParameter.setMaximum(200)
    self.radiusNeedleParameter.setValue(2)
    radiusLabel = qt.QLabel("Radius of the needles (mm): ")
    parameterFrame.addRow(radiusLabel, self.radiusNeedleParameter)

    self.algoVersParameter = qt.QSpinBox()
    self.algoVersParameter.setMinimum(0)
    self.algoVersParameter.setMaximum(9)
    self.algoVersParameter.setValue(3)
    algoLabel = qt.QLabel("Needle detection version: ")
    parameterFrame.addRow(algoLabel, self.algoVersParameter)

    # Keep top-level combo box and developer spinbox bidirectionally synchronized
    def _on_algo_spin_changed(val):
      if 0 <= val < self.algoComboBox.count and self.algoComboBox.currentIndex != val:
        self.algoComboBox.setCurrentIndex(val)

    def _on_algo_combo_changed(idx):
      if idx >= 0 and self.algoVersParameter.value != idx:
        self.algoVersParameter.setValue(idx)

    self.algoVersParameter.valueChanged.connect(_on_algo_spin_changed)
    self.algoComboBox.currentIndexChanged.connect(_on_algo_combo_changed)

    self.__devFrame = ctk.ctkCollapsibleButton()
    self.__devFrame.text = "Obturators, Isosurfaces && R&&D Tools"
    self.__devFrame.collapsed = 1
    devFrame = qt.QFormLayout(self.__devFrame)

    self.fiducialObturatorButton = qt.QPushButton('Start Giving Obturator Needle Tips')
    self.fiducialObturatorButton.checkable = True
    self.fiducialObturatorButton.setStyleSheet(secondary_toggle_style)
    self.fiducialObturatorButton.toggled.connect(self.onStartStopGivingObturatorNeedleTipsToggled)

    self.renderObturatorNeedlesButton = qt.QPushButton('Render Obturator Needles')
    self.renderObturatorNeedlesButton.checkable = False
    self.renderObturatorNeedlesButton.clicked.connect(lambda _=False: self.logic.drawObturatorNeedles())

    self.displayFiducialButton = qt.QPushButton('Display Labels On Needles')
    self.displayFiducialButton.clicked.connect(lambda _=False: logic.displayFiducial())

    self.displayContourButton = qt.QPushButton('Draw Radiation Isosurfaces')
    self.displayContourButton.checkable = False
    self.displayContourButton.clicked.connect(lambda _=False: logic.drawIsoSurfaces())

    self.hideContourButton = qt.QPushButton('Hide Radiation Isosurfaces')
    self.hideContourButton.checkable = True
    self.hideContourButton.setStyleSheet(secondary_toggle_style)
    self.hideContourButton.clicked.connect(lambda _=False: logic.hideIsoSurfaces())
    self.hideContourButton.setEnabled(0)

    self.filterButton = qt.QPushButton('Preprocessing (SITK)')
    self.filterButton.checkable = False
    self.filterButton.clicked.connect(lambda _=False: logic.filterWithSITK())
    self.filterButton.setEnabled(1)

    self.parSearchButton = qt.QPushButton('Parameter Search')
    self.parSearchButton.checkable = False
    self.parSearchButton.clicked.connect(lambda _=False: logic.parSearch())
    self.parSearchButton.setEnabled(1)

    self.setAsValNeedlesButton = qt.QPushButton('Use Segmented Needles for Validation')
    self.setAsValNeedlesButton.checkable = False
    self.setAsValNeedlesButton.clicked.connect(lambda _=False: logic.setAllNeedleTubesAsValidationNeedles())
    self.setAsValNeedlesButton.setEnabled(1)
    self.setAsValNeedlesButton.setStyleSheet(primary_action_style)

    editorWidgetParent = slicer.qMRMLWidget()
    editorWidgetParent.setLayout(qt.QVBoxLayout())
    editorWidgetParent.setMRMLScene(slicer.mrmlScene)
    editorWidgetParent.hide()
    self.editorWidget = EditorWidget(editorWidgetParent, False)
    self.editUtil = self.editorWidget.editUtil
    self.currentLabel = None
    self.setWandEffectOptions()
    self.editUtil.setCurrentEffect("DefaultTool")
    self.editorWidget.setup()
    self.editorWidget.toolsBox.actions["NeedleFinder"] = qt.QAction(None)
    self.undoRedo = self.editorWidget.toolsBox.undoRedo
    self.currentLabel = self.editUtil.getLabel()
    self.editorWidget.editLabelMapsFrame.setText("Edit Segmentation")
    self.editorWidget.editLabelMapsFrame.contentsCollapsed.connect(self.onEditorCollapsed)
    editorWidgetParent.show()
    self.editUtil.setCurrentEffect("NeedleFinder")

    self.scenePath = qt.QLineEdit()
    self.scenePath.setPlaceholderText("Scene folder path for Clean Scene...")
    self.cleanSceneButton = qt.QPushButton('Clean Scene')
    self.cleanSceneButton.clicked.connect(lambda _=False: logic.cleanScene())

    obturatorRow = qt.QHBoxLayout()
    obturatorRow.setSpacing(6)
    obturatorRow.addWidget(self.fiducialObturatorButton, 1)
    obturatorRow.addWidget(self.renderObturatorNeedlesButton, 1)

    contourRow = qt.QHBoxLayout()
    contourRow.setSpacing(6)
    contourRow.addWidget(self.displayContourButton, 1)
    contourRow.addWidget(self.hideContourButton, 1)

    rdRow = qt.QHBoxLayout()
    rdRow.setSpacing(6)
    rdRow.addWidget(self.filterButton, 1)
    rdRow.addWidget(self.parSearchButton, 1)

    cleanSceneRow = qt.QHBoxLayout()
    cleanSceneRow.setSpacing(6)
    cleanSceneRow.addWidget(self.scenePath, 2)
    cleanSceneRow.addWidget(self.cleanSceneButton, 1)

    devFrame.addRow(obturatorRow)
    devFrame.addRow(contourRow)
    devFrame.addRow(self.setAsValNeedlesButton)
    devFrame.addRow(self.templateRegistrationButton)
    devFrame.addRow(rdRow)
    devFrame.addRow(cleanSceneRow)
    devFrame.addWidget(editorWidgetParent)

    self.layout.addRow(self.__segmentationFrame)
    self.layout.addRow(self.__validationFrame)
    self.layout.addRow(self.__parameterFrame)
    self.layout.addRow(self.__devFrame)

    resetButton = qt.QPushButton('Reset Module')
    resetButton.clicked.connect(lambda _=False: self.onReload())
    self.widget = slicer.qMRMLWidget()
    self.widget.setLayout(self.layout)
    self.layout2.addWidget(self.widget)

    self.initTableView()
    self.initTableViewControlPoints()

    vnJPG = getNode("Case *")
    if vnJPG:
      print("showing ground 2d image truth in green view")
      sw = slicer.app.layoutManager().sliceWidget("Green")
      cn = sw.mrmlSliceCompositeNode()
      cn.SetBackgroundVolumeID(vnJPG.GetID())
      slicer.app.layoutManager().sliceWidget("Green").sliceLogic().GetBackgroundLayer().Modified()
      sGreen = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeGreen") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")
      sGreen.SetSliceVisible(0)
      sGreen.SetOrientationToAxial()
      sw.fitSliceToBackground()
      sGreen.Modified()

    self.onResetParameters()
    self.setupShortcuts()

  def setWandEffectOptions(self, tolerance=20, maxPixels=200, fillMode="Volume"):
    """Set the wand logic parameters in parameter node."""
    profprint()
    parameterNode = self.editUtil.getParameterNode()
    parameterNode.SetParameter("WandEffect,tolerance", str(tolerance))
    parameterNode.SetParameter("WandEffect,maxPixels", str(maxPixels))
    parameterNode.SetParameter("WandEffect,fillMode", fillMode)
    wandOpt = EditorLib.WandEffectOptions()
    wandOpt.setMRMLDefaults()
    wandOpt.__del__()

  def keyPressEvent(self, event):
    print("You Pressed: " + event.text())

  def setupShortcuts(self):
    """Set up hot keys for various actions."""
    profprint()
    macros = (
        ("Ctrl+Return", self.segmentNeedle),
        ("Ctrl+z", self.logic.deleteLastNeedle),
        ("Ctrl+y", self.acceptNeedleTipEstimate),
        ("Ctrl+n", self.rejectNeedleTipEstimate),
        ("Ctrl+u", self.acceptNeedleTipEstimateAsNewTempMarker),
    )

    mainWindow = slicer.util.mainWindow()
    for keys, f in macros:
      if mainWindow:
        k = qt.QKeySequence(keys)
        s = qt.QShortcut(k, mainWindow)
        s.activated.connect(f)
        s.activatedAmbiguously.connect(f)
      print("'%s' -> '%s'" % (keys, f.__name__))
      globals()['nfw'] = slicer.modules.NeedleFinderWidget
      globals()['nfl'] = slicer.modules.NeedleFinderWidget.logic
      print("nfl -> NeedleFinderLogic")
      print("nfw -> NeedleFinderWidget")

  def segmentNeedle(self):
    """Helper function for Ctrl+Enter."""
    profprint()
    if self.fiducialButton.isEnabled():
      print("new checked state: ", not self.fiducialButton.checked)
      self.onStartStopGivingNeedleTipsToggled(not self.fiducialButton.checked)

  def rejectNeedleTipEstimate(self):
    """Helper function for Ctrl+n: delete tip estimate and jump back to temp marker."""
    profprint()
    tempFidNodes = slicer.mrmlScene.GetNodesByName('.tip? [Ctrl+y/n/u]')
    for i in range(tempFidNodes.GetNumberOfItems()):
      node = tempFidNodes.GetItemAsObject(i)
      if node:
        slicer.mrmlScene.RemoveNode(node)
    tempFidNodes = slicer.mrmlScene.GetNodesByName('.temp')
    rasTemp = [0.0, 0.0, 0.0]
    for i in range(tempFidNodes.GetNumberOfItems()):
      node = tempFidNodes.GetItemAsObject(i)
      if node:
        get_fiducial_coords(node, rasTemp)
    slRed = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
    slYel = slicer.app.layoutManager().sliceWidget("Yellow").sliceLogic()
    slGrn = slicer.app.layoutManager().sliceWidget("Green").sliceLogic()
    slRed.SetSliceOffset(rasTemp[2])
    slYel.SetSliceOffset(rasTemp[0])
    self.logic.resetCoronalSegment()
    slGrn.SetSliceOffset(rasTemp[1])
    node = getNode('*.tempN')
    if node:
      slicer.mrmlScene.RemoveNode(node)

  def acceptNeedleTipEstimate(self):
    """Helper function for Ctrl+y: delete temp and estimated tip markers and start needle detection from estimated tip."""
    profprint()
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    imageData = volumeNode.GetImageData()
    spacing = volumeNode.GetSpacing()
    colorVar = random.randrange(50, 100, 1)
    tempFidNodes = slicer.mrmlScene.GetNodesByName('.tip? [Ctrl+y/n/u]')
    coord = [0.0, 0.0, 0.0]
    for i in range(tempFidNodes.GetNumberOfItems()):
      node = tempFidNodes.GetItemAsObject(i)
      if node:
        get_fiducial_coords(node, coord)
    ijk = self.logic.ras2ijk(coord)
    if sum(coord):
      self.logic.needleDetectionThread([ijk], imageData, spacing=spacing)
    if self.autoStopTip.isChecked():
      self.logic.needleDetectionUPThread(ijk, imageData, colorVar, spacing)
    print("deleting temp marker and segmentation")
    for name in ('.temp', '.tip? [Ctrl+y/n/u]'):
      tempFidNodes = slicer.mrmlScene.GetNodesByName(name)
      for i in range(tempFidNodes.GetNumberOfItems()):
        node = tempFidNodes.GetItemAsObject(i)
        if node:
          slicer.mrmlScene.RemoveNode(node)
    self.tempPointList = []
    self.logic.resetCoronalSegment()
    node = getNode('*.tempN')
    if node:
      slicer.mrmlScene.RemoveNode(node)

  def acceptNeedleTipEstimateAsNewTempMarker(self):
    """Helper function for Ctrl+u: delete temp and estimated tip markers and start needle up detection again from estimated tip."""
    profprint()
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    imageData = volumeNode.GetImageData()
    spcg = volumeNode.GetSpacing()
    org = volumeNode.GetOrigin()
    colorVar = random.randrange(50, 100, 1)
    tempFidNodes = slicer.mrmlScene.GetNodesByName('.tip? [Ctrl+y/n/u]')
    coord = [0.0, 0.0, 0.0]
    for i in range(tempFidNodes.GetNumberOfItems()):
      node = tempFidNodes.GetItemAsObject(i)
      if node:
        get_fiducial_coords(node, coord)
        ijk = self.logic.ras2ijk(coord)
        rasPrevTip = self.logic.ijk2ras(ijk)
    node = getNode('*.tempN')
    if node:
      slicer.mrmlScene.RemoveNode(node)
    if sum(coord):
      self.logic.controlPoints = []
      ijkTipEstimate = self.logic.needleDetectionUPThread(ijk, imageData, colorVar, spcg, tipOnly=False, strName=".tempN", script=True)
      rasTipEstimate = self.logic.ijk2ras(ijkTipEstimate)
    print("moving temp markers to new segment")
    tempFidNodes = slicer.mrmlScene.GetNodesByName('.tip? [Ctrl+y/n/u]')
    ras = self.logic.ijk2ras(ijkTipEstimate)
    for i in range(tempFidNodes.GetNumberOfItems()):
      node = tempFidNodes.GetItemAsObject(i)
      if node:
        set_fiducial_coords(node, ras)
    tempFidNodes = slicer.mrmlScene.GetNodesByName('.temp')
    for i in range(tempFidNodes.GetNumberOfItems()):
      node = tempFidNodes.GetItemAsObject(i)
      if node:
        set_fiducial_coords(node, rasPrevTip)
        self.tempPointList.append(rasPrevTip)
        print("tempPointList: ", self.tempPointList)
    slRed = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
    slYel = slicer.app.layoutManager().sliceWidget("Yellow").sliceLogic()
    node = getNode('*.tempN')
    if node:
      self.logic.reformatCoronalView4NeedleSegment(rasPrevTip, rasTipEstimate)
    kx = 1 + ijkTipEstimate[2]
    print("z slice off.: ", (kx - 1) * spcg[2] + org[2])
    slRed.SetSliceOffset((kx - 1) * spcg[2] + org[2])
    print("x slice off.: ", org[0] - ijkTipEstimate[0] * spcg[0])
    slYel.SetSliceOffset(org[0] - ijkTipEstimate[0] * spcg[0])

  def cleanup(self):
    """Clean up memory."""
    profprint()
    self.removeObservers()

  def refreshObservers(self):
    """When the layout changes, drop the observers from all the old widgets and create new observers for the newly created widgets."""
    profprint()
    self.removeObservers()
    layoutManager = slicer.app.layoutManager()
    sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    for nodeIndex in range(sliceNodeCount):
      sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
      sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
      if sliceWidget:
        style = sliceWidget.sliceView().interactorStyle()
        self.sliceWidgetsPerStyle[style] = sliceWidget
        events = ("LeftButtonPressEvent", "RightButtonPressEvent", "KeyPressEvent", "KeyReleaseEvent")
        for event in events:
          tag = style.AddObserver(event, self.processEvent)
          self.styleObserverTags.append([style, tag])

  def onReload(self, moduleName="NeedleFinder"):
    """Generic reload method for any scripted module."""
    if profiling:
      profbox()
    import importlib
    import sys
    reload_order = [
        "NeedleFinderLib.Constants",
        "NeedleFinderLib.DeflectionModelData",
        "NeedleFinderLib.ProfilingUtils",
        "NeedleFinderLib.LegacyEditorCompat",
        "NeedleFinderLib.RustCore",
        "NeedleFinderLib.GeometryUtils",
        "NeedleFinderLib.AssistedSegmentation",
        "NeedleFinderLib.SceneAndEvaluation",
        "NeedleFinderLib.TrackingAlgorithms",
        "NeedleFinderLib",
    ]
    for mod_name in reload_order:
      if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)

  def onReloadAndTest(self, moduleName="NeedleFinder"):
    """Generic reload and test method for any scripted module."""
    print("onReloadAndTest")
    msgbox(whoami())
    try:
      self.onReload()
      evalString = 'globals()["%s"].%sTest()' % (moduleName, moduleName)
      tester = eval(evalString)
      tester.runTest()
    except Exception as e:
      import traceback
      traceback.print_exc()
      qt.QMessageBox.warning(
          slicer.util.mainWindow(),
          "Reload and Test",
          'Exception!\n\n' + str(e) + "\n\nSee Python Console for Stack Trace",
      )

  def onStartStopGivingNeedleTipsToggled(self, checked=True):
    """Start/stop giving needle tips."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    self.fiducialButton.checked = checked
    if checked:
      self.startGivingControlPointsButton.checked = 0
      self.fiducialObturatorButton.checked = 0
      self.start()
      self.fiducialButton.text = "2. Stop Giving Needle Tips [CTRL + ENTER]"
      widget.editUtil.setCurrentEffect("NeedleFinder")
    else:
      self.stop()
      self.fiducialButton.text = "2. Start Giving Needle Tips [CTRL + ENTER]"
      widget.editUtil.setCurrentEffect("DefaultTool")
      widget.resetDetectionButton.setEnabled(1)
      tempFidNodes = slicer.mrmlScene.GetNodesByName('.temp')
      for i in range(tempFidNodes.GetNumberOfItems()):
        node = tempFidNodes.GetItemAsObject(i)
        if node:
          slicer.mrmlScene.RemoveNode(node)
    widget.deleteNeedleButton.setEnabled(1)

  def onSkipSegLimit(self):
    """Skip providing segmentation limit plane, as already in scene."""
    profprint()
    self.logic.placeAxialLimitMarker(assign=False)

  def onAutoDetectAllNeedlesClicked(self, _=False):
    """Automatically detect and segment all needles in the active volume."""
    profprint()
    if self.fiducialButton.checked:
      self.fiducialButton.checked = 0
    min_conf = float(self.minConfidenceSpinBox.value)
    self.logic.autoDetectAllNeedles(minConfidence=min_conf)

  def onColorByConfidenceToggled(self, checked):
    """Live-toggle needle 3D model and table swatch colors between confidence colormap and per-needle index colors."""
    profprint()
    self.logic.updateNeedleColorsByConfidence(bool(checked))

  def onStartStopGivingObturatorNeedleTipsToggled(self, checked):
    """Start/stop giving obturator needle tips."""
    profprint()
    if checked:
      self.fiducialButton.checked = 0
      self.fiducialButton.text = "2. Start Giving Needle Tips [CTRL + ENTER]"
      self.startGivingControlPointsButton.checked = 0
      self.start(self.obturatorNeedleTipClicks)
      self.fiducialObturatorButton.text = "Stop Giving Obturator Needle Tips"
    else:
      self.stop()
      self.fiducialObturatorButton.text = "Start Giving Obturator Needle Tips"

  def onStartStopGivingValidationControlPointsToggled(self, checked):
    """Start/stop needle validation control points."""
    profprint()
    if checked:
      self.fiducialObturatorButton.checked = 0
      self.fiducialButton.checked = 0
      self.fiducialButton.text = "2. Start Giving Needle Tips [CTRL + ENTER]"
      self.start(self.needleValidationClicks)
      self.startGivingControlPointsButton.text = "Stop Giving Control Points"
    else:
      self.stop()
      self.startGivingControlPointsButton.text = "Start Giving Control Points"

  def onStartAssistModeToggled(self, checked):
    """Start/stop assisted manual segmentation mode."""
    profprint()
    if checked:
      self.fiducialObturatorButton.checked = 0
      self.fiducialButton.checked = 0
      self.fiducialButton.text = "2. Start Giving Needle Tips [CTRL + ENTER]"
      self.start(self.addCTLPoints)
      self.startAssistModeButton.text = "Stop Assisted Manual Segmentation"
    else:
      self.stop()
      self.startAssistModeButton.text = "Start Assisted Manual Segmentation"

  def start(self, process=0):
    """Start observing the mouse clicks given by user."""
    profprint()
    logic = self.logic
    logic.changeCursor(1)
    self.removeObservers()
    layoutManager = slicer.app.layoutManager()
    sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    for nodeIndex in range(sliceNodeCount):
      sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
      sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
      if sliceWidget:
        style = sliceWidget.sliceView().interactorStyle()
        self.sliceWidgetsPerStyle[style] = sliceWidget
        events = ("LeftButtonPressEvent", "RightButtonPressEvent", "EnterEvent", "LeaveEvent", "KeyPressEvent", "KeyReleaseEvent")
        for event in events:
          if process == self.needleValidationClicks:
            tag = style.AddObserver(event, self.processEventNeedleValidation)
          elif process == self.addManualTipClicks:
            tag = style.AddObserver(event, self.processEventAddManualTips)
          elif process == self.addCTLPoints:
            tag = style.AddObserver(event, self.processEventAddCTLPoints)
          elif process == self.obturatorNeedleTipClicks:
            tag = style.AddObserver(event, self.processEventAddObturatorNeedleTips)
            dn = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode().GetDisplayNode()
            w = dn.GetWindow()
            l = dn.GetLevel()
            dn.AddObserver(vtk.vtkCommand.ModifiedEvent, lambda c, e: logic.setWL(dn, w, l))
          else:
            tag = style.AddObserver(event, self.processEvent)
          self.styleObserverTags.append([style, tag])

  def stop(self):
    """Stop observing the mouse clicks given by user."""
    profprint()
    self.logic.changeCursor(0)
    self.removeObservers()
    self.fiducialObturatorButton.checked = 0
    self.fiducialButton.checked = 0
    self.validationNeedleButton.checked = 0

  def removeObservers(self):
    """Remove observers and reset."""
    profprint()
    for observee, tag in self.styleObserverTags:
      observee.RemoveObserver(tag)
    self.styleObserverTags = []
    self.sliceWidgetsPerStyle = {}

  def processEvent(self, observee, event=None):
    """Observe events in the needle segmentation mode."""
    if frequent:
      profprint()
    widget = slicer.modules.NeedleFinderWidget
    logic = slicer.modules.NeedleFinderWidget.logic
    ras = [0.0, 0.0, 0.0]
    if self.CrosshairNode is None or self.CrosshairNode.GetScene() != slicer.mrmlScene:
      self.CrosshairNode = slicer.mrmlScene.GetNthNodeByClass(0, 'vtkMRMLCrosshairNode')
    if self.CrosshairNode:
      self.CrosshairNode.GetCursorPositionRAS(ras)

    if observee in self.sliceWidgetsPerStyle:
      sliceWidget = self.sliceWidgetsPerStyle[observee]
      sliceLogic = sliceWidget.sliceLogic()
      interactor = observee.GetInteractor()
      key = interactor.GetKeySym()

      if event == "KeyReleaseEvent" and key == 'Shift_L' or key == 'Shift_R':
        tempFidNodes = slicer.mrmlScene.GetNodesByName('.temp')
        if tempFidNodes.GetNumberOfItems() > 0:
          for i in range(tempFidNodes.GetNumberOfItems()):
            node = tempFidNodes.GetItemAsObject(i)
            if node:
              set_fiducial_coords(node, ras)
              self.tempPointList.append(ras)
              print("tempPointList: ", self.tempPointList)
            if sliceLogic not in self.wandLogics:
              if not self.labelMapNode:
                self.createAddOrSelectLabelMapNode()
              print("creating new segment logic")
              wl = EditorLib.WandEffectLogic(sliceLogic)
              wl.undoRedo = self.undoRedo
              wl.editUtil = self.editUtil
              self.wandLogics[sliceLogic] = wl
            print("tracking needle upwards")
            self.setWandEffectOptions()
            wl = self.wandLogics[sliceLogic]
            xy = interactor.GetEventPosition()
            print("xy: ", xy)
            if wl.labelAtXY(xy):
              self.editUtil.setLabel(wl.labelAtXY(xy))
            else:
              print("new label")
              self.currentLabel += 1
              self.editUtil.setLabel(self.currentLabel)
            slRed = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
            slYel = slicer.app.layoutManager().sliceWidget("Yellow").sliceLogic()
            volumeNode = slRed.GetBackgroundLayer().GetVolumeNode()
            imageData = volumeNode.GetImageData()
            labelImage = self.labelMapNode.GetImageData()
            ijk = self.logic.ras2ijk(ras)
            ixStart, jxStart, kxStart = ijk[0], ijk[1], ijk[2]
            shape = list(labelImage.GetDimensions())
            spcg = self.labelMapNode.GetSpacing()
            org = self.labelMapNode.GetOrigin()
            if widget.algoVersParameter.value > 4:
              print("wanding")
              self.wandLogics[sliceLogic].apply(xy)
              print("shape: ", shape)
              shape.reverse()
              ijkMid = None
              labelArray = vtk.util.numpy_support.vtk_to_numpy(labelImage.GetPointData().GetScalars()).reshape(shape)
              for kx in range(max(int(kxStart) - 0, 0), min(int(kxStart + 20 / spcg[2]), shape[0])):
                print("kx", kx)
                ijkTipEstimate = ijkMid
                ijkMid = np.array([0, 0, 0])
                midPtCtr = 0
                for ix in range(max(int(ixStart - 10 / spcg[0]), 0), min(int(ixStart + 10 / spcg[0]), shape[2])):
                  for jx in range(max(int(jxStart - 10 / spcg[1]), 0), min(int(jxStart + 10 / spcg[1]), shape[1])):
                    if labelArray[kx, jx, ix] == self.currentLabel:
                      print("curLab, labelArr[x]= ", self.currentLabel, labelArray[kx, jx, ix])
                      ijkMid += [ix, jx, kx]
                      midPtCtr += 1
                    if labelArray[kx, jx, ix] and labelArray[kx, jx, ix] != self.currentLabel:
                      labelArray[kx, jx, ix] = 0
                if not midPtCtr:
                  print("empty slice found ijkTipEstimate=", ijkTipEstimate)
                  break
                else:
                  print("non-empty slice found")
                  ijkMid = ijkMid / float(midPtCtr)
            else:
              colorVar = random.randrange(50, 100, 1)
              ijk = self.logic.ras2ijk(ras)
              node = getNode('*.tempN')
              if node:
                slicer.mrmlScene.RemoveNode(node)
              print("needle detection upwards")
              self.logic.controlPoints = []
              ijkTipEstimate = logic.needleDetectionUPThread(ijk, imageData, colorVar, spcg, tipOnly=False, strName=".tempN", script=True)
              rasTipEstimate = logic.ijk2ras(ijkTipEstimate)
              kx = 1 + ijkTipEstimate[2]
              node = getNode('*.tempN')
              if node:
                logic.reformatCoronalView4NeedleSegment(ras, rasTipEstimate)
            print("z slice off.: ", (kx - 1) * spcg[2] + org[2])
            slRed.SetSliceOffset((kx - 1) * spcg[2] + org[2])
            print("x slice off.: ", org[0] - ijkTipEstimate[0] * spcg[0])
            slYel.SetSliceOffset(org[0] - ijkTipEstimate[0] * spcg[0])
            tempFidNodes = slicer.mrmlScene.GetNodesByName('.tip? [Ctrl+y/n/u]')
            for i in range(tempFidNodes.GetNumberOfItems()):
              node = tempFidNodes.GetItemAsObject(i)
              if node:
                set_fiducial_coords(node, logic.ijk2ras(ijkTipEstimate))
            if not tempFidNodes.GetNumberOfItems():
              create_fiducial_node(
                  name='.tip? [Ctrl+y/n/u]',
                  ras=logic.ijk2ras(ijkTipEstimate),
                  color=(0.0, 1.0, 0.0),
                  glyph_scale=2.0,
                  text_scale=4.0,
                  locked=False,
                  attributes={'TemporaryTip': '1'},
              )
        else:
          create_fiducial_node(
              name='.temp',
              ras=ras,
              color=(1.0, 1.0, 0.0),
              glyph_scale=2.0,
              text_scale=4.0,
              locked=True,
              attributes={'TemporaryFiducial': '1'},
          )

      elif event == "LeftButtonPressEvent":
        self.logic.t0 = time.perf_counter()
        colorVar = random.randrange(50, 100, 1)
        volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
        imageData = volumeNode.GetImageData()
        spacing = volumeNode.GetSpacing()
        ijk = self.logic.ras2ijk(ras)
        print('spacing: ', spacing)
        self.logic.needleDetectionThread([ijk], imageData, spacing=np.array(spacing))
        if self.autoStopTip.isChecked():
          self.logic.needleDetectionUPThread(ijk, imageData, colorVar, spacing)
        print("deleting temp marker and segmentation")
        for name in ('.temp', '.tip? [Ctrl+y/n/u]'):
          tempFidNodes = slicer.mrmlScene.GetNodesByName(name)
          for i in range(tempFidNodes.GetNumberOfItems()):
            node = tempFidNodes.GetItemAsObject(i)
            if node:
              slicer.mrmlScene.RemoveNode(node)
        self.tempPointList = []

  def clearLabelMap(self, label=None):
    """Erase the contents of the label map."""
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    print("clearing label map")
    self.undoRedo.saveState()
    labelImage = self.labelMapNode.GetImageData()
    shape = list(labelImage.GetDimensions())
    shape.reverse()
    labelArray = vtk.util.numpy_support.vtk_to_numpy(labelImage.GetPointData().GetScalars()).reshape(shape)
    if not label:
      labelArray[:] = 0
    else:
      labelArray[labelArray == label] = 0
    self.editUtil.markVolumeNodeAsModified(widget.labelMapNode)

  def processEventNeedleValidation(self, observee, event=None):
    """Get the mouse clicks and create a fiducial node at this position."""
    if frequent:
      profprint()
    if observee in self.sliceWidgetsPerStyle and event == "LeftButtonPressEvent":
      sliceWidget = self.sliceWidgetsPerStyle[observee]
      interactor = observee.GetInteractor()
      xy = interactor.GetEventPosition()
      xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
      ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

      ijk = self.logic.ras2ijk(ras)

      self.logic.t0 = time.perf_counter()
      widget = slicer.modules.NeedleFinderWidget
      widget.stepNeedle += 1
      self.logic.placeNeedleShaftEvalMarker(ijk, widget.editNeedleTxtBox.value, self.logic.findNextStepNumber(widget.editNeedleTxtBox.value))
      self.logic.drawValidationNeedles()

  def processEventAddCTLPoints(self, observee, event=None):
    """Get the mouse clicks and create a fiducial node at this position."""
    if frequent:
      profprint()
    if observee in self.sliceWidgetsPerStyle and event == "LeftButtonPressEvent":
      sliceWidget = self.sliceWidgetsPerStyle[observee]
      interactor = observee.GetInteractor()
      xy = interactor.GetEventPosition()
      xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
      ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

      needleNumber = self.logic.assignNeedle(ras)
      print('needlenumber to assign: ', needleNumber)
      if needleNumber is None:
        needleNumber = max(self.logic.findNeedles(all=1)) + 1

      ijk = self.logic.ras2ijk(ras)

      self.logic.t0 = time.perf_counter()
      widget = slicer.modules.NeedleFinderWidget
      widget.stepNeedle += 1
      self.logic.placeNeedleShaftEvalMarker(ijk, needleNumber, self.logic.findNextStepNumber(needleNumber))
      self.logic.drawValidationNeedles(needleNumber)

  def processEventAddObturatorNeedleTips(self, observee, event=None):
    """Get the mouse clicks and create an obturator fiducial node at this position."""
    profprint()
    if observee in self.sliceWidgetsPerStyle and event == "LeftButtonPressEvent":
      sliceWidget = self.sliceWidgetsPerStyle[observee]
      interactor = observee.GetInteractor()
      xy = interactor.GetEventPosition()
      xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
      ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

      colorVar = random.randrange(50, 100, 1)
      volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
      imageData = volumeNode.GetImageData()
      spacing = volumeNode.GetSpacing()
      ijk = self.logic.ras2ijk(ras)
      self.logic.t0 = time.perf_counter()
      self.logic.obturatorNeedle(ijk, imageData, colorVar, spacing)
      self.obtuNeedle += 1

  def processEventAddManualTips(self, observee, event=None):
    """Get the mouse clicks and create a manual tip fiducial node at this position."""
    profbox()
    if observee in self.sliceWidgetsPerStyle and event == "LeftButtonPressEvent":
      sliceWidget = self.sliceWidgetsPerStyle[observee]
      interactor = observee.GetInteractor()
      xy = interactor.GetEventPosition()
      xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
      ras = sliceWidget.sliceView().convertXYZToRAS(xyz)
      self.logic.addManualTip(ras)

  def onSave(self):
    """Show file dialog to save parameters."""
    profprint()
    self.dirDialog = qt.QFileDialog(self.parent)
    self.dirDialog.setDirectory(slicer.modules.needlefinder.path.replace("NeedleFinder.py", "Config"))
    self.dirDialog.options = self.dirDialog.DontUseNativeDialog
    self.dirDialog.acceptMode = self.dirDialog.AcceptSave
    self.dirDialog.defaultSuffix = "cfg"
    self.dirDialog.setNameFilter("Configuration file (*.cfg)")
    self.dirDialog.fileSelected.connect(self.saveFileSelected)
    self.dirDialog.show()

  def saveFileSelected(self, fileName):
    """Save parameters."""
    profprint()
    self.logic.saveParameters(fileName)

  def onLoad(self):
    """Show file dialog to load parameter file."""
    profprint()
    self.dirDialog = qt.QFileDialog(self.parent)
    self.dirDialog.setDirectory(slicer.modules.needlefinder.path.replace("NeedleFinder.py", "Config"))
    self.dirDialog.options = self.dirDialog.DontUseNativeDialog
    self.dirDialog.acceptMode = self.dirDialog.AcceptOpen
    self.dirDialog.defaultSuffix = "cfg"
    self.dirDialog.setNameFilter("Configuration File (*.cfg)")
    self.dirDialog.fileSelected.connect(self.onLoadFileSelected)
    self.dirDialog.show()

  def onLoadFileSelected(self, fileName):
    """Load parameters."""
    profprint()
    self.logic.loadParameters(fileName)

  def onResetParameters(self):
    """Load default parameter file."""
    profprint()
    fileName = slicer.modules.needlefinder.path.replace("NeedleFinder.py", "Config/default.cfg")
    self.logic.loadParameters(fileName)

"""

########################################################################################################################
NEEDLEFINDER LOGIC
########################################################################################################################

"""

class NeedleFinderLogic(
    ScriptedLoadableModuleLogic,
    TrackingAlgorithmsMixin,
    SceneAndEvaluationMixin,
    AssistedSegmentationMixin,
):
  """
  This class implements all the actual
  computation done by the module.  The interface
  is such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget
  """
  # productive

  def __init__(self):
    """
    init's the class
    """
    # productive
    profprint()
    ScriptedLoadableModuleLogic.__init__(self)
    # initialisation of global variables
    self.option = self.setLabels()
    self.color = self.setColors()
    self.color255 = self.setColors255()
    self.p = self.setHolesCoordinates()
    self.t0 = 0

    self.row = 0
    self.col = 0
    self.round = 1
    self.fiducialNode = None
    # widget.stepNeedle         = 0
    self.ptNumber = 0
    self.table = None
    self.tableCTL = None
    self.view = None
    self.viewCTL = None
    self.model = None
    self.modelCTL = None
    self.contourNode = None
    self.lastNeedleNames = []
    self.enableScreenshots = 0
    self.screenshotScaleFactor = 1
    self.estimatorReference = 0
    self.controlPoints = []
    self.observerTags = {}
    self.observeManualNeedles()
    self.lastTime = time.time()

    self.previousValues = [[0, 0, 0]]
    self.tableValueCtrPt = [[[999, 999, 999] for i in range(100)] for j in range(100)]


  def getName(self):
    """
    return class name
    """
    return self.__class__.__name__

  def loadCTLPointsInTable(self):

    # get list of fiducial points
    path = slicer.mrmlScene.GetRootDirectory()
    fname = os.listdir(path)
    print(fname)
    pointList=[]

    nodes = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLAnnotationHierarchyNode','Fiducials List')
    addPointBool = 0
    if nodes.GetNumberOfItems()==0:
      addPointBool = 1
    elif nodes.GetNumberOfItems()==1:
      l = nodes.GetItemAsObject(0)
      v = vtk.vtkCollection()
      l.GetAssociatedChildrenNodes(v)
      print('number of children: ', v.GetNumberOfItems())
      if v.GetNumberOfItems()==0:
        addPointBool = 1
    print('*'*50)
    print('%d list(s) of fiducials found with the same name - add point bool: %d' % (int(nodes.GetNumberOfItems()), addPointBool))
    print('*'*50)

    for file in fname:
        if fnmatch.fnmatch(file, '*.acsv'):
            with open(os.path.join(path, file)) as f:
                content = f.readlines()
                s = content[-1]
                sArray = s.split('|')
                fs = file.split('.')
                fss = fs[1].split('-')
                if len(fss)>1:
                  try:
                    coord = np.array([int(fss[0]),int(fss[1]),float(sArray[1]),float(sArray[2]),float(sArray[3])])
                    pointList.append(coord)
                  except Exception:
                    print(fss)
    pointList = np.array(pointList)

    self.deleteDuplicateMarkers()

    for point in pointList:
      self.placeNeedleShaftEvalMarker(point[2:], int(point[0]),int(point[1]), type = 'ras', createPoint = addPointBool )

    self.deleteDuplicateNeedles()

    self.renameObjects() # rename if \r was added

    # observe visibility of manual needles to propagate on ctl points
    self.observeManualNeedles()

    widget = slicer.modules.NeedleFinderWidget
    widget.viewCTL.sortByColumn(2, qt.Qt.AscendingOrder)
    widget.viewCTL.sortByColumn(1, qt.Qt.AscendingOrder)

    self.lockControlPoints(widget.editNeedleTxtBox.value)
    self.unlockControlPoints(widget.editNeedleTxtBox.value)

  def observeFiducialPoints(self):
    '''
    Add observers to all fiducial points to trigger an event if modified
    :return:
    '''
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    nodes = slicer.util.getNodes('.*-*')
    for node in list(nodes.values()):
      nth = node.GetName()
      val = widget.observerTagsFid.get(nth)
      if val:
        node.RemoveObserver(val)
        del widget.observerTagsFid[nth]

    for node in list(nodes.values()):
      if is_fiducial_node(node):
        nth = node.GetName()
        widget.observerTagsFid[nth] = add_fiducial_modified_observer(node, self.addToUndoList)

  def observeSingleFiducial(self, node, event):
    '''
    Add observer to a single fiducial point to trigger an event if modified
    :param node:
    :param event:
    :return:
    '''
    try:
      widget = slicer.modules.NeedleFinderWidget
      if is_fiducial_node(node):
          nth = node.GetName()
          widget.observerTagsFid[nth] = add_fiducial_modified_observer(node, self.addToUndoList)
    except Exception:
      pass


  def addToUndoList(self, modifiedNode, event):
    '''
    Add node name and node coord to the undo list
    :param modifiedNode:
    :param event:
    :return:
    '''
    widget = slicer.modules.NeedleFinderWidget
    if time.time() - self.lastTime > 1:
      d = {
          'name': modifiedNode.GetName(),
          'coord': get_fiducial_coords(modifiedNode),
      }
      if d != widget.undoListFid[-1]:
        widget.undoListFid.append(d)
        widget.undoListFid.pop(0)
        self.lastTime = time.time()

  def undoFid(self):
    '''
    Revert the last modified fiducial point to its previously recorded position
    :return:
    '''
    profprint()
    widget = slicer.modules.NeedleFinderWidget
    print('..Undoing fiducial point mvt for pt ', widget.undoListFid[-1].get('name'))
    d = widget.undoListFid[-1]
    if d.get('name'):
      coord = d['coord']
      node = getNode(d['name'])
      set_fiducial_coords(node, coord)
    widget.undoListFid.pop(-1)
    widget.undoListFid.insert(0,{})



  def observeManualNeedles(self):

    nodes = slicer.util.getNodes('manual-seg_*')
    for node in list(nodes.values()):
      nth = node.GetAttribute('nth')
      vals = self.observerTags.get(nth)
      if vals != None:
        dNode = node.GetDisplayNode()
        for val in vals:
          dNode.RemoveObserver(val)
        del self.observerTags[nth]

    for node in list(nodes.values()):
      if node != None and node.GetClassName() == 'vtkMRMLModelNode':
        dNode = node.GetDisplayNode()
        nth = node.GetAttribute('nth')
        dNode.SetDescription(nth)
        if self.observerTags.get(nth) == None:
          self.observerTags[nth] = [dNode.AddObserver('ModifiedEvent', self.followVisibility)]
        else:
          self.observerTags[nth].append(dNode.AddObserver('ModifiedEvent', self.followVisibility))

  def followVisibility(self, dn, eventId ): #needleModelNode=0, displayNode=0,
    '''
    propagate visibility of needle to control points
    :param displayNodeID:
    :return:
    '''
    nth = dn.GetDescription()
    v = dn.GetVisibility()
    print(v)
    print('.' + nth +'-*')
    nodes = slicer.util.getNodes('.' + nth +'-*')
    for node in list(nodes.values()):
      print(node.GetID())
      if is_fiducial_node(node):
        node.SetDisplayVisibility(v)


  def hasImageData(self, volumeNode):
    """ Test:
    This is a dummy logic method that
    returns true if the passed in volume
    node has valid image data
    """
    # test
    profprint()
    if not volumeNode:
      print('no volume node')
      return False
    if volumeNode.GetImageData() == None:
      print('no image data')
      return False
    return True

  def delayDisplay(self, message, msec=1000):
    """ Test:
    logic version of delay display
    """
    profprint()
    # test
    print(message)
    self.info = qt.QDialog()
    self.infoLayout = qt.QVBoxLayout()
    self.info.setLayout(self.infoLayout)
    self.label = qt.QLabel(message, self.info)
    self.infoLayout.addWidget(self.label)
    qt.QTimer.singleShot(msec, self.info.close)
    self.info.exec_()

  def takeScreenshot(self, name, description, type=-1, annotate=True):
    """
    show the message even if not taking a screen shot
    ??? used?
    """
    # framework? #test
    profbox()
    self.delayDisplay(description)

    if self.enableScreenshots == 0:
      return

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == -1:
      # full window
      widget = slicer.util.mainWindow()
    elif type == slicer.qMRMLScreenShotDialog().FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog().ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog().Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog().Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog().Green:
      # green slice window
      widget = lm.sliceWidget("Green")

    # grab and convert to vtk image data
    qpixMap = widget.grab() if hasattr(widget, 'grab') else qt.QPixmap().grabWidget(widget)
    qimage = qpixMap.toImage()
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage, imageData)

    if annotate == True:
      annotationLogic = slicer.modules.annotations.logic()
      annotationLogic.CreateSnapShot(name, description, type, self.screenshotScaleFactor, imageData)

  def run(self, inputVolume, outputVolume, enableScreenshots=0, screenshotScaleFactor=1):
    """
    Test: Run the actual algorithm
    """
    # test
    profprint()
    self.delayDisplay('Running the aglorithm')

    self.enableScreenshots = enableScreenshots
    self.screenshotScaleFactor = screenshotScaleFactor

    self.takeScreenshot('NeedleFinder-Start', 'Start', -1)

    return True

  def drawIsoSurfaces(self):
    """ Draw isosurfaces from models of the visible needles only.
    This shall indicate radiation influence zones.
    """
    # research
    profprint()

    slicer.modules.NeedleFinderWidget.hideContourButton.setEnabled(1)
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')

    v = vtk.vtkAppendPolyData()
    canContinue = 0
    for modelNode in list(modelNodes.values()):
      print("for")
      if modelNode.GetAttribute("nth") != None and modelNode.GetDisplayVisibility() == 1 :
        canContinue = 1
        v.AddInputData(modelNode.GetPolyData())

    if canContinue == 1:
      modeller = vtk.vtkImplicitModeller()
      modeller.SetInputConnection(v.GetOutputPort())
      modeller.SetSampleDimensions(60, 60, 60)
      modeller.SetCapping(0)
      modeller.AdjustBoundsOn()
      modeller.SetProcessModeToPerVoxel()
      modeller.SetAdjustDistance(1)
      modeller.SetMaximumDistance(1.0)
      modeller.Update()

      contourFilter = vtk.vtkContourFilter()
      contourFilter.SetNumberOfContours(1)
      contourFilter.SetInputConnection(modeller.GetOutputPort())
      contourFilter.ComputeNormalsOn()
      contourFilter.ComputeScalarsOn()
      contourFilter.UseScalarTreeOn()
      contourFilter.SetValue(1, 10)
      # contourFilter.SetValue(2,13)
      # contourFilter.SetValue(3,15)
      # contourFilter.SetValue(4,20)
      # contourFilter.SetValue(5,25)
      contourFilter.Update()
      isoSurface = contourFilter.GetOutputDataObject(0)

      self.AddContour(isoSurface)

  def hideIsoSurfaces(self):
    """
    Hide radiation isosurfaces from models of the visible needles only
    """
    # research
    profprint()
    contourNode = getNode(self.contourNode)
    widget = slicer.modules.NeedleFinderWidget
    if contourNode != None:
      contourNode.SetDisplayVisibility(abs(widget.hideContourButton.isChecked() - 1))
      contourNode.GetModelDisplayNode().SetVisibility2D(abs(widget.hideContourButton.isChecked() - 1))

  def displayNeedleTube(self, ID):
    """
    from segmentation report, show/hide needle tube
    """
    # productive #onButton #report
    profprint()
    modelNode = getNode('vtkMRMLModelNode' + str(ID))
    displayNode = modelNode.GetModelDisplayNode()
    nVisibility = displayNode.GetVisibility()
    # print nVisibility
    if nVisibility:
      displayNode.Visibility2DOff()
      displayNode.SetVisibility(0)
      # also turn off yellow slice
      sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow")
      if sYellow == None :
        sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")
      sYellow.SetSliceVisible(0)
      reformatLogic = slicer.vtkSlicerReformatLogic()
      reformatLogic.SetSliceNormal(sYellow, 1, 0, 0)
      sYellow.Modified()
    else:
      displayNode.Visibility2DOn()
      displayNode.SetVisibility(1)

  def reformatSagittalView4Needle(self, ID):
    """
    reformat sagittal view to be tangent to needle and display a 3D plane
    """
    # productive #onButton #report
    profprint()
    for i in range(2):  # workaround update problem
      modelNode = getNode('vtkMRMLModelNode' + str(ID))
      polyData = modelNode.GetPolyData()
      nb = polyData.GetNumberOfPoints()
      base = [0, 0, 0]
      tip = [0, 0, 0]
      polyData.GetPoint(nb - 1, tip)
      polyData.GetPoint(0, base)
      a, b, c = tip[0] - base[0], tip[1] - base[1], tip[2] - base[2]
      # print a,b,c
      sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow")
      if sYellow == None :
        sYellow = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode2")
      reformatLogic = slicer.vtkSlicerReformatLogic()
      sYellow.SetSliceVisible(1)
      denom = b if abs(b) > 1e-9 else 1e-9
      reformatLogic.SetSliceNormal(sYellow, 1, -a / denom, 0)
      m = sYellow.GetSliceToRAS()
      m.SetElement(0, 3, base[0])
      m.SetElement(1, 3, base[1])
      m.SetElement(2, 3, base[2])
      sYellow.Modified()

  def findLabelNeedleID(self, ID):
    print("findLabelNeedleID")
    msgbox(whoami())
    """
    Takes the needle (vtkMRMLModelNode) with the right ID
    Evaluates the z-position of every 20 points of the vtkPolyData
    Takes the closest one to the surface of the template holes
    Find the closest hole to the needle and assign the label to the needle
    """
    # research
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    imageData = volumeNode.GetImageData()
    imageDimensions = imageData.GetDimensions()
    m = vtk.vtkMatrix4x4()
    minZ = None
    mindist = None
    volumeNode.GetIJKToRASMatrix(m)
    Z = m.GetElement(2, 3)
    needleNode = slicer.mrmlScene.GetNodeByID(ID)
    polydata = needleNode.GetPolyData()
    nb = polydata.GetNumberOfPoints()
    for i in range(nb):
      if 20 * i < nb:
        pt = [0, 0, 0]
        polydata.GetPoint(20 * i, pt)
        if (pt[2] - Z) ** 2 < minZ or minZ == None:
          minZ = (pt[2] - Z) ** 2
          bestNB = 20 * i
    hole = self.setNeedleCoordinates()
    # print bestNB
    A = [0, 0, 0]
    polydata.GetPoint(bestNB, A)
    for j in range(63):
      delta = ((hole[0][j] - (A[0])) ** 2 + (hole[1][j] - A[1]) ** 2) ** (0.5)
      if delta < mindist or mindist == None:
        bestmatch = j
        mindist = delta

    result = [bestmatch, mindist]
    return result

  def computerPolydataAndMatrix(self):
    """
    ??? Used?
    """
    print("computerPolydataAndMatrix")
    msgbox(whoami())
    Cylinder = vtk.vtkCylinderSource()

    Cylinder.SetResolution(1000)
    Cylinder.SetCapping(1)
    Cylinder.SetHeight(float(200.0))
    Cylinder.SetRadius(float(1.0))
    self.m_polyCylinder = Cylinder.GetOutput()

    quad = vtk.vtkQuadric()
    quad.SetCoefficients(1, 1, 1, 0, 0, 0, 0, 1, 0, 0)
    sample = vtk.vtkSampleFunction()
    sample.SetModelBounds(-30, 30, -60, 60, -30, 30)
    sample.SetCapping(0)
    sample.SetComputeNormals(1)
    sample.SetSampleDimensions(50, 50, 50)
    sample.SetImplicitFunction(quad)
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(sample.GetOutputPort())
    contour.ComputeNormalsOn()
    contour.ComputeScalarsOn()
    contour.GenerateValues(4, 0, 100)
    self.m_polyRadiation = contour.GetOutput()

    self.m_vtkmat = vtk.vtkMatrix4x4()
    self.m_vtkmat.Identity()

    RestruMatrix = vtk.vtkMatrix4x4()
    WorldMatrix = vtk.vtkMatrix4x4()
    Restru2WorldMatrix = vtk.vtkMatrix4x4()

    RestruMatrix.SetElement(0, 0, 0)
    RestruMatrix.SetElement(1, 0, 0)
    RestruMatrix.SetElement(2, 0, 0)
    RestruMatrix.SetElement(3, 0, 1)

    RestruMatrix.SetElement(0, 1, 1)
    RestruMatrix.SetElement(1, 1, 0)
    RestruMatrix.SetElement(2, 1, 0)
    RestruMatrix.SetElement(3, 1, 1)

    RestruMatrix.SetElement(0, 2, 0)
    RestruMatrix.SetElement(1, 2, 1)
    RestruMatrix.SetElement(2, 2, 0)
    RestruMatrix.SetElement(3, 2, 1)

    RestruMatrix.SetElement(0, 3, 0)
    RestruMatrix.SetElement(1, 3, 0)
    RestruMatrix.SetElement(2, 3, 1)
    RestruMatrix.SetElement(3, 3, 1)

    WorldMatrix.SetElement(0, 0, 0)
    WorldMatrix.SetElement(1, 0, 0)
    WorldMatrix.SetElement(2, 0, 0)
    WorldMatrix.SetElement(3, 0, 1)

    WorldMatrix.SetElement(0, 1, 1)
    WorldMatrix.SetElement(1, 1, 0)
    WorldMatrix.SetElement(2, 1, 0)
    WorldMatrix.SetElement(3, 1, 1)

    WorldMatrix.SetElement(0, 2, 0)
    WorldMatrix.SetElement(1, 2, 0)
    WorldMatrix.SetElement(2, 2, -1)
    WorldMatrix.SetElement(3, 2, 1)

    WorldMatrix.SetElement(0, 3, 0)
    WorldMatrix.SetElement(1, 3, 1)
    WorldMatrix.SetElement(2, 3, 0)
    WorldMatrix.SetElement(3, 3, 1)

    WorldMatrix.Invert()
    Restru2WorldMatrix.Multiply4x4(RestruMatrix, WorldMatrix, self.m_vtkmat)

  def AddContour(self, polyData):
    """
    Add caculated isosurfaces (self.drawIsoSurfaces) around visible needles to the scene
    and add opacity, color...
    Used by drawIsoSurface,0
    """
    # research
    # called by drawIsoSurfaces
    profprint()
    # print polyData
    scene = slicer.mrmlScene
    modelNodes = slicer.util.getNodes('vtkMRMLModelNode*')
    # ??? i suppose here is a bug:
    print('modelNodes: ', modelNodes)
    print('self.ContourNode: ', self.contourNode)
    #!!! self.contourNode is None

    if self.contourNode is not None:
      oldContourNode = getNode(self.contourNode) if isinstance(self.contourNode, str) else self.contourNode
      if oldContourNode:
        if oldContourNode.GetStorageNode():
          slicer.mrmlScene.RemoveNode(oldContourNode.GetStorageNode())
        oldContourNode.RemoveAllDisplayNodeIDs()
        slicer.mrmlScene.RemoveNode(oldContourNode)

    modelNode = slicer.vtkMRMLModelNode()
    modelNode.SetScene(scene)
    modelNode.SetAndObservePolyData(polyData)
    modelNode.SetName('Contours')

    # display node
    displayNode = slicer.vtkMRMLModelDisplayNode()
    displayNode.SetScene(scene)
    displayNode.SetVisibility(1)
    displayNode.SetOpacity(0.06)
    displayNode.SetVisibility2D(1)
    displayNode.SetScalarVisibility(1)
    displayNode.SetActiveScalarName('ImageScalars')
    displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileHotToColdRainbow2.txt')
    displayNode.SetScalarRange(10, 40)
    displayNode.SetBackfaceCulling(0)
    scene.AddNode(displayNode)
    scene.AddNode(modelNode)
    modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    displayNode.SetInputPolyDataConnection(modelNode.GetPolyDataConnection())
    self.contourNode = modelNode.GetID()

    # self.contourNode = modelNode.GetID()

    qt.QApplication.processEvents()

  #----------------------------------------------------------------------------------------------
  """ Needle Detection"""
  #----------------------------------------------------------------------------------------------

  def array2(self):
    """
    Used if needle tips input is given trough a labelmap.
    Extract the coordinates of each labels (after IslandEffect)
    """
    profbox(whoami())
    # research
    inputLabelID = self.__needleLabelSelector.currentNode().GetID()
    labelnode = slicer.mrmlScene.GetNodeByID(inputLabelID)
    i = labelnode.GetImageData()
    shape = list(i.GetDimensions())
    shape.reverse()
    a = vtk.util.numpy_support.vtk_to_numpy(i.GetPointData().GetScalars()).reshape(shape)
    labels = []
    val = [[0, 0, 0] for i in range(a.max() + 1)]
    for i in range(2, a.max() + 1):
      w = np.transpose(np.where(a == i))
      # labels.append(w.mean(axis=0))
      val[i] = [0, 0, 0]
      val[i][0] = w[int(round(w.shape[0] / 2))][2]
      val[i][1] = w[int(round(w.shape[0] / 2))][1]
      val[i][2] = w[int(round(w.shape[0] / 2))][0]
      if val[i] not in self.previousValues:
        labels.append(val[i])
        self.previousValues.append(val[i])
    return labels


  def factorial(self, n):
    """
    factorial(n): return the factorial of the integer n.
    factorial(0) = 1
    factorial(n) with n<0 is -factorial(abs(n))
    """
    if frequent: print(profprint())
    return factorial(n)

  def binomial(self, n, k):
    """
    calc's binomial coefficient
    """
    if frequent: profprint()
    return binomial(n, k)

  def Fibonacci(self, n):
    """
    calc's Fibonacci #
    """
    if frequent: profprint()
    return fibonacci(n)

  def stepSize(self, k, l):
    """
    The size of the step depends on:
    - the length of the needle
    - how many control points per needle
    """
    return step_size(k, l)

  def stepSizeAndre(self, k, l):
    """
    The size of the step depends on:
    - the length of the needle
    - how many control points per needle
    """
    return step_size_andre(k, l)

  def stepSize13(self, k, l):
    """MICCAI13 version
    The size of the step depends on:
    - the length of the needle
    - how many control points per needle
    """
    return step_size_13(k, l)

  def sortTable(self, table, cols):
    """
    sort a table by multiple columns
        table: a list of lists (or tuple of tuples) where each inner list
               represents a row
        cols:  a list (or tuple) specifying the column numbers to sort by
               e.g. (1,0) would sort by column 1, then by column 0
    """
    profprint()
    return sort_table(table, cols)

  def sortTableReverse(self, table, cols):
    """
    sort a table by multiple columns
        table: a list of lists (or tuple of tuples) where each inner list
               represents a row
        cols:  a list (or tuple) specifying the column numbers to sort by
               e.g. (1,0) would sort by column 1, then by column 0
    """
    profprint()
    return sort_table(table, cols, reverse=True)

  def ijk2ras(self, A, volumeNode=None):
    """
    Convert IJK coordinates to RAS coordinates. The transformation matrix is the one
    of the active volume on the red slice
    """
    if frequent: profprint()
    return ijk_to_ras(A, volumeNode)

  def ras2ijk(self, A, volumeNode=None):
    """
    Convert RAS coordinates to IJK coordinates. The transformation matrix is the one
    of the active volume on the red slice
    """
    if frequent: profprint()
    return ras_to_ijk(A, volumeNode)

"""

########################################################################################################################
TESTS
########################################################################################################################

"""

class NeedleFinderTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  """

  def getName(self):
    """
    return class name
    """
    return self.__class__.__name__

  def delayDisplay(self, message, msec=1000):
    """ Test:
    This utility method displays a small dialog and waits.
    This does two things: 1) it lets the event loop catch up
    to the state of the test so that rendering and widget updates
    have all taken place before the test continues and 2) it
    shows the user/developer/tester the state of the test
    so that we'll know when it breaks.
    """
    # test
    profprint()
    print(message)
    self.info = qt.QDialog()
    self.infoLayout = qt.QVBoxLayout()
    self.info.setLayout(self.infoLayout)
    self.label = qt.QLabel(message, self.info)
    self.infoLayout.addWidget(self.label)
    qt.QTimer.singleShot(msec, self.info.close)
    self.info.exec_()

  def setUp(self):
    """ Test:
    Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    # test
    profprint()
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """ Test:
    Run as few or as many tests as needed here.
    """
    # test #framework #productive
    profprint()

    self.setUp()
    self.test_NeedleFinder1()
    self.test_EndToEndWorkflow()
    self.test_UserInterfaceAndInteractions()

  def test_NeedleFinder1(self):
    """
    Unit test
    """
    # test #framework
    profprint()
    """
    Ideally you should have several levels of tests.  At the lowest level
    tests sould exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test", 50)
    #
    # first, get some data
    #
    import urllib.request, urllib.parse, urllib.error
    downloads = (
        ('https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url, name, loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        print(('Requesting download %s from %s...\n' % (name, url)))
        urllib.request.urlretrieve(url, filePath)
      if loader:
        print(('Loading %s...\n' % (name,)))
        loader(filePath)
    self.delayDisplay('Finished with download and loading\n', 50)

    volumeNode = getNode(pattern="FA")
    logic = NeedleFinderLogic()
    self.assertTrue(logic.hasImageData(volumeNode))
    self.assertTrue(is_rust_available())
    self.delayDisplay('Test passed!', 50)

  def test_EndToEndWorkflow(self):
    """Comprehensive end-to-end integration test of NeedleFinder in Slicer 5+."""
    self.delayDisplay("Starting end-to-end workflow test", 50)
    slicer.util.selectModule("NeedleFinder")
    slicer.app.processEvents()

    widget = slicer.modules.NeedleFinderWidget
    logic = widget.logic
    self.assertIsNotNone(widget)
    self.assertIsNotNone(logic)

    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    self.assertIsNotNone(volumeNode)

    # Set standard axial IJK-to-RAS matrix so k increases along +Z (RAS)
    ijkToRas = vtk.vtkMatrix4x4()
    ijkToRas.Identity()
    ijkToRas.SetElement(0, 0, -1.0)
    ijkToRas.SetElement(1, 1, -1.0)
    ijkToRas.SetElement(2, 2, 1.3)
    volumeNode.SetIJKToRASMatrix(ijkToRas)

    # 1. Axial segmentation limit marker & label map creation
    sliceNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetSliceNode()
    sliceNode.SetSliceOffset(20.0 * 1.3)
    logic.placeAxialLimitMarker(assign=True)
    self.assertEqual(widget.axialSegmentationLimit, 20)
    self.assertIsNotNone(widget.labelMapNode)

    # 2. Manual validation control points & VTK 9 spline/tube generation
    imageData = volumeNode.GetImageData()
    spacing = volumeNode.GetSpacing()
    dims = imageData.GetDimensions()
    mid_i, mid_j = dims[0] // 2, dims[1] // 2

    widget.editNeedleTxtBox.value = 1
    for step_idx, k_val in enumerate([90, 55, 20], start=1):
      logic.placeNeedleShaftEvalMarker([mid_i, mid_j, k_val], 1, step_idx)
    logic.drawValidationNeedles(1)
    manualNode1 = getNode("manual-seg_1")
    self.assertIsNotNone(manualNode1)
    self.assertEqual(manualNode1.GetPolyData().GetNumberOfPoints(), 2550)

    widget.editNeedleTxtBox.value = 2
    for step_idx, k_val in enumerate([90, 55, 20], start=1):
      logic.placeNeedleShaftEvalMarker([mid_i + 15, mid_j + 15, k_val], 2, step_idx)
    logic.drawValidationNeedles(2)
    manualNode2 = getNode("manual-seg_2")
    self.assertIsNotNone(manualNode2)
    self.assertEqual(manualNode2.GetPolyData().GetNumberOfPoints(), 2550)

    # 3. Assisted segmentation & centerline control point estimation
    centerline = logic.getCenterLine(manualNode1.GetPolyData())
    self.assertEqual(len(centerline), 50)
    assigned = logic.assignNeedle(centerline[25])
    self.assertEqual(assigned, 1)

    # 4. Obturator needles
    widget.initObturatorNeedles()
    widget.obtuNeedle = 0
    logic.obturatorNeedle([mid_i - 20, mid_j, 90], imageData, 0, spacing)
    widget.obtuNeedle = 1
    logic.obturatorNeedle([mid_i - 20, mid_j, 20], imageData, 0, spacing)
    widget.stop()
    obtuNode = getNode("obturator-seg_0")
    self.assertIsNotNone(obtuNode)

    # 5. All 6 Needle Detection Algorithms (0..5) + Hausdorff Evaluation
    tips, names = logic.returnTipsFromNeedleModels()
    self.assertEqual(len(tips), 2)
    for algo in (0, 1, 2, 3, 4, 5):
      widget.algoVersParameter.value = algo
      logic.deleteAllAutoNeedlesFromScene()
      logic.needleDetectionThread(tips, imageData, spacing=np.array(spacing), script=True, names=names)
      hd_res = logic.evaluate(script=False)
      self.assertEqual(len(hd_res), 2)

    # 6. Interactive detection, table synchronization, and deleteLastNeedle
    widget.algoVersParameter.value = 3
    logic.deleteAllAutoNeedlesFromScene()
    logic.needleDetectionThread([tips[0]], imageData, spacing=np.array(spacing), script=False)
    self.assertGreaterEqual(widget.row, 1)
    widget.deleteNeedleButton.setEnabled(True)
    logic.deleteLastNeedle()

    # 7. Parameter persistence (save & load)
    cfg_path = os.path.join(slicer.app.temporaryPath, "test_needlefinder_params.cfg")
    logic.saveParameters(cfg_path)
    logic.loadParameters(cfg_path)
    self.assertTrue(os.path.exists(cfg_path))
    self.delayDisplay("End-to-end workflow test passed!", 50)

  def test_UserInterfaceAndInteractions(self):
    """Test Qt UI controls, button signals, table buttons, and VTK slice-view click handlers."""
    self.delayDisplay("Starting UI and slice-view interaction test", 50)
    widget = slicer.modules.NeedleFinderWidget
    logic = widget.logic

    # 1. Test "1. Select Current Axial Slice as Seg. Limit" button click
    sliceWidgetRed = slicer.app.layoutManager().sliceWidget("Red")
    sliceNodeRed = sliceWidgetRed.sliceLogic().GetBackgroundLayer().GetSliceNode()
    sliceNodeRed.SetSliceOffset(26.0)
    widget.templateSliceButton.click()
    slicer.app.processEvents()
    self.assertTrue(widget.fiducialButton.isEnabled())
    self.assertIn("current:", widget.templateSliceButton.text)

    # 2. Test "Start Giving Control Points" toggle & simulated slice-view clicks
    widget.editNeedleTxtBox.setValue(3)
    widget.startGivingControlPointsButton.click()
    slicer.app.processEvents()
    self.assertTrue(widget.startGivingControlPointsButton.checked)
    self.assertEqual(widget.startGivingControlPointsButton.text, "Stop Giving Control Points")

    style = sliceWidgetRed.sliceView().interactorStyle()
    interactor = style.GetInteractor()
    for z_offset in (117.0, 71.5, 26.0):
      sliceNodeRed.SetSliceOffset(z_offset)
      interactor.SetEventPosition(100, 100)
      widget.processEventNeedleValidation(style, "LeftButtonPressEvent")
    slicer.app.processEvents()

    manualNode3 = getNode("manual-seg_3")
    self.assertIsNotNone(manualNode3)
    self.assertEqual(manualNode3.GetPolyData().GetNumberOfPoints(), 2550)

    # Untoggle control points mode
    widget.startGivingControlPointsButton.click()
    slicer.app.processEvents()
    self.assertFalse(widget.startGivingControlPointsButton.checked)
    self.assertEqual(widget.startGivingControlPointsButton.text, "Start Giving Control Points")

    # 3. Test "Next Validation Needle" button & "Render Manual Needle" button
    prev_val = widget.editNeedleTxtBox.value
    widget.validationNeedleButton.click()
    slicer.app.processEvents()
    self.assertEqual(widget.editNeedleTxtBox.value, prev_val + 1)
    widget.drawValidationNeedlesButton.click()
    slicer.app.processEvents()

    # 4. Test "Hide Marker Texts" toggle button
    widget.hideAnnotationTextButton.click()
    slicer.app.processEvents()
    self.assertTrue(widget.hideAnnotationTextButton.checked)
    widget.hideAnnotationTextButton.click()
    slicer.app.processEvents()
    self.assertFalse(widget.hideAnnotationTextButton.checked)

    # 5. Test "Assisted Manual Segmentation" toggle & simulated slice-view click
    widget.startAssistModeButton.click()
    slicer.app.processEvents()
    self.assertTrue(widget.startAssistModeButton.checked)
    self.assertEqual(widget.startAssistModeButton.text, "Stop Assisted Manual Segmentation")
    sliceNodeRed.SetSliceOffset(50.0)
    interactor.SetEventPosition(100, 100)
    widget.processEventAddCTLPoints(style, "LeftButtonPressEvent")
    slicer.app.processEvents()
    widget.startAssistModeButton.click()
    slicer.app.processEvents()
    self.assertFalse(widget.startAssistModeButton.checked)

    # 6. Test "2. Start Giving Needle Tips" toggle (via Ctrl+Return shortcut handler) & live click
    widget.algoVersParameter.setValue(3)
    widget.segmentNeedle()
    slicer.app.processEvents()
    self.assertTrue(widget.fiducialButton.checked)
    widget.CrosshairNode = slicer.mrmlScene.GetNthNodeByClass(0, 'vtkMRMLCrosshairNode')
    if widget.CrosshairNode:
      widget.CrosshairNode.SetCrosshairRAS(-128.0, -128.0, 117.0)
    sliceNodeRed.SetSliceOffset(117.0)
    widget.processEvent(style, "LeftButtonPressEvent")
    slicer.app.processEvents()
    widget.segmentNeedle()
    slicer.app.processEvents()
    self.assertFalse(widget.fiducialButton.checked)

    # 7. Test Segmentation Report QTableView embedded buttons (Show/Hide & View)
    self.assertIsNotNone(widget.view)
    self.assertGreater(widget.model.rowCount(), 0)
    showHideBtn = widget.view.indexWidget(widget.model.index(0, 2))
    viewBtn = widget.view.indexWidget(widget.model.index(0, 3))
    self.assertIsNotNone(showHideBtn)
    self.assertIsNotNone(viewBtn)
    showHideBtn.click()
    slicer.app.processEvents()
    showHideBtn.click()
    slicer.app.processEvents()
    viewBtn.click()
    slicer.app.processEvents()

    # 8. Test Manual Segmentation Report QTableView embedded buttons (Goto)
    self.assertIsNotNone(widget.viewCTL)
    self.assertGreater(widget.modelCTL.rowCount(), 0)
    gotoBtn = widget.viewCTL.indexWidget(widget.modelCTL.index(0, 3))
    self.assertIsNotNone(gotoBtn)
    gotoBtn.click()
    slicer.app.processEvents()

    # 9. Test Radiation Isosurface buttons ("Draw Radiation Isosurfaces" & "Hide Radiation Isosurfaces")
    widget.displayContourButton.click()
    slicer.app.processEvents()
    self.assertTrue(widget.hideContourButton.isEnabled())
    self.assertIsNotNone(getNode("Contours"))
    widget.hideContourButton.click()
    slicer.app.processEvents()
    widget.hideContourButton.click()
    slicer.app.processEvents()

    # 10. Test "Delete Last Segmented Needle" & "Reset Default Parameters" buttons
    self.assertTrue(widget.deleteNeedleButton.isEnabled())
    widget.deleteNeedleButton.click()
    slicer.app.processEvents()
    widget.resetParametersButton.click()
    slicer.app.processEvents()
    self.assertEqual(widget.algoVersParameter.value, 3)

    # 11. Test "Auto-Detect All Needles" button and 3D tubular detector
    volumeNode = sliceWidgetRed.sliceLogic().GetBackgroundLayer().GetVolumeNode()
    imageData = volumeNode.GetImageData()
    dims = imageData.GetDimensions()
    vol_arr = vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars()).reshape(dims[2], dims[1], dims[0])
    mid_i, mid_j = dims[0] // 2, dims[1] // 2
    # Embed 3 realistic dark needle tracks into bright tissue background
    vol_arr[:, max(0, mid_j - 35):min(dims[1], mid_j + 35), max(0, mid_i - 35):min(dims[0], mid_i + 35)] = 200
    for ni, nj, k_end in ((mid_i - 12, mid_j - 8, 70), (mid_i, mid_j + 6, 75), (mid_i + 12, mid_j - 6, 68)):
      for k in range(20, min(k_end, dims[2])):
        vol_arr[k, nj - 1:nj + 2, ni - 1:ni + 2] = 10
    imageData.GetPointData().GetScalars().Modified()
    imageData.Modified()

    widget.axialSegmentationLimit = 20
    widget.minConfidenceSpinBox.setValue(25.0)
    widget.autoDetectAllButton.click()
    slicer.app.processEvents()
    self.assertGreaterEqual(widget.model.rowCount(), 3)
    self.assertIsNotNone(getNode("*auto-seg_0*"))
    comment_item = widget.model.item(0, 4)
    self.assertIsNotNone(comment_item)
    self.assertIn("Conf:", comment_item.text())
    widget.colorByConfidenceCheckBox.setChecked(True)
    slicer.app.processEvents()
    self.assertTrue(widget.colorByConfidenceCheckBox.isChecked())
    widget.colorByConfidenceCheckBox.setChecked(False)
    slicer.app.processEvents()

    self.delayDisplay("UI and slice-view interaction test passed!", 50)
