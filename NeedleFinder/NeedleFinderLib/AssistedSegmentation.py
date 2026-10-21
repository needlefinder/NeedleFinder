"""Template landmark registration, MRML scene cleaning, and assisted manual needle segmentation."""

import fnmatch
import os
import xml.etree.ElementTree
from xml.etree.ElementTree import tostring
import numpy as np
import vtk
import slicer

from .GeometryUtils import (
    angle_between_vectors,
    compute_center_of_mass_ijk,
    extract_centerline,
)
from .LegacyEditorCompat import (
    EditorLib,
    Helper,
    LabelStatisticsLogic,
    create_fiducial_node,
    create_roi_node,
    get_fiducial_coords,
)
from .ProfilingUtils import (
    getNode,
    logprint,
    msgbox,
    profbox,
    profprint,
    whoami,
    whosdaddy,
)


class AssistedSegmentationMixin:
  """Mixin providing template registration, MRML XML scene cleanup, and assisted control point assignment."""

  def autoregistration(self):
    #####################################################################################
    # First we get the volume node
    #####################################################################################
    vl = slicer.modules.volumes.logic()
    red_logic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
    volumeNode = red_logic.GetBackgroundLayer().GetVolumeNode()
    imageData = red_logic.GetBackgroundLayer().GetVolumeNode()
    imageDimensions = imageData.GetImageData().GetDimensions()
    m = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(m)
    #####################################################################################
    # We load the template scene, with the fiducial lists 'moving' and 'fixed'
    #####################################################################################
    self.loadTemplate()
    #####################################################################################
    # We put back the volume node
    #####################################################################################
    Helper.SetBgFgVolumes(imageData.GetID(), None)
    print('BG set')
    #####################################################################################
    # set ROI
    # We create a ROI to constrain search space
    #####################################################################################
    print('setting ROI')
    c = 9
    roi = create_roi_node(
        name='obturator_ROI',
        center_ras=[0.0, 0.0, m.GetElement(2, 3) + imageDimensions[2] / c],
        radius_xyz=[100.0, 100.0, imageDimensions[2] / c],
        locked=True,
        visibility=True,
    )
    print('roi set')
    #####################################################################################
    # crop volume
    #####################################################################################
    print('cropVolume')
    cropVolumeNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLCropVolumeParametersNode')
    cropVolumeNode.SetScene(slicer.mrmlScene)
    cropVolumeNode.SetName('obturator_CropVolume_node')
    cropVolumeNode.SetIsotropicResampling(False)
    slicer.mrmlScene.AddNode(cropVolumeNode)
    cropVolumeNode.SetInputVolumeNodeID(volumeNode.GetID())
    cropVolumeNode.SetROINodeID(roi.GetID())
    cropVolumeLogic = slicer.modules.cropvolume.logic()
    cropVolumeLogic.Apply(cropVolumeNode)
    roiVolume = slicer.mrmlScene.GetNodeByID(cropVolumeNode.GetOutputVolumeNodeID())
    roiVolume.SetName("template-area-ROI")
    #####################################################################################
    # we create a label map for our threshold effect
    #####################################################################################
    print('labelmap')
    labelsColorNode = slicer.modules.colors.logic().GetColorTableNodeID(10)
    roiSegmentation = vl.CreateAndAddLabelVolume(slicer.mrmlScene, roiVolume, 'marker_segmentation')
    roiSegmentation.GetDisplayNode().SetAndObserveColorNodeID(labelsColorNode)
    #####################################################################################
    # We threshold between 80 (to be optimized) and the max value
    # Indeed, the markers are bright
    #####################################################################################
    print('threshold')
    thresh = vtk.vtkImageThreshold()
    thresh.SetInputData(roiVolume.GetImageData())
    maxThresh = roiVolume.GetImageData().GetScalarRange()[1]
    thresh.ThresholdBetween(80, maxThresh)
    thresh.SetInValue(255)
    thresh.SetOutValue(0)
    #####################################################################################
    # We apply an erode filter
    #####################################################################################
    print('erode')
    erode = vtk.vtkImageContinuousErode3D()
    erode.SetInputConnection(thresh.GetOutputPort())
    erode.SetKernelSize(3, 3, 1)
    erode.Update()
    roiSegmentation.SetAndObserveImageData(erode.GetOutputDataObject(0))
    #####################################################################################
    # Set Label
    #####################################################################################
    print('set label')
    appLogic = slicer.app.applicationLogic()
    selectionNode = appLogic.GetSelectionNode()
    selectionNode.SetReferenceActiveLabelVolumeID(roiSegmentation.GetID())
    appLogic.PropagateVolumeSelection()
    #####################################################################################
    # island effect
    # We apply the island effect that isolates each group of white voxels and apply a
    # different label to each
    #####################################################################################
    print('island effect')
    editUtil = EditorLib.EditUtil.EditUtil()
    parameterNode = editUtil.getParameterNode()
    sliceLogic = editUtil.getSliceLogic()
    islandsEffect = EditorLib.IdentifyIslandsEffectOptions()
    islandsEffect.setMRMLDefaults()
    islandsEffect.__del__()
    islandTool = EditorLib.IdentifyIslandsEffectLogic(sliceLogic)
    parameterNode.SetParameter("IslandEffect,minimumSize", '100')
    islandTool.removeIslands()
    LabelStatisticsLogic(volumeNode, roiSegmentation)
    labelData = roiSegmentation.GetImageData()
    stataccum = vtk.vtkImageAccumulate()
    stataccum.SetInputData(labelData)
    stataccum.Update()
    lo = int(stataccum.GetMin()[0])
    hi = int(stataccum.GetMax()[0])
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    if shNode:
      shNode.CreateFolderItem(shNode.GetSceneItemID(), 'LabelMarkers')
    labelsColorNodeX = slicer.modules.colors.logic().GetColorTableNodeID(10)
    labelX = vl.CreateAndAddLabelVolume(slicer.mrmlScene, roiVolume, 'labelX')
    labelX.GetDisplayNode().SetAndObserveColorNodeID(labelsColorNodeX)
    #####################################################################################
    # We compute the size of each island to filter out the one that do not fit our
    # size criteria
    #####################################################################################
    center = []
    stats = LabelStatisticsLogic(roiVolume, roiSegmentation).labelStats
    print('labels')
    for i in range(lo, hi + 1):
        print('\t...%d' % i)
        labelsColorNodeX = slicer.modules.colors.logic().GetColorTableNodeID(10)
        labelX = vl.CreateAndAddLabelVolume(slicer.mrmlScene, roiVolume, 'labelX')
        labelX.GetDisplayNode().SetAndObserveColorNodeID(labelsColorNodeX)
        thresholder = vtk.vtkImageThreshold()
        thresholder.SetInputConnection(roiSegmentation.GetImageDataConnection())
        thresholder.SetInValue(i + 1)
        thresholder.SetOutValue(0)
        thresholder.ReplaceOutOn()
        thresholder.ThresholdBetween(i, i)
        thresholder.SetOutputScalarType(roiSegmentation.GetImageData().GetScalarType())
        thresholder.Update()
        labelX.SetAndObserveImageData(thresholder.GetOutput())
        mm3 = stats.get((i, 'Volume mm^3'))
        if mm3 and 100 < mm3 < 465:
          center.append(self.ijk2ras(self.getCenterOfMass(labelX, 1), labelX))
    #####################################################################################
    # We filter and sort the mass centers
    #####################################################################################
    self.getAndSortFiducialPoints(center)
    print('fiducial sorted')
    #####################################################################################
    # We do the landmark registration
    #####################################################################################
    self.firstRegistration()
    print('Registration Done!!!')
    Helper.SetBgFgVolumes(imageData.GetID(), None)

  def getAndSortFiducialPoints(self, center):
      """
      Filter the points to keep the ones that fit a frame
      Order the point to then have the right orientation for the template
      :param center: list of center of mass of elements that have a size between 100 and 465 mm^3
      :return:
      """
      markerCenters = center
      nbCenter = len(center)
      for k in range(nbCenter):
          point = [0]
          for i in range(nbCenter):
              U, V, W = 0, 0, 0
              for j in range(nbCenter):
                  d = 0
                  if i != j and markerCenters[i] != (0, 0, 0):
                      d2 = (markerCenters[i][0] - markerCenters[j][0])**2 + (markerCenters[i][1] - markerCenters[j][1])**2 + (markerCenters[i][2] - markerCenters[j][2])**2
                      d = d2**0.5
                  if 45 <= d <= 53:
                      U += 1
                  elif 53 < d < 60:
                      V += 1
                  elif 70 <= d < 80:
                      W += 1
              if U + V + W >= 3:
                  point.extend([i])
      point.remove(0)
      minX = [999, 999, 999, 999]
      maxX = [-999, -999, -999, -999]
      sorted_idx = [[0, 0, 0] for _ in range(4)]
      sortedConverted = [[0, 0, 0] for _ in range(4)]
      for i in range(2):
          for k in point:
              if markerCenters[k][0] <= minX[0]:
                  minX[0] = markerCenters[k][0]
                  minX[1] = k
              elif markerCenters[k][0] <= minX[2]:
                  minX[2] = markerCenters[k][0]
                  minX[3] = k
              if markerCenters[k][0] >= maxX[0]:
                  maxX[0] = markerCenters[k][0]
                  maxX[1] = k
              elif markerCenters[k][0] >= maxX[2]:
                  maxX[2] = markerCenters[k][0]
                  maxX[3] = k
      if markerCenters[minX[1]][1] < markerCenters[minX[3]][1]:
          sorted_idx[0] = minX[1]
          sorted_idx[1] = minX[3]
      else:
          sorted_idx[0] = minX[3]
          sorted_idx[1] = minX[1]
      if markerCenters[maxX[1]][1] > markerCenters[maxX[3]][1]:
          sorted_idx[2] = maxX[1]
          sorted_idx[3] = maxX[3]
      else:
          sorted_idx[2] = maxX[3]
          sorted_idx[3] = maxX[1]
      sorted2 = [
          sorted_idx[2],
          sorted_idx[3],
          sorted_idx[0],
          sorted_idx[1],
      ]
      fixed_list = getNode('Fiducial List_fixed')
      if fixed_list and hasattr(slicer.modules, 'annotations'):
          slicer.modules.annotations.logic().SetActiveHierarchyNodeID(fixed_list.GetID())
      for k in range(4):
          create_fiducial_node(str(k), markerCenters[sorted2[k]])

      sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed") or slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode1")
      if sRed:
          m = sRed.GetSliceToRAS()
          m.SetElement(0, 3, sortedConverted[3][0])
          m.SetElement(1, 3, sortedConverted[3][1])
          m.SetElement(2, 3, sortedConverted[3][2])
          sRed.Modified()
      return sorted2

  def loadTemplate(self):
      '''
      Load CAD files of the template and obturator
      :return:
      '''
      a = None
      if not bool(getNode('Template')):
          pathToScene = slicer.modules.igyne.path.replace("iGyne/iGyne.py", "iGyne/Resources/Template/4points/Template.mrml")
          a = slicer.util.loadScene(pathToScene)
      print('template loaded')
      return a

  def getCenterOfMass(self, volumeNode, step=1):
      '''
      Compute center of mass of a volume node using vectorized NumPy indexing.
      :param volumeNode:
      :param step:
      :return:
      '''
      if hasattr(volumeNode, "GetLabelMap") and volumeNode.GetLabelMap() == 0:
        print(('Warning: input volume is not labelmap: \'' + volumeNode.GetName() + '\''))
      return compute_center_of_mass_ijk(volumeNode, step=step)

  def firstRegistration(self):
      '''
      landmark registration (fiducial registration CLI Module)
      '''
      tNode = getNode('vtkMRMLLinearTransformNode_Template')
      slicer.mrmlScene.AddNode(tNode)
      sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLAnnotationHierarchyNode')
      movingLandmarks = vtk.vtkCollection()
      for nodeIndex in range(sliceNodeCount):
        sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLAnnotationHierarchyNode')
        if sliceNode.GetName() == "Fiducial List_moved":
          sliceNode.GetAssociatedChildrenNodes(movingLandmarks)
      parameters = {
          "fixedLandmarks": getNode('Fiducial List_fixed').GetID(),
          "movingLandmarks": getNode('Fiducial List_moving').GetID(),
          "saveTransform": tNode,
          "transformType": "Rigid",
          "rms": "",
          "outputMessage": "",
      }
      fidreg = slicer.modules.fiducialregistration
      cli_node = slicer.cli.run(fidreg, None, parameters)
      cli_node.AddObserver('ModifiedEvent', self.processRegistrationCompletion)

  def processRegistrationCompletion(self, node, event):
      '''
      Once the registration is done, display a message to say it is finished
      :param node:
      :param event:
      :return:
      '''
      status = node.GetStatusString()
      if status == 'Completed':
          templateNode = getNode('Template')
          obturatorNode = getNode('Obturator_reg')
          df = templateNode.GetDisplayNode()
          df.SetVisibility2D(1)
          do = obturatorNode.GetDisplayNode()
          do.SetVisibility2D(1)
          tNode = getNode('vtkMRMLLinearTransformNode_Template')
          templateNode.SetAndObserveTransformNodeID(tNode.GetID())
          obturatorNode.SetAndObserveTransformNodeID(tNode.GetID())

  def controlPointsEstimation(self, n=None, N=11):
      nodes = slicer.util.getNodes('manual-seg_*')
      for key, node in zip(list(nodes.keys()), list(nodes.values())):
          needleNumber = int(key.lstrip('manual-seg_'))
          if n is not None and needleNumber != n:
            continue
          polyData = node.GetPolyData()

          previousPts = slicer.util.getNodes('.' + str(needleNumber) + '-*')
          for previousPt in list(previousPts.values()):
            slicer.mrmlScene.RemoveNode(previousPt)

          ctrlPts = []
          centerline = self.getCenterLine(polyData)

          for i in range(N):
              p = centerline[min(int(i * 50) // (N - 1), len(centerline) - 1)]
              ctrlPts.append(p)
              self.placeNeedleShaftEvalMarker(p, int(needleNumber), N - i, type='ras')

  def getCenterLine(self, polydata):
      '''
      :param polydata: vtkPolydata
      :return: list of 50 points at the center of the needle
      '''
      return extract_centerline(polydata, num_points=50, num_sides=50)

  def cleanScene(self, path=None):
    '''
    Clean the MRML scene from duplicates
    :param path: ex '/cases/01/'
    :return:
    '''
    if not path:
      path = slicer.modules.NeedleFinderWidget.scenePath.text
    print(path)

    if path[-4:] != 'mrml':
      if path[-1:] != '/':
        path += '/'
      fname = os.listdir(path)
      mrmlfile = None
      for file in fname:
              if fnmatch.fnmatch(file, '20*Scene.mrml'):
                  mrmlfile = path + file
    else:
      mrmlfile = path
      a = mrmlfile.split('/')
      path = ''
      for s in range(len(a) - 1):
          path += a[s] + '/'

    e = xml.etree.ElementTree.parse(mrmlfile).getroot()
    e = self.removeDuplicateMRMLFromScene(e)
    e = self.removeAnnotationButKeepTemplate(e)
    e = self.removeDuplicateNeedlesFromScene(e)

    with open(path + "/cleanedScene.mrml", "w") as text_file:
      text_file.write(tostring(e, encoding='unicode'))

  def removeDuplicateNeedlesFromScene(self, e):
      c = list(e)
      indexToRemove = []
      for i, child in enumerate(e):
              if child.tag[:5] == 'Model':
                  if len(child.attrib['name'].split('_')) > 2:
                      indexToRemove.append(i)
                      if c[i - 2].tag[:5] == 'Model':
                          indexToRemove.append(i - 2)
                      if c[i - 1].tag[:5] == 'Model':
                          indexToRemove.append(i - 1)

      indexToRemove.sort(reverse=True)
      for i in indexToRemove:
          e.remove(c[i])
      return e

  def removeAnnotationButKeepTemplate(self, e):
    c = list(e)
    indexToRemove = []
    removeListIDS = []
    for i, child in enumerate(e):
        if child.tag == 'AnnotationFiducials' and child.attrib['name'] != 'template slice position':
            try:
              n = child.attrib['references'].split(';')
              for t in n:
                  t2 = t.split(':')
                  if len(t2) > 1:
                      t3 = t2[1].split(' ')
                      removeListIDS.append(t3)
                      removeListIDS.append([child.attrib['id']])
            except Exception:
              pass

    r = [item for sublist in removeListIDS for item in sublist]
    for i, child in enumerate(e):
        if child.attrib['id'] in r or child.tag == 'AnnotationHierarchyNode':
            indexToRemove.append(i)

    indexToRemove.sort(reverse=True)
    for i in indexToRemove:
        e.remove(c[i])
    return e

  def removeDuplicateMRMLFromScene(self, e):
    c = list(e)
    indexToRemove = []
    keepIDS = []
    IDToKeep = None

    for i, child in enumerate(e):
        if child.tag == 'SliceComposite' and child.attrib['name'] == 'SliceComposite':
            IDToKeep = child.attrib['backgroundVolumeID']
            keepIDS.append(child.attrib['backgroundVolumeID'])

    for i, child in enumerate(e):
        if child.tag == 'Volume' and child.attrib['id'] == IDToKeep:
            try:
                n = child.attrib['references'].split(';')
                for t in n:
                    t2 = t.split(':')
                    if len(t2) > 1:
                        t3 = t2[1].split(' ')
                        keepIDS.append(t3[0])
                        keepIDS.append([child.attrib['id']])
            except Exception:
                pass

    for i, child in enumerate(e):
        if child.tag[:6] == 'Volume' and child.attrib['id'] not in keepIDS:
            indexToRemove.append(i)

    indexToRemove.sort(reverse=True)
    for i in indexToRemove:
        e.remove(c[i])
    return e

  def removeAnnotationFromScene(self, e):
      c = list(e)
      indexToRemove = []
      for i, child in enumerate(e):
          if child.tag[:10] == 'Annotation':
              indexToRemove.append(i)

      indexToRemove.sort(reverse=True)
      for i in indexToRemove:
          e.remove(c[i])
      return e

  ######################################################################################################################
  # ASSISTED MANUAL SEGMENTATION
  ######################################################################################################################

  def angle(self, v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    return angle_between_vectors(v1, v2)

  def getOrientationVect(self, a, b):
      """Returns the AB vector."""
      return np.array(a) - np.array(b)

  def getNeedleOrientation(self, needleNumber):
      """Returns the vector defined by lowest point | highest point of a needle."""
      controlPoints = self.sortNeedlePoints(needleNumber)
      return self.getOrientationVect(controlPoints[0], controlPoints[-1])

  def sortNeedlePoints(self, needleNumber):
      """Returns sorted needle control points."""
      nodes = slicer.util.getNodes('.%d-*' % needleNumber)
      controlPointsUnsorted = []
      for node in list(nodes.values()):
          p = [0.0, 0.0, 0.0]
          get_fiducial_coords(node, p)
          controlPointsUnsorted.append(p)
      controlPoints = self.sortTable(controlPointsUnsorted, (2, 1, 0))
      return controlPoints

  def getNeedleLowestPoint(self, needleNumber):
      controlPoints = self.sortNeedlePoints(needleNumber)
      return controlPoints[0]

  def getNeedleHighestPoint(self, needleNumber):
      controlPoints = self.sortNeedlePoints(needleNumber)
      return controlPoints[-1]

  def findHigherNeedles(self, pt):
      """Return the needle that is the best aligned to the point."""
      nodes = slicer.util.getNodes('manual-seg_*')
      candidates = []
      validNeedles = self.findNeedles()
      for node in list(nodes.values()):
          name = node.GetName()
          nb = int(name.split('_')[1])
          if nb in validNeedles:
              lp = self.getNeedleLowestPoint(nb)
              if lp[2] > pt[2]:
                  candidates.append([name, self.angle(self.getOrientationVect(pt, lp), self.getNeedleOrientation(nb))])
      return candidates

  def findLowerNeedles(self, pt):
      """Return the needle that is the best aligned to the point."""
      nodes = slicer.util.getNodes('manual-seg_*')
      candidates = []
      validNeedles = self.findNeedles()
      for node in list(nodes.values()):
          name = node.GetName()
          nb = int(name.split('_')[1])
          if nb in validNeedles:
              hp = self.getNeedleHighestPoint(nb)
              if hp[2] < pt[2]:
                  theta = self.angle(self.getNeedleOrientation(nb), -self.getOrientationVect(pt, hp))
                  candidates.append([name, min(theta, abs(theta - np.pi))])
      return candidates

  def findCloseNeedles(self, pt, filterOutNeedles=[]):
      '''Find the closest needle to the point, which is not in the list filterOutNeedles.'''
      nodes = slicer.util.getNodes('manual-seg_*')
      candidates = []
      validNeedles = self.findNeedles()
      for node in list(nodes.values()):
          name = node.GetName()
          nb = int(name.split('_')[1])
          if nb in validNeedles and name not in filterOutNeedles:
              p = node.GetPolyData()
              x = self.getCenterLine(p)
              dists = [self.distance(q, pt) for q in x]
              candidates.append([name, min(dists)])
      return candidates

  def findMatchingCTLPoint(self, mo, pt):
      candidates = []
      for isolPt in self.findIsolatedCTLPoints():
          pt2 = [0.0, 0.0, 0.0]
          get_fiducial_coords(isolPt, pt2)
          a = self.angle(self.getOrientationVect(pt2, pt), mo)
          ori = min([a, abs(np.pi - a)])
          print(isolPt.GetName(), ori)
          if ori < 0.1:
              candidates.append([int(isolPt.GetName().strip('.').split('-')[0]), ori])
      return candidates

  def assignNeedle(self, pt):
      '''Assign needle number to point.'''
      candidate = None
      result = None
      lowest = None
      highest = None
      closest = None

      mo = self.meanOrientation()
      cLow = self.findLowerNeedles(pt)
      cHigh = self.findHigherNeedles(pt)
      cClose = self.findCloseNeedles(pt)

      if cLow: lowest = self.sortTable(cLow, (1, 0))[0]
      if cHigh: highest = self.sortTable(cHigh, (1, 0))[0]
      if cClose: closest = self.sortTable(cClose, (1, 0))[0]

      if lowest and highest:
          candidate = self.sortTable([lowest, highest], (1, 0))[0]
      elif lowest:
          candidate = lowest
      elif highest:
          candidate = highest

      if closest:
          if closest[1] < 1:
              result = closest[0]
          elif closest[1] < 2:
              if candidate and candidate[1] < 0.3:
                  result = candidate[0]
              else:
                  result = closest[0]
          elif candidate and candidate[1] < 0.3:
              result = candidate[0]
      elif candidate and candidate[1] < 0.3:
          result = candidate[0]

      if result:
          result = int(result.strip('manual-seg_'))
      if not result:
          candidates = self.findMatchingCTLPoint(mo, pt)
          if candidates:
              result = self.sortTable(candidates, (1, 0))[0][0]
      if not result:
          print('no good candidate to assign to -> new point')
      return result

  def findNeedles(self, all=None):
      needleNbs = []
      needles = slicer.util.getNodes('manual-seg_*')
      for needle in list(needles.values()):
          name = needle.GetName()
          nb = int(name.split('_')[1])
          points = slicer.util.getNodes('.%d-*' % nb)
          if len(points) > 1 or all is not None:
              needleNbs.append(nb)
      return needleNbs

  def meanOrientation(self):
      '''Compute Mean orientation of needles.'''
      orientation = 0
      c = 1
      for nb in self.findNeedles():
          c += 1
          orientation += self.getNeedleOrientation(nb)
      return orientation / c

  def findIsolatedCTLPoints(self):
      needles = slicer.util.getNodes('manual-seg_*')
      needleNbs = [int(needle.GetName().split('_')[1]) for needle in list(needles.values())]
      isolatedPts = []
      for nb in needleNbs:
          points = slicer.util.getNodes('.%d-*' % nb)
          if 0 < len(points) < 2:
              isolatedPts.append(list(points.values())[0])
      return isolatedPts
