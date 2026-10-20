# -*- coding: utf-8 -*-
"""Template landmark registration, MRML scene cleaning, and assisted manual needle segmentation."""

import fnmatch
import math
import os
import os.path
import xml.etree.ElementTree
from xml.etree.ElementTree import tostring
import numpy
import numpy as np
import vtk
import qt
import ctk
import slicer

from .LegacyEditorCompat import EditorLib, Helper, LabelStatisticsLogic
from .ProfilingUtils import (
    getNode,
    logprint,
    msgbox,
    profbox,
    profprint,
    whoami,
    whosdaddy,
)


class AssistedSegmentationMixin(object):
  """Mixin providing template registration, MRML XML scene cleanup, and assisted control point assignment."""

  def autoregistration(self):
    #####################################################################################
    # First we get the volume node
    #####################################################################################
    vl = slicer.modules.volumes.logic()
    red_logic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
    volumeNode = red_logic.GetBackgroundLayer().GetVolumeNode()
    imageData = red_logic.GetBackgroundLayer().GetVolumeNode()
    dn = imageData.GetScalarVolumeDisplayNode()
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
    roi = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationROINode')
    slicer.mrmlScene.AddNode(roi)
    roi.SetROIAnnotationVisibility(1)
    roi.SetRadiusXYZ(100,100,imageDimensions[2]/c)
    roi.SetXYZ(0,0,m.GetElement(2,3)+imageDimensions[2]/c)
    roi.SetLocked(1)
    print('roi set')
    #####################################################################################
    # crop volume
    #####################################################################################
    print('cropVolume')
    cropVolumeNode =slicer.mrmlScene.CreateNodeByClass('vtkMRMLCropVolumeParametersNode')
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
    # thresh.ReplaceOutOn()
    # thresh.ReplaceInOn()
    # thresh.Update()
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
    lm = slicer.app.layoutManager()
    islandsEffect = EditorLib.IdentifyIslandsEffectOptions()
    islandsEffect.setMRMLDefaults()
    islandsEffect.__del__()
    islandTool = EditorLib.IdentifyIslandsEffectLogic(sliceLogic)
    parameterNode.SetParameter("IslandEffect,minimumSize",'100')
    islandTool.removeIslands()
    LabelStatisticsLogic(volumeNode,roiSegmentation)
    labelData = roiSegmentation.GetImageData()
    stataccum = vtk.vtkImageAccumulate()
    stataccum.SetInputData(labelData)
    stataccum.Update()
    lo = int(stataccum.GetMin()[0])
    hi = int(stataccum.GetMax()[0])
    # label stats
    labelStats = {}
    labelStats['Labels'] = []
    hierarchyNode = slicer.vtkMRMLModelHierarchyNode()
    hierarchyNode.SetScene( slicer.mrmlScene )
    hierarchyNode.SetName('LabelMarkers')
    slicer.mrmlScene.AddNode(hierarchyNode)
    #
    labelsColorNodeX = slicer.modules.colors.logic().GetColorTableNodeID(10)
    labelX = vl.CreateAndAddLabelVolume(slicer.mrmlScene, roiVolume, 'labelX')
    labelX.GetDisplayNode().SetAndObserveColorNodeID(labelsColorNodeX)
    #####################################################################################
    # We compute the size of each island to filter out the one that do not fit our
    # size criteria
    #####################################################################################
    #
    center = []
    #
    stats = LabelStatisticsLogic(roiVolume, roiSegmentation).labelStats
    print('labels')
    for i in range(lo,hi+1):
        #
        print('\t...%d' % i)
        labelsColorNodeX = slicer.modules.colors.logic().GetColorTableNodeID(10)
        labelX = vl.CreateAndAddLabelVolume(slicer.mrmlScene, roiVolume, 'labelX')
        labelX.GetDisplayNode().SetAndObserveColorNodeID(labelsColorNodeX)
        thresholder = vtk.vtkImageThreshold()
        thresholder.SetInputConnection(roiSegmentation.GetImageDataConnection())
        thresholder.SetInValue(i+1)
        thresholder.SetOutValue(0)
        thresholder.ReplaceOutOn()
        thresholder.ThresholdBetween(i,i)
        thresholder.SetOutputScalarType(roiSegmentation.GetImageData().GetScalarType())
        thresholder.Update()
        labelX.SetAndObserveImageData(thresholder.GetOutput())
        mm3 = stats.get((i, 'Volume mm^3'))
        if mm3 <465 and mm3> 100:
          center.append(self.ijk2ras(self.getCenterOfMass(labelX, 1 ),labelX))
          # self.__registrationStatus.setText('Found Marker ' +str(i)+'...')
    # print(center)
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
      # self.__registrationStatus.setText('Registration processing...')
      # pNode = self.parameterNode()
      # fixedAnnotationList = slicer.mrmlScene.GetNodeByID(pNode.GetParameter('fixedLandmarksListID'))
      # if fixedAnnotationList != None:
      #   fixedAnnotationList.RemoveAllChildrenNodes()
      markerCenters = center
      nbCenter = len(center)
      for k in range(nbCenter):
          point = [0]
          for i in range(nbCenter):
              U,V,W = 0,0,0
              for j in range(nbCenter):
                  d = 0
                  if i != j and markerCenters[i]!=(0,0,0):
                      d2 = (markerCenters[i][0]-markerCenters[j][0])**2+(markerCenters[i][1]-markerCenters[j][1])**2+(markerCenters[i][2]-markerCenters[j][2])**2
                      d = d2**0.5
                  # print markerCenters[i],markerCenters[j]
                  #print d
                  if d >=45 and d<=53:
                      U += 1
                  elif d >53 and d<60:
                      V +=1
                  elif d >=70 and d<80:
                      W +=1
              #print U,V,W
              if U+V+W>=3:
                #print markerCenters[i]
                  point.extend([i])
      point.remove(0)
      minX = [999,999,999,999]
      maxX = [-999,-999,-999,-999]
      sorted = [[0,0,0] for l in range(4)]
      sortedConverted = [[0,0,0] for l in range(4)]
      for i in range(2):
          for k in point:
              if markerCenters[k][0]<= minX[0]:
                  minX[0] = markerCenters[k][0]
                  minX[1] = k
              elif markerCenters[k][0]<= minX[2]:
                  minX[2] = markerCenters[k][0]
                  minX[3] = k
              if markerCenters[k][0]>= maxX[0]:
                  maxX[0] = markerCenters[k][0]
                  maxX[1] = k
              elif markerCenters[k][0]>= maxX[2]:
                  maxX[2] = markerCenters[k][0]
                  maxX[3] = k
      if markerCenters[minX[1]][1] < markerCenters[minX[3]][1]:
          sorted[0] = minX[1]
          sorted[1] = minX[3]
      else:
          sorted[0] = minX[3]
          sorted[1] = minX[1]
      if markerCenters[maxX[1]][1]>markerCenters[maxX[3]][1]:
          sorted[2] = maxX[1]
          sorted[3] = maxX[3]
      else:
          sorted[2] = maxX[3]
          sorted[3] = maxX[1]
      sorted2 = [0,0,0,0]
      if 1:#self.horizontalTemplate.isChecked():
          sorted2[0]=sorted[2]
          sorted2[2]=sorted[0]
          sorted2[1]=sorted[3]
          sorted2[3]=sorted[1]
      else:
          sorted2[0]=sorted[3]
          sorted2[2]=sorted[1]
          sorted2[1]=sorted[0]
          sorted2[3]=sorted[2]
      # logic = slicer.modules.annotations.logic()
      # logic.SetActiveHierarchyNodeID(pNode.GetParameter('fixedLandmarksListID'))
      # if pNode.GetParameter("Template")=='4points':
      #     nbPoints=4
      # elif pNode.GetParameter("Template")=='3pointsCorners':
      #     nbPoints=3
      l = slicer.modules.annotations.logic()
      l.SetActiveHierarchyNodeID(getNode('Fiducial List_fixed').GetID())
      for k in range(4) :
          fiducial = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationFiducialNode')
          fiducial.SetReferenceCount(fiducial.GetReferenceCount()-1)
          fiducial.SetFiducialCoordinates(markerCenters[sorted2[k]])
          fiducial.SetName(str(k))
          fiducial.Initialize(slicer.mrmlScene)

      sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
      if sRed ==None :
          sRed = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode1")
      # sRed.SetSliceVisible(1)
      m= sRed.GetSliceToRAS()
      m.SetElement(0,3,sortedConverted[3][0])
      m.SetElement(1,3,sortedConverted[3][1])
      m.SetElement(2,3,sortedConverted[3][2])
      sRed.Modified()
      return sorted2

  def loadTemplate(self):
      '''
      Load CAD files of the template and obturator
      :return:
      '''
      if not bool(getNode('Template')):
          pathToScene = slicer.modules.igyne.path.replace("iGyne/iGyne.py","iGyne/Resources/Template/4points/Template.mrml")
          a = slicer.util.loadScene( pathToScene)
      print('template loaded')
      return a

  def getCenterOfMass(self, volumeNode,step):
      '''
      Compute center of mass of a volume node. If it is too slow, increase 'step' to skip through some slices
      :param volumeNode:
      :param step:
      :return:
      '''
      centerOfMass = [0,0,0]
      if volumeNode.GetLabelMap() == 0:
        print(('Warning: input volume is not labelmap: \'' + volumeNode.GetName() + '\''))
      #
      numberOfStructureVoxels = 0
      sumX = sumY = sumZ = 0
      #
      volume = volumeNode.GetImageData()
      for z in range(volume.GetExtent()[4], volume.GetExtent()[5]+1):
        for y in range(volume.GetExtent()[2], volume.GetExtent()[3]+1,step):
          for x in range(volume.GetExtent()[0], volume.GetExtent()[1]+1,step):
            voxelValue = volume.GetScalarComponentAsDouble(x,y,z,0)
            if voxelValue>0:
              numberOfStructureVoxels = numberOfStructureVoxels+1
              sumX = sumX + x
              sumY = sumY + y
              sumZ = sumZ + z
      #
      if numberOfStructureVoxels > 0:
        centerOfMass[0] = sumX / numberOfStructureVoxels
        centerOfMass[1] = sumY / numberOfStructureVoxels
        centerOfMass[2] = sumZ / numberOfStructureVoxels
      #
      return centerOfMass

  def firstRegistration(self):
      '''
      landmark registration (fiducial registration CLI Module)
      '''
      # pNode = self.parameterNode()
      tNode = getNode('vtkMRMLLinearTransformNode_Template')
      slicer.mrmlScene.AddNode(tNode)
      sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLAnnotationHierarchyNode')
      movingLandmarks = vtk.vtkCollection()
      for nodeIndex in range(sliceNodeCount):
        sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLAnnotationHierarchyNode')
        if sliceNode.GetName() == "Fiducial List_moved":
          sliceNode.GetAssociatedChildrenNodes(movingLandmarks)
      OutputMessage = ""
      RMS = ""
      parameters = {}
      parameters["fixedLandmarks"] = getNode('Fiducial List_fixed').GetID()
      parameters["movingLandmarks"] = getNode('Fiducial List_moving').GetID()
      parameters["saveTransform"] = tNode
      parameters["transformType"] = "Rigid"
      parameters["rms"] = RMS
      parameters["outputMessage"] = OutputMessage
      fidreg = slicer.modules.fiducialregistration
      __cliNode = None
      __cliNode = slicer.cli.run(fidreg, __cliNode, parameters)
      __cliObserverTag = __cliNode.AddObserver('ModifiedEvent', self.processRegistrationCompletion)
      # self.__registrationStatus.setText('Wait ...')
      # self.firstRegButton.setEnabled(0)


  def processRegistrationCompletion(self,node, event):
      '''
      Once the registration is done, display a message to say it is finished
      :param node:
      :param event:
      :return:
      '''
      status = node.GetStatusString()
      # self.__registrationStatus.setText('Registration '+status)
      if status == 'Completed':
          # self.firstRegButton.setEnabled(1)
          # pNode = self.parameterNode()
          templateNode = getNode('Template')
          obturatorNode = getNode('Obturator_reg')
          df = templateNode.GetDisplayNode()
          df.SetVisibility2D(1)
          do = obturatorNode.GetDisplayNode()
          do.SetVisibility2D(1)
          tNode = getNode('vtkMRMLLinearTransformNode_Template')
          # roiNode = slicer.mrmlScene.GetNodeByID()
          templateNode.SetAndObserveTransformNodeID(tNode.GetID())
          obturatorNode.SetAndObserveTransformNodeID(tNode.GetID())
          # roiNode.SetAndObserveTransformNodeID(tNode)
          # Helper.SetBgFgVolumes(pNode.GetParameter('baselineVolumeID'),'')
          # pNode.SetParameter('followupTransformID', self.__followupTransform.GetID())
          # self.registered = 1


  def controlPointsEstimation(self, n=None, N = 11):
      nodes = slicer.util.getNodes('manual-seg_*')
      for key, node in zip(list(nodes.keys()), list(nodes.values())):
          needleNumber = int(key.lstrip('manual-seg_')) #node.GetAttribute('nth')
          if n!=None and needleNumber!=n:
            continue
          polyData = node.GetPolyData()

          previousPts = slicer.util.getNodes('.'+str(needleNumber)+'-*')
          for previousPt in list(previousPts.values()):
            slicer.mrmlScene.RemoveNode(previousPt)

          ctrlPts = []
          centerline = self.getCenterLine(polyData)

          for i in range(N):
              # p = polyData.GetPoint(int(i*2500/(N-1)))
              p = centerline[min(int(i*50)//(N-1), len(centerline)-1)]
              ctrlPts.append(p)
              self.placeNeedleShaftEvalMarker(p,int(needleNumber), N-i, type='ras' )

  def getCenterLine(self, polydata):
      '''
      :param polydata: vtkPolydata
      :return: list of 50 points at the center of the needle
      '''
      centerline = []
      for i in range(50):
          p1 = polydata.GetPoint(i*50)
          p2 = polydata.GetPoint(i*50 + 25)
          p = (np.array(p1) + np.array(p2))/2
          centerline.append(p)
      return centerline

  def cleanScene(self, path = None):
    '''
    Clean the MRML scene from duplicates
    :param path: ex '/cases/01/'
    :return:
    '''
    # path  = './'

    if not path:
      path  = slicer.modules.NeedleFinderWidget.scenePath.text
    print(path)


    if path[-4:] != 'mrml':
      if path[-1:] != '/':
        path += '/'
      fname = os.listdir(path)
      mrmlfile = None
      for file in fname:
              if fnmatch.fnmatch(file, '20*Scene.mrml'):
                  mrmlfile =  path + file

    else:
      mrmlfile = path
      a = mrmlfile.split('/')
      path = ''
      for s in range(len(a)-1):
          path += a[s]+'/'

    e = xml.etree.ElementTree.parse(mrmlfile).getroot()
    e = self.removeDuplicateMRMLFromScene(e)
    # e = self.removeAnnotationFromScene(e)
    e = self.removeAnnotationButKeepTemplate(e)
    e = self.removeDuplicateNeedlesFromScene(e)

    tostring(e)
    text_file = open(path + "/cleanedScene.mrml", "w")
    text_file.write(tostring(e, encoding='unicode'))
    text_file.close()


  def removeDuplicateNeedlesFromScene(self, e):
      c = list(e)
      indexToRemove = []
      for i, child in enumerate(e):
              if child.tag[:5] == 'Model':
      #             print i, child.tag, child.attrib['name']
                  if len(child.attrib['name'].split('_'))>2:
                      indexToRemove.append(i)
                      if c[i-2].tag[:5] == 'Model':
                          indexToRemove.append(i-2)
                      if c[i-1].tag[:5] == 'Model':
                          indexToRemove.append(i-1)

      indexToRemove.sort()
      indexToRemove.reverse()

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
                  if len(t2)>1:
                      t3 = t2[1].split(' ')
                      removeListIDS.append(t3)
                      removeListIDS.append([child.attrib['id']])
            except:
              pass

    r = [item for sublist in removeListIDS for item in sublist]
    for i, child in enumerate(e):
        if child.attrib['id'] in r or child.tag == 'AnnotationHierarchyNode':
            indexToRemove.append(i)

    indexToRemove.sort()
    indexToRemove.reverse()

    for i in indexToRemove:
        e.remove(c[i])

    return e

  def removeDuplicateMRMLFromScene(self, e):
    c = list(e)
    indexToRemove = []
    keepIDS = []

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
                    if len(t2)>1:
                        t3 = t2[1].split(' ')
                        keepIDS.append(t3[0])
                        keepIDS.append([child.attrib['id']])
            except:
                pass

    for i, child in enumerate(e):
        if child.tag[:6] == 'Volume' and child.attrib['id'] not in keepIDS:
            indexToRemove.append(i)

    indexToRemove.sort()
    indexToRemove.reverse()

    for i in indexToRemove:
        e.remove(c[i])

    return e

  def removeAnnotationFromScene(self, e):
      c = list(e)
      indexToRemove = []
      for i, child in enumerate(e):
          if child.tag[:10] == 'Annotation':
              indexToRemove.append(i)

      indexToRemove.sort()
      indexToRemove.reverse()

      for i in indexToRemove:
          e.remove(c[i])

      return e

  ######################################################################################################################
  # ASSISTED MANUAL SEGMENTATION
  ######################################################################################################################

  def angle(self, v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

  def getOrientationVect(self, a,b):
      """ Returns the AB vector"""
      return np.array(a)-np.array(b)

  def getNeedleOrientation(self, needleNumber):
      """ Returns the vector define by lowest point | highest point of a needle"""
      controlPoints = self.sortNeedlePoints(needleNumber)

      return self.getOrientationVect(controlPoints[0], controlPoints[-1])

  def sortNeedlePoints(self, needleNumber):
      """ Returns Needle Lowest point"""
      # sort needle points
      nodes = slicer.util.getNodes('.%d-*' % needleNumber)
      controlPointsUnsorted = []
      for node in list(nodes.values()):
          p = [0,0,0]
          node.GetFiducialCoordinates(p)
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
      """ Return the needle that is the best aligned to the point"""
      nodes = slicer.util.getNodes('manual-seg_*')
      candidates = []
      validNeedles = self.findNeedles()
      for node in list(nodes.values()):
          name = node.GetName()
          nb = int(name.split('_')[1]) # get needle number
          if nb in validNeedles:
              lp = self.getNeedleLowestPoint(nb)
              if lp[2] > pt[2]:
                  candidates.append([name, self.angle(self.getOrientationVect(pt,lp), self.getNeedleOrientation(nb))])

      return candidates

  def findLowerNeedles(self, pt):
      """ Return the needle that is the best aligned to the point"""
      nodes = slicer.util.getNodes('manual-seg_*')
      candidates = []
      validNeedles = self.findNeedles()
      for node in list(nodes.values()):
          name = node.GetName()
          nb = int(name.split('_')[1]) # get needle number
          if nb in validNeedles:
              hp = self.getNeedleHighestPoint(nb)
              if hp[2] < pt[2]:
                  theta = self.angle(self.getNeedleOrientation(nb),-self.getOrientationVect(pt, hp))
                  candidates.append([name, min(theta,abs(theta-np.pi))])

      return candidates

  # findBestLowerNeedles(pt)
  # pt = [-15,-23,-29]

  def findCloseNeedles(self, pt, filterOutNeedles=[]):
      ''' Find the closest needle to the point, which is in not in the list filteroutneedles
      '''
      nodes = slicer.util.getNodes('manual-seg_*')
      candidates = []
      validNeedles = self.findNeedles()
      for node in list(nodes.values()):
          name = node.GetName()
          nb = int(name.split('_')[1]) # get needle number
          if nb in validNeedles and name not in filterOutNeedles:
              p = node.GetPolyData()
              x = self.getCenterLine(p)
              dists = []
              for q in x:
                  dists.append(self.distance(q,pt))
              candidates.append([name, min(dists)])

      return candidates

  def findMatchingCTLPoint(self, mo,pt):
      candidates = []
      for isolPt in self.findIsolatedCTLPoints():
          pt2 = [0,0,0]
          isolPt.GetFiducialCoordinates(pt2)
          a = self.angle(self.getOrientationVect(pt2,pt), mo)
          ori = min([a, abs(np.pi-a)])
          print(isolPt.GetName(), ori)
          if ori <0.1:
              candidates.append([int(isolPt.GetName().strip('.').split('-')[0]), ori])
      if not candidates: 'no matching point'
      return candidates

  def assignNeedle(self, pt):
      ''' Assign needle number to point
      '''
      candidate = None
      result = None
      lowest = None
      highest = None
      closest = None

      mo = self.meanOrientation()
      cLow = self.findLowerNeedles(pt)
      cHigh = self.findHigherNeedles(pt)
      cClose = self.findCloseNeedles(pt)

      if cLow: lowest = self.sortTable(cLow,(1,0))[0]
      if cHigh: highest = self.sortTable(cHigh,(1,0))[0]
      if cClose: closest = self.sortTable(cClose,(1,0))[0]

      if lowest and highest:
          candidate = self.sortTable([lowest,highest], (1,0))[0]
      elif lowest:
          candidate = lowest
      elif highest:
          candidate = highest

      if closest:
          if closest[1]<1:
              result = closest[0]
          elif closest[1]<2:
              if candidate and candidate[1]<0.3:
                  result = candidate[0]
              else:
                  result = closest[0]
          elif candidate and candidate[1]<0.3:
              result = candidate[0]

      elif candidate and candidate[1]<0.3:
          result = candidate[0]

      if result: result = int(result.strip('manual-seg_'))
      if not result:
          candidates = self.findMatchingCTLPoint(mo,pt)
          if candidates : result = self.sortTable(candidates,(1,0))[0][0]
      if not result: print('no good candidate to assign to -> new point')
      return result

  def findNeedles(self, all = None):
      needleNbs = []
      needles = slicer.util.getNodes('manual-seg_*')
      for needle in list(needles.values()):
          name = needle.GetName()
          nb = int(name.split('_')[1])
          points = slicer.util.getNodes('.%d-*'% nb)
          if len(points)>1 or all != None:
              needleNbs.append(nb)

      return needleNbs

  def meanOrientation(self):
      ''' Compute Mean orientation of needles'''
      orientation = 0
      c = 1
      for nb in self.findNeedles():
          c += 1
          orientation += self.getNeedleOrientation(nb)

      return orientation/c

  def findIsolatedCTLPoints(self):
      needles = slicer.util.getNodes('manual-seg_*')

      needleNbs = []
      isolatedPts = []
      for needle in list(needles.values()):
          name = needle.GetName()
          needleNbs.append(int(name.split('_')[1]))

      for nb in needleNbs:
          points = slicer.util.getNodes('.%d-*'% nb)
          if len(points)<2:
              isolatedPts.append(list(points.values())[0])

      return isolatedPts
