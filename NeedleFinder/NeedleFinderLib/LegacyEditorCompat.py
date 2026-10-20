"""Compatibility layer for legacy Slicer 4 Editor/LabelStatistics APIs on Slicer 5+."""

import ctk
import numpy as np
import SimpleITK as sitk
import sitkUtils
import slicer


def get_node_or_none(pattern=None, index=0, scene=None):
    """Return a MRML node matching `pattern`, or `None` if not found (Slicer 5 safe)."""
    try:
        if scene is not None:
            return slicer.util.getNode(pattern=pattern, index=index, scene=scene)
        return slicer.util.getNode(pattern=pattern, index=index)
    except slicer.util.MRMLNodeNotFoundException:
        return None


try:
    import EditorLib
    from Editor import EditorWidget
except ImportError:

    class _DummyUndoRedo:
        """No-op undo/redo state tracker for compatibility."""

        def saveState(self):
            pass

    class _DummyToolsBox:
        """Compatibility container for EditorWidget.toolsBox."""

        def __init__(self):
            self.actions = {}
            self.undoRedo = _DummyUndoRedo()

    class _CompatEditUtil:
        """Slicer 5 replacement for legacy EditorLib.EditUtil."""

        PARAMETER_NODE_NAME = "EditorParameterNode"

        def getParameterNode(self):
            node = get_node_or_none(self.PARAMETER_NODE_NAME)
            if not node:
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLScriptedModuleNode", self.PARAMETER_NODE_NAME
                )
                node.SetSingletonTag(self.PARAMETER_NODE_NAME)
            return node

        def getSliceLogic(self):
            lm = slicer.app.layoutManager()
            if lm and lm.sliceWidget("Red"):
                return lm.sliceWidget("Red").sliceLogic()
            return None

        def getLabel(self):
            val = self.getParameterNode().GetParameter("label")
            return int(val) if val else 1

        def setLabel(self, label):
            self.getParameterNode().SetParameter("label", str(int(label)))

        def setCurrentEffect(self, effect_name):
            self.getParameterNode().SetParameter("effect", str(effect_name))

        def markVolumeNodeAsModified(self, volume_node):
            if volume_node:
                if volume_node.GetImageData():
                    volume_node.GetImageData().Modified()
                volume_node.Modified()

    class _CompatEffectOptions:
        """No-op effect options container for compatibility."""

        def setMRMLDefaults(self):
            pass

        def __del__(self):
            pass

    class _CompatWandEffectLogic:
        """SimpleITK ConnectedThreshold implementation of the legacy WandEffectLogic."""

        def __init__(self, slice_logic=None):
            self.sliceLogic = slice_logic
            self.undoRedo = _DummyUndoRedo()
            self.editUtil = _CompatEditUtil()

        def _xy_to_ijk(self, xy, layer_logic):
            if not layer_logic:
                return [0, 0, 0]
            xy_to_ijk = layer_logic.GetXYToIJKTransform()
            ijk = xy_to_ijk.TransformDoublePoint((xy[0], xy[1], 0.0))
            return [int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))]

        def labelAtXY(self, xy):
            if not self.sliceLogic:
                return 0
            label_layer = self.sliceLogic.GetLabelLayer()
            label_node = label_layer.GetVolumeNode() if label_layer else None
            if not label_node or not label_node.GetImageData():
                return 0
            ijk = self._xy_to_ijk(xy, label_layer)
            dims = label_node.GetImageData().GetDimensions()
            if 0 <= ijk[0] < dims[0] and 0 <= ijk[1] < dims[1] and 0 <= ijk[2] < dims[2]:
                return int(
                    label_node.GetImageData().GetScalarComponentAsDouble(
                        ijk[0], ijk[1], ijk[2], 0
                    )
                )
            return 0

        def apply(self, xy):
            if not self.sliceLogic:
                return
            bg_layer = self.sliceLogic.GetBackgroundLayer()
            label_layer = self.sliceLogic.GetLabelLayer()
            bg_node = bg_layer.GetVolumeNode() if bg_layer else None
            label_node = label_layer.GetVolumeNode() if label_layer else None
            if not bg_node or not label_node:
                return
            ijk = self._xy_to_ijk(xy, bg_layer)
            dims = bg_node.GetImageData().GetDimensions()
            if not (0 <= ijk[0] < dims[0] and 0 <= ijk[1] < dims[1] and 0 <= ijk[2] < dims[2]):
                return
            param_node = self.editUtil.getParameterNode()
            tol = float(param_node.GetParameter("WandEffect,tolerance") or 20.0)
            seed_val = bg_node.GetImageData().GetScalarComponentAsDouble(
                ijk[0], ijk[1], ijk[2], 0
            )
            bg_sitk = sitkUtils.PullVolumeFromSlicer(bg_node)
            mask_sitk = sitk.ConnectedThreshold(
                bg_sitk,
                seedList=[(int(ijk[0]), int(ijk[1]), int(ijk[2]))],
                lower=float(seed_val - tol),
                upper=float(seed_val + tol),
                replaceValue=1,
            )
            mask_arr = sitk.GetArrayFromImage(mask_sitk)
            label_arr = slicer.util.arrayFromVolume(label_node)
            label_arr[mask_arr > 0] = self.editUtil.getLabel()
            slicer.util.arrayFromVolumeModified(label_node)

    class _CompatIdentifyIslandsEffectLogic:
        """SimpleITK ConnectedComponent implementation of IdentifyIslandsEffectLogic."""

        def __init__(self, slice_logic=None):
            self.sliceLogic = slice_logic
            self.editUtil = _CompatEditUtil()

        def removeIslands(self):
            selection_node = slicer.app.applicationLogic().GetSelectionNode()
            label_id = selection_node.GetActiveLabelVolumeID()
            label_node = slicer.mrmlScene.GetNodeByID(label_id) if label_id else None
            if not label_node:
                return
            param_node = self.editUtil.getParameterNode()
            min_size = int(param_node.GetParameter("IslandEffect,minimumSize") or 0)
            label_sitk = sitkUtils.PullVolumeFromSlicer(label_node)
            cc = sitk.ConnectedComponent(label_sitk > 0)
            relabeled = sitk.RelabelComponent(cc, minimumObjectSize=max(1, min_size))
            sitkUtils.PushVolumeToSlicer(
                sitk.Cast(relabeled, label_sitk.GetPixelID()), label_node
            )

    class EditorLib:
        """Compatibility namespace for legacy EditorLib."""

        class EditUtil:
            EditUtil = _CompatEditUtil

        WandEffectOptions = _CompatEffectOptions
        IdentifyIslandsEffectOptions = _CompatEffectOptions
        WandEffectLogic = _CompatWandEffectLogic
        IdentifyIslandsEffectLogic = _CompatIdentifyIslandsEffectLogic

    class EditorWidget:
        """Compatibility widget replacing legacy Slicer 4 EditorWidget."""

        def __init__(self, parent=None, showVolumesFrame=False):
            self.parent = parent
            self.showVolumesFrame = showVolumesFrame
            self.editUtil = _CompatEditUtil()
            self.toolsBox = _DummyToolsBox()
            self.editLabelMapsFrame = ctk.ctkCollapsibleButton(self.parent)
            self.editLabelMapsFrame.text = "Edit Segmentation"
            self.editLabelMapsFrame.collapsed = True
            if self.parent and self.parent.layout():
                self.parent.layout().addWidget(self.editLabelMapsFrame)
            self.masterNode = None
            self.mergeNode = None

        def setup(self):
            pass

        def setMasterNode(self, node):
            self.masterNode = node

        def setMergeNode(self, node):
            self.mergeNode = node


try:
    from LabelStatistics import LabelStatisticsLogic
except ImportError:

    class LabelStatisticsLogic:
        """Compute label volume statistics (mm^3) using NumPy."""

        def __init__(self, master_volume, label_volume):
            self.masterVolume = master_volume
            self.labelVolume = label_volume
            self.labelStats = {"Labels": []}
            if not label_volume or not label_volume.GetImageData():
                return
            spacing = label_volume.GetSpacing()
            voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
            arr = slicer.util.arrayFromVolume(label_volume)
            unique_labels, counts = np.unique(arr, return_counts=True)
            for lbl, cnt in zip(unique_labels, counts):
                lbl_int = int(lbl)
                self.labelStats["Labels"].append(lbl_int)
                self.labelStats[(lbl_int, "Volume mm^3")] = float(cnt) * voxel_vol


class Helper:
    """Helper utility for setting background/foreground slice viewer volumes."""

    @staticmethod
    def SetBgFgVolumes(bg_id, fg_id):
        slicer.util.setSliceViewerLayers(background=bg_id, foreground=fg_id)
