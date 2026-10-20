#-----------------------------------------------------------------------------
set(MODULE_NAME NeedleFinder)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/AssistedSegmentation.py
  ${MODULE_NAME}Lib/Constants.py
  ${MODULE_NAME}Lib/DeflectionModelData.py
  ${MODULE_NAME}Lib/GeometryUtils.py
  ${MODULE_NAME}Lib/LegacyEditorCompat.py
  ${MODULE_NAME}Lib/ProfilingUtils.py
  ${MODULE_NAME}Lib/SceneAndEvaluation.py
  ${MODULE_NAME}Lib/TrackingAlgorithms.py
  )

set(MODULE_PYTHON_RESOURCES
  Config/default.cfg
  Config/miccai15.cfg
  Config/miccai2013.cfg
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )

