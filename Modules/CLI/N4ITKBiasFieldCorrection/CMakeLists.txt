
#-----------------------------------------------------------------------------
set(MODULE_NAME N4ITKBiasFieldCorrection) # Do not use 'project()'

#-----------------------------------------------------------------------------
SEMMacroBuildCLI(
  NAME ${MODULE_NAME}
  LOGO_HEADER ${Slicer_SOURCE_DIR}/Resources/NAMICLogo.h
  #TARGET_LIBRARIES vtkITK ITKVtkGlue ${ITK_LIBRARIES}
  TARGET_LIBRARIES vtkITK ${ITK_LIBRARIES}
  INCLUDE_DIRECTORIES
    ${vtkITK_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
