#RUTA A LA LIBRERIA ITK (ESTA DEBE SER EDITADA SEGÚN LA UBICACIÓN EN DONDE SE INSTALO Y COMPILO ITK)
SET(ITK_DIR "/home/nicolas/Documents/InsightToolkit-5.1.2")

#MINIMA VERSION DE CMAKE QUE DESEAMOS USAR
cmake_minimum_required(VERSION 2.8)

#NOMBRE DEL PROYECTO 
project(cf)

#IMPORTAR ITK 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
else()
  find_package(ItkVtkGlue REQUIRED)
  include(${ItkVtkGlue_USE_FILE})
  set(Glue ItkVtkGlue)
endif()

add_executable(cf cf.cxx)

#ENLAZAR A LIBRERIA ITK
target_link_libraries(cf ${Glue} ${VTK_LIBRARIES} ${ITK_LIBRARIES})