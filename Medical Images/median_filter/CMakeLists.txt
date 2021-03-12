#RUTA A LA LIBRERIA ITK (ESTA DEBE SER EDITADA SEGÚN LA UBICACIÓN EN DONDE SE INSTALO Y COMPILO ITK)
SET(ITK_DIR "/home/nicolas/Documents/InsightToolkit-5.1.2")

#MINIMA VERSION DE CMAKE QUE DESEAMOS USAR
cmake_minimum_required(VERSION 2.8)

#NOMBRE DEL PROYECTO 
project(median)

#IMPORTAR ITK 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(median median.cxx)

#ENLAZAR A LIBRERIA ITK
target_link_libraries(median ${ITK_LIBRARIES})