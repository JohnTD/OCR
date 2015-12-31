cmake_minimum_required ( VERSION 2.8 )
project ( OCR )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

aux_source_directory( . DIR_SRCS )
add_executable( OCR ${DIR_SRCS} )

target_link_libraries( OCR ${OpenCV_LIBS} )
