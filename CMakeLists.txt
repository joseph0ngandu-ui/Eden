cmake_minimum_required(VERSION 3.22)
project(Eden VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Qt6 modules
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Quick QuickControls2 Charts Network Sql)

# Find ZeroMQ for IPC
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Find ONNX Runtime for GPU acceleration
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "/usr/include/onnxruntime" "/usr/local/include/onnxruntime")
find_library(ONNXRUNTIME_LIB NAMES onnxruntime PATHS "/usr/lib" "/usr/local/lib")

# Qt6 auto tools
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Source files
set(SOURCES
    ui/src/main.cpp
    ui/src/edenapp.cpp
    ui/src/backtestmanager.cpp
    ui/src/gpumanager.cpp
    ui/src/workermanager.cpp
    ui/src/chartcanvas.cpp
    ui/src/thememanager.cpp
)

# Header files
set(HEADERS
    ui/include/edenapp.h
    ui/include/backtestmanager.h
    ui/include/gpumanager.h
    ui/include/workermanager.h
    ui/include/chartcanvas.h
    ui/include/thememanager.h
)

# QML resource file
qt6_add_resources(QML_RESOURCES ui/qml/qml.qrc)

# Create executable
qt6_add_executable(Eden ${SOURCES} ${HEADERS} ${QML_RESOURCES})

# Link Qt6 libraries
target_link_libraries(Eden
    Qt6::Core
    Qt6::Widgets  
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::Charts
    Qt6::Network
    Qt6::Sql
)

# Link ZeroMQ
target_include_directories(Eden PRIVATE ${ZMQ_INCLUDE_DIRS})
target_link_directories(Eden PRIVATE ${ZMQ_LIBRARY_DIRS})
target_link_libraries(Eden ${ZMQ_LIBRARIES})

# Link ONNX Runtime if available
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    target_include_directories(Eden PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(Eden ${ONNXRUNTIME_LIB})
    target_compile_definitions(Eden PRIVATE ONNX_AVAILABLE)
endif()

# Include directories
target_include_directories(Eden PRIVATE ui/include)

# Windows specific settings
if(WIN32)
    # Enable Windows 10/11 styling
    target_compile_definitions(Eden PRIVATE WIN32_LEAN_AND_MEAN)
    
    # Set executable properties
    set_target_properties(Eden PROPERTIES
        WIN32_EXECUTABLE TRUE
        OUTPUT_NAME "Eden"
    )
    
    # Add Windows resources
    target_sources(Eden PRIVATE ui/resources/eden.rc)
endif()

# macOS specific settings
if(APPLE)
    set_target_properties(Eden PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/ui/resources/Info.plist
    )
endif()

# Copy Python worker to build directory
add_custom_target(copy_python_workers ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/worker/python
        ${CMAKE_BINARY_DIR}/worker/python
)

# Copy shared data
add_custom_target(copy_shared_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shared
        ${CMAKE_BINARY_DIR}/shared
)

# Copy data directory
add_custom_target(copy_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data
        ${CMAKE_BINARY_DIR}/data
)

add_dependencies(Eden copy_python_workers copy_shared_data copy_data)

# Installation rules
if(WIN32)
    # Install to Program Files on Windows
    install(TARGETS Eden RUNTIME DESTINATION bin)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/worker DESTINATION .)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/shared DESTINATION .)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/data DESTINATION .)
endif()