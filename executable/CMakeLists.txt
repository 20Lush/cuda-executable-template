cmake_minimum_required(VERSION 3.22)

# Lets initialize the project for our actual executable
project(CUDA_EXE_NAME
    VERSION ${PROJECT_VERSION}
    DESCRIPTION "CUDA EXECUTABLE Project"
    LANGUAGES CUDA CXX
)
message(STATUS "Project name: " ${PROJECT_NAME})

# Put some versioning info into our runtime environment
configure_file(
    "${PROJECT_SOURCE_DIR}/include/version.hpp.in"
    "${PROJECT_SOURCE_DIR}/include/version.hpp"
)

# Define what source files we actually are going to compile
set(SOURCE_FILES
    src/kernel.cu
    src/main.cpp
)

# Define what header files are going to be exposed(only put interfaces!)
set(PUBLIC_INCLUDE_FILES
    include/main.hpp
    include/kernel.hpp
)

# Do a CMake add_executable to mark it for build
add_executable( ${PROJECT_NAME}
    ${SOURCE_FILES}
)

# There might be some includes we want at build-time here
include_directories("${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")

# The installed directory is going to have its own include directory.
# In the unlikely scenario this executable is used in another project,
# they'll need the public headers.
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

find_package(CUDAToolkit REQUIRED)

# Put any compile features we'd want in here, like language standards
target_compile_features(${PROJECT_NAME}
    PUBLIC
        cxx_std_20
)

# Set some target properties of the installed binaries, liked if they have position independent code
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        LINKER_LANGUAGE CXX
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
)

# Link the libraries we want, like OpenCV or the cuda toolkit
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        # any dependency library we are linking...
        CUDA::cudart
        CUDA::cuda_driver
)

# Send to boilerplate!
message(STATUS "Installing executable to " ${CMAKE_INSTALL_PREFIX} "/...")
# Found in INSTALLER.cmake
DefaultInstall(${PROJECT_NAME})

AddUnitTestsDirectory(test)