cmake_minimum_required(VERSION 3.10)
project(InterviewPrepCPP)

# Set modern C++ standard (C++17 is a good baseline for modern interviews)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 1. Find all practice.cpp files in any immediate subfolder
file(GLOB PRACTICE_FILES "*/practice.cpp")

# 2. Loop through each file found
foreach(SOURCE_FILE ${PRACTICE_FILES})

# Extract the folder name to use as the executable name
# e.g., from "memory_management/practice.cpp" -> "memory_management"
get_filename_component(DIR_PATH ${SOURCE_FILE} DIRECTORY)
get_filename_component(TARGET_NAME ${DIR_PATH} NAME)

# 3. Create an executable named after the folder
add_executable(${TARGET_NAME} ${SOURCE_FILE})

endforeach()