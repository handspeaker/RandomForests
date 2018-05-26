cmake_minimum_required(VERSION 3.10)
project(RandomForests_)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(RandomForests_
        csv_reader.h
        main.cpp
        MnistPreProcess.cpp
        MnistPreProcess.h
        Node.cpp
        Node.h
        RandomForest.cpp
        RandomForest.h
        Sample.cpp
        Sample.h
        Tree.cpp
        Tree.h)
