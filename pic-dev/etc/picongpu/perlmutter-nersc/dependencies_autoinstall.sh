#!/usr/bin/env bash
# Copyright 2023-2025 Axel Huebl, Marco Garten, Klaus Steiniger, Pawel Ordyna
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#
# last updated: 2025-03-10

PIC_BRANCH="dev"
PROJECT=$proj
echo $PROJECT

# get PIConGPU profile
if [ ! -f "$PIC_PROFILE" ]; then
    printf "Source a profile!\n"
    exit 1
else
    source "$PIC_PROFILE"
fi

set -euf -o pipefail
# create temporary directory for software source files
export SOURCE_DIR="$CFS/$PROJECT/$USER/lib_run_tmp"
mkdir -p $SOURCE_DIR
# Boost
if [ ! -d "$BOOST_ROOT" ]; then
    cd $SOURCE_DIR
    curl -L -s -o boost_1_87_0.tar.gz \
        https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz
    tar -xzf boost_1_87_0.tar.gz
    cd boost_1_87_0/
    ./bootstrap.sh --with-libraries=atomic,chrono,context,date_time,fiber,filesystem,math,program_options,serialization,system,thread --prefix=$BOOST_ROOT \
        CC=$(which cc) CXX=$(which CC)
    ./b2 cxxflags="-std=c++20" -j 10 && ./b2 install
fi

#   c-blosc
if [ ! -d "$BLOSC_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b v2.17.0 https://github.com/Blosc/c-blosc2.git \
        $SOURCE_DIR/c-blosc
    mkdir c-blosc-build
    cd c-blosc-build
    cmake -DCMAKE_INSTALL_PREFIX=$BLOSC_ROOT \
        -DMPI_C_COMPILER=cc -DMPI_CXX_COMPILER=CC \
        $SOURCE_DIR/c-blosc
    make -j 10 install
fi

#   PNGwriter
if [ ! -d "$PNGwriter_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b 0.7.0 https://github.com/pngwriter/pngwriter.git \
        $SOURCE_DIR/pngwriter
    mkdir pngwriter-build
    cd pngwriter-build
    cmake -DCMAKE_INSTALL_PREFIX=$PNGwriter_ROOT \
        $SOURCE_DIR/pngwriter
    make -j 10 install
fi

#   HDF5
if [ ! -d "$HDF5_ROOT" ]; then
    cd $SOURCE_DIR
    curl -Lo hdf5-1.14.6.tar.gz \
        https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_6/downloads/hdf5-1.14.6.tar.gz
    tar -xzf hdf5-1.14.6.tar.gz
    cd hdf5-1.14.6
    ./configure --enable-parallel --enable-shared --prefix $HDF5_ROOT CC=$(which cc) CXX=$(which CC)
    make -j 10 && make install
fi

#   ADIOS2
# force usage of MPI and HDF5 and point directly to MPI headers and libraries
if [ ! -d "$ADIOS2_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git \
        $SOURCE_DIR/adios2
        cd $SOURCE_DIR/adios2
        sed -i 's|if (ADIOS2_HAVE_MPI_CLIENT_SERVER)|if (TRUE)|' cmake/DetectOptions.cmake
    mkdir $SOURCE_DIR/adios2-build
    cd $SOURCE_DIR/adios2-build
    cmake $SOURCE_DIR/adios2 -DADIOS2_BUILD_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=$ADIOS2_ROOT -DADIOS2_USE_Fortran=OFF \
        -DADIOS2_USE_BZip2=OFF \
        -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON \
        -DMPI_CXX_COMPILER=$(which CC) -DMPI_C_COMPILER=$(which cc) \
        -DMPI_CXX_HEADER_DIR=${MPICH_DIR}/include \
        -DMPI_C_HEADER_DIR=${MPICH_DIR}/include \
        -DMPI_mpi_gnu_123_LIBRARY=${MPICH_DIR}/lib/libmpi_gnu_123.so
    make -j 10 && make install
fi

#   openPMD-api
if [ ! -d "OPENPMD_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b 0.16.1 https://github.com/openPMD/openPMD-api.git \
        $SOURCE_DIR/openpmd-api
    mkdir $SOURCE_DIR/openpmd-api-build
    cd $SOURCE_DIR/openpmd-api-build
    cmake $SOURCE_DIR/openpmd-api \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
        -DMPI_CXX_COMPILER=$(which CC) -DMPI_C_COMPILER=$(which cc) \
        -DMPI_CXX_HEADER_DIR=${MPICH_DIR}/include \
        -DMPI_C_HEADER_DIR=${MPICH_DIR}/include \
        -DMPI_mpi_gnu_123_LIBRARY=${MPICH_DIR}/lib/libmpi_gnu_123.so \
        -DCMAKE_INSTALL_PREFIX="$OPENPMD_ROOT"
    make -j 10 install
fi


# message to user
echo ''
echo 'edit user & email within picongpu.profile, e.g. via:'
echo '    vim $PIC_PROFILE'
echo 'delete temporary folder for library compilation'
printf "    rm -rf %s\n" $SOURCE_DIR
