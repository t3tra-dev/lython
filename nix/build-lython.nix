{
  lib,
  stdenv,
  cmake,
  ninja,
  llvm,
  mlir,
  tblgen,
  zlib,
  zstd,
  libxml2,
  ncurses,
}:

stdenv.mkDerivation {
  pname = "lython";
  version = "0.1.0";

  src = ../.;

  nativeBuildInputs = [
    cmake
    ninja
    llvm
    mlir
    tblgen
  ];

  buildInputs = [
    llvm
    mlir
    zlib
    zstd
    libxml2
    ncurses
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    "-DLLVM_DIR=${llvm.dev}/lib/cmake/llvm"
    "-DMLIR_DIR=${mlir.dev}/lib/cmake/mlir"
    # nixpkgs's mlir does not install mlir-tblgen into its `out`; the tablegen
    # binary lives in `llvmPackages_23.tblgen`. Lython's CMake requires
    # mlir-tblgen (and llvm-tblgen) at configure time, so we hand it the
    # absolute paths here and skip the find_program search.
    "-DLLVM_TABLEGEN_EXE=${tblgen}/bin/llvm-tblgen"
    "-DMLIR_TABLEGEN_EXE=${tblgen}/bin/mlir-tblgen"
    # tests/CMakeLists.txt pulls googletest through FetchContent, which has no
    # network inside the Nix sandbox. The unit and golden suites are run from
    # the devShell, where the fetch can reach the network.
    "-DLYTHON_BUILD_TESTS=OFF"
  ];

  meta = {
    description = "An LLVM/MLIR 23 based Python compiler toolchain";
    homepage = "https://github.com/t3tra-dev/lython";
    # Without this `nix run` looks for bin/lython (the pname) and fails; the
    # installed binary is lyc.
    mainProgram = "lyc";
    platforms = [ "x86_64-linux" ];
  };
}
