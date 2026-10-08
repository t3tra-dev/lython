{
  lib,
  stdenv,
  cmake,
  ninja,
  llvm23,
  mlir23,
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
    llvm23
    mlir23
  ];

  buildInputs = [
    llvm23
    mlir23
    zlib
    zstd
    libxml2
    ncurses
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    "-DLLVM_DIR=${llvm23}/lib/cmake/llvm"
    "-DMLIR_DIR=${mlir23}/lib/cmake/mlir"
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
