{
  lib,
  stdenv,
  fetchurl,
  cmake,
  ninja,
  python3,
  zlib,
  zstd,
  libxml2,
  ncurses,
  llvm23,
}:

let
  version = "23.1.2";
in
stdenv.mkDerivation (finalAttrs: {
  pname = "mlir";
  inherit version;

  # Lython's CMake requires the shared `MLIR` CMake target, which MLIR only
  # exports from a BUILD_SHARED_LIBS=ON build. LLVM refuses that flag
  # alongside LLVM_LINK_LLVM_DYLIB, and nixpkgs' MLIR builds static-only, so
  # MLIR is built standalone here against the shared LLVM 23 above.
  src = fetchurl {
    url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-${version}/llvm-project-${version}.src.tar.xz";
    hash = "sha256-yYu+8IorTCYTzVDpqprntpsf5sFrLEA3O8CrYRb994o=";
  };

  sourceRoot = "llvm-project-${version}.src/mlir";

  # MLIR's standalone detection looks one directory up for an LLVM source
  # tree; a full one there makes it treat the absent in-tree LLVM build as
  # the tablegen source and no generated header appears. nixpkgs leaves a
  # blank `llvm/` for the same reason.
  postUnpack = ''
    rm -rf llvm-project-${version}.src/llvm/*
  '';

  nativeBuildInputs = [
    cmake
    ninja
    python3
  ];

  buildInputs = [
    llvm23
    zlib
    zstd
    libxml2
    ncurses
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "LLVM_DIR" "${llvm23}/lib/cmake/llvm")
    (lib.cmakeFeature "LLVM_TABLEGEN_EXE" "${llvm23}/bin/llvm-tblgen")
    (lib.cmakeBool "LLVM_BUILD_LLVM_DYLIB" true)
    (lib.cmakeBool "LLVM_LINK_LLVM_DYLIB" true)
    (lib.cmakeBool "LLVM_ENABLE_RTTI" true)
    (lib.cmakeBool "LLVM_ENABLE_ASSERTIONS" false)
    (lib.cmakeBool "LLVM_ENABLE_LIBXML2" true)
    (lib.cmakeBool "LLVM_ENABLE_ZLIB" true)
    (lib.cmakeBool "LLVM_ENABLE_ZSTD" true)
    (lib.cmakeBool "MLIR_INCLUDE_TESTS" false)
    (lib.cmakeBool "MLIR_ENABLE_BINDINGS_PYTHON" false)
    (lib.cmakeBool "MLIR_BUILD_EXAMPLES" false)
    # add_tablegen() only installs mlir-tblgen when LLVM_BUILD_UTILS is set;
    # Lython's CMake looks it up by name on PATH.
    (lib.cmakeBool "LLVM_BUILD_UTILS" true)
    (lib.cmakeBool "LLVM_INSTALL_TOOLCHAIN_ONLY" false)
  ];

  requiredSystemFeatures = [ "big-parallel" ];

  meta = {
    description = "MLIR ${version} built against LLVM ${version}";
    homepage = "https://mlir.llvm.org/";
    license = lib.licenses.asl20;
    platforms = lib.platforms.unix;
  };
})
