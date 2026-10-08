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
  glibc,
  makeWrapper,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "llvm";
  version = "23.1.2";

  # The nixpkgs LLVM 21 packaging cannot be pointed at 23: its
  # gnu-install-dirs patch no longer applies (llvm_setup_rpath moved), and
  # Lython needs the shared `LLVM` and `MLIR` CMake targets that a plain
  # nixpkgs MLIR build does not export. Building the release monorepo
  # directly keeps LLVM and MLIR in one tree with the exact configuration
  # Lython's CMakeLists asks for.
  src = fetchurl {
    url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-${finalAttrs.version}/llvm-project-${finalAttrs.version}.src.tar.xz";
    hash = "sha256-yYu+8IorTCYTzVDpqprntpsf5sFrLEA3O8CrYRb994o=";
  };

  sourceRoot = "llvm-project-${finalAttrs.version}.src/llvm";

  nativeBuildInputs = [
    cmake
    ninja
    python3
    makeWrapper
  ];

  buildInputs = [
    zlib
    zstd
    libxml2
    ncurses
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    # clang is the linker driver `lyc` shells out to for AOT and for the WASI
    # target, and lld supplies wasm-ld; both must be the LLVM lyc is linked
    # against, which is also why they live in this derivation.
    (lib.cmakeFeature "LLVM_ENABLE_PROJECTS" "clang;lld")
    (lib.cmakeFeature "LLVM_TARGETS_TO_BUILD" "X86;AArch64;WebAssembly")
    (lib.cmakeBool "LLVM_BUILD_LLVM_DYLIB" true)
    (lib.cmakeBool "LLVM_LINK_LLVM_DYLIB" true)
    (lib.cmakeBool "LLVM_ENABLE_ASSERTIONS" false)
    (lib.cmakeBool "LLVM_ENABLE_DUMP" true)
    (lib.cmakeBool "LLVM_ENABLE_RTTI" true)
    (lib.cmakeBool "LLVM_ENABLE_LIBXML2" true)
    (lib.cmakeBool "LLVM_ENABLE_ZLIB" true)
    (lib.cmakeBool "LLVM_ENABLE_ZSTD" true)
    (lib.cmakeBool "LLVM_ENABLE_TERMINFO" true)
    (lib.cmakeBool "LLVM_ENABLE_SPHINX" false)
    (lib.cmakeBool "LLVM_INCLUDE_TESTS" false)
    (lib.cmakeBool "LLVM_INCLUDE_BENCHMARKS" false)
    (lib.cmakeBool "LLVM_BUILD_TOOLS" true)
    (lib.cmakeBool "LLVM_INSTALL_UTILS" true)
    (lib.cmakeBool "LLVM_ENABLE_BINDINGS" false)
    (lib.cmakeBool "LLVM_ENABLE_IDE" false)
    (lib.cmakeBool "LLVM_BUILD_EXAMPLES" false)
    (lib.cmakeBool "LLVM_INSTALL_TOOLCHAIN_ONLY" false)
  ];

  # A bare clang in Nix cannot find the C++ runtime: it does not probe the
  # gcc on PATH (that gcc is a wrapper script, and glibc lives in its own
  # store path anyway), so AOT linking fails on crt1.o / libstdc++. The
  # wrapper hands clang the same four things nixpkgs' cc-wrapper does: the
  # GCC installation directory, its libstdc++, glibc's headers and its
  # start files. `--inherit-argv0` keeps clang++ in C++ driver mode.
  postFixup = ''
    gccInstallDir=$(echo ${stdenv.cc.cc}/lib/gcc/*/*)
    ldso=$(echo ${glibc}/lib/ld-linux*.so.2)
    for tool in clang clang++; do
      wrapProgram "$out/bin/$tool" \
        --inherit-argv0 \
        --add-flags "--gcc-install-dir=$gccInstallDir" \
        --add-flags "-isystem ${glibc.dev}/include" \
        --add-flags "-B${glibc}/lib" \
        --add-flags "-L${glibc}/lib" \
        --add-flags "-L${stdenv.cc.cc.lib}/lib" \
        --add-flags "-fuse-ld=$out/bin/ld.lld" \
        --add-flags "-Wl,--dynamic-linker=$ldso" \
        --add-flags "-Wl,-rpath,${glibc}/lib" \
        --add-flags "-Wl,-rpath,${stdenv.cc.cc.lib}/lib"
    done
  '';

  requiredSystemFeatures = [ "big-parallel" ];

  meta = {
    description = "LLVM ${finalAttrs.version}, shared library with exported CMake packages";
    homepage = "https://llvm.org";
    license = lib.licenses.asl20;
    platforms = lib.platforms.unix;
  };
})
