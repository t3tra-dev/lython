{
  lib,
  llvmPackages_23,
  cmake,
  ninja,
  python3,
  makeWrapper,
  gtest,
  withTests ? false,
}:

let
  llvm = llvmPackages_23;
in
# ⛔ Not the default stdenv: on Linux that is GCC, which builds this tree but
# is not what any other build of it uses. llvmPackages_23's stdenv is the
# clang that matches the LLVM it links against, on every system.
llvm.stdenv.mkDerivation {
  pname = "lython";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions (
      [
        ../CMakeLists.txt
        ../src
        ../tools
      ]
      ++ lib.optionals withTests [
        ../tests
        ../examples
        # ModelCorrespondenceTest reads the proof's syntax.
        ../proof
      ]
    );
  };

  nativeBuildInputs = [
    cmake
    ninja
    python3
    makeWrapper
    llvm.tblgen
  ];

  # nixpkgs' LLVM and MLIR, from cache.nixos.org: lyc links them as shared
  # libraries out of the store, the same way it links Homebrew's or apt's.
  buildInputs = [
    llvm.llvm
    llvm.mlir
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "LLVM_DIR" "${lib.getDev llvm.llvm}/lib/cmake/llvm")
    (lib.cmakeFeature "MLIR_DIR" "${lib.getDev llvm.mlir}/lib/cmake/mlir")
    (lib.cmakeBool "LYTHON_BUILD_TESTS" withTests)
  ]
  ++ lib.optionals withTests [
    # tests/CMakeLists.txt fetches googletest, and the sandbox has no network.
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_GOOGLETEST" "${gtest.src}")
  ];

  doCheck = withTests;
  # The golden suites link executables (clang, lld) like a user's lyc does.
  nativeCheckInputs = [
    llvm.clang
    llvm.lld
  ];
  checkPhase = ''
    runHook preCheck
    ctest -j$NIX_BUILD_CORES --output-on-failure -LE bench
    runHook postCheck
  '';

  # lyc links an AOT executable by running clang (and lld for other targets):
  # the clang that matches its LLVM, wrapped so that it finds this system's
  # libc and C++ runtime.
  postFixup = ''
    wrapProgram $out/bin/lyc \
      --prefix PATH : ${
        lib.makeBinPath [
          llvm.clang
          llvm.lld
        ]
      }
  '';

  meta = {
    description = "An LLVM/MLIR 23 based Python compiler toolchain";
    homepage = "https://github.com/t3tra-dev/lython";
    mainProgram = "lyc";
    platforms = [
      "aarch64-darwin"
      "aarch64-linux"
      "x86_64-linux"
    ];
  };
}
