{
  description = "Lython: an LLVM/MLIR 23 based Python compiler toolchain";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";

    # Formatting tooling stays behind `formatter`/`checks` only. Referencing
    # it from `packages` or `devShells` would make `nix run`/`nix build` fetch
    # the tooling input again.
    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      ...
    }:
    let
      # nix-systems/default, inlined: four system names are not worth an input
      # every `nix run` user would fetch.
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;

      pkgsFor = system: import nixpkgs { inherit system; };

      treefmtEvalFor =
        system:
        treefmt-nix.lib.evalModule (pkgsFor system) {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true;
          settings.formatter = { };
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          llvmPackages = pkgs.llvmPackages_23;
          lython = pkgs.callPackage ./nix/build-lython.nix {
            llvm = llvmPackages.llvm;
            mlir = llvmPackages.mlir;
            tblgen = llvmPackages.tblgen;
          };
        in
        {
          default = lython;
          inherit lython;
        }
      );

      formatter = forAllSystems (system: (treefmtEvalFor system).config.build.wrapper);

      checks = forAllSystems (system: {
        treefmt = (treefmtEvalFor system).config.build.check self;
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          llvmPackages = pkgs.llvmPackages_23;
        in
        {
          default = pkgs.mkShell {
            packages = [
              llvmPackages.llvm
              llvmPackages.mlir
              # mlir-tblgen and llvm-tblgen live only in `tblgen`; nixpkgs's
              # mlir derivation does not install them into its `out`. Lython's
              # CMake looks them up on PATH at configure time, so the
              # devShell has to bring them in.
              llvmPackages.tblgen
              llvmPackages.clang
              llvmPackages.lld
              pkgs.cmake
              pkgs.ninja
              pkgs.gnumake
              pkgs.pkg-config
              pkgs.python3
              pkgs.uv
              pkgs.gdb
              # `lyc` shells out to the clang in llvm23 for AOT and WASI
              # linking; that clang locates the C++ runtime by finding gcc on
              # PATH.
              pkgs.gcc
              pkgs.binutils
            ];

            LLVM_DIR = "${llvmPackages.llvm.dev}/lib/cmake/llvm";
            MLIR_DIR = "${llvmPackages.mlir.dev}/lib/cmake/mlir";

            shellHook = ''
              export LLVM_DIR="${llvmPackages.llvm.dev}/lib/cmake/llvm"
              export MLIR_DIR="${llvmPackages.mlir.dev}/lib/cmake/mlir"
              echo "LLVM_DIR=$LLVM_DIR"
              echo "MLIR_DIR=$MLIR_DIR"
            '';
          };
        }
      );
    };
}
