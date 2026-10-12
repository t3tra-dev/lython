{
  description = "Lython: an LLVM/MLIR 23 based Python compiler toolchain";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";

    # Dev-only inputs: referenced from `checks`/`formatter` below, never from
    # `packages`/`devShells`. Flake inputs are fetched lazily, so `nix run` and
    # `nix build` fetch nothing but nixpkgs. A flake-parts setup could not do
    # this: mkFlake forces its `imports`, dragging these five inputs into every
    # evaluation of `packages.default`. The follows also collapses the lock
    # from three nixpkgs revisions to one.
    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
    git-hooks-nix.url = "github:cachix/git-hooks.nix";
    git-hooks-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      git-hooks-nix,
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

      # Formatting and git-hook configs stay behind `formatter`/`checks` only.
      # Referencing them from `packages` or `devShells` would make `nix run`
      # fetch the tooling inputs again.
      treefmtEvalFor =
        system:
        treefmt-nix.lib.evalModule (pkgsFor system) {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true;
          settings.formatter = { };
        };

      preCommitFor =
        system:
        git-hooks-nix.lib.${system}.run {
          src = ./.;
          hooks = {
            treefmt = {
              enable = true;
              # The hook must format with this repo's generated treefmt config
              # (nixfmt), not a bare treefmt that would find no treefmt.toml.
              package = (treefmtEvalFor system).config.build.wrapper;
            };
            ripsecrets.enable = true;
            gitleaks = {
              enable = true;
              entry = "${(pkgsFor system).gitleaks}/bin/gitleaks protect --staged";
              language = "system";
            };
          };
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
        pre-commit = preCommitFor system;
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
