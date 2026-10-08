{
  description = "Lython: an LLVM/MLIR 23 based Python compiler toolchain";

  # `nix run github:t3tra-dev/lython` otherwise compiles two LLVM trees from
  # source; this lets a first-time user accept the binary cache instead.
  nixConfig = {
    extra-substituters = [ "https://comamoca-lython.cachix.org" ];
    extra-trusted-public-keys = [
      "comamoca-lython.cachix.org-1:AsNMU2Rio0MerjVsy2UJgjt6cP/ywp7tk0peqoxB1VU="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    git-hooks-nix.url = "github:cachix/git-hooks.nix";
  };

  outputs =
    inputs@{
      self,
      systems,
      nixpkgs,
      flake-parts,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.treefmt-nix.flakeModule
        inputs.git-hooks-nix.flakeModule
      ];
      systems = import inputs.systems;

      perSystem =
        {
          config,
          pkgs,
          system,
          ...
        }:
        let
          # Lython hard-requires LLVM/MLIR major 23 (LYTHON_LLVM_MAJOR_VERSION
          # in CMakeLists.txt); nixpkgs ships 21. The overlay drops the
          # nixpkgs-built toolchain in under the name the rest of the flake
          # (and downstream users) expects.
          llvm23Overlay = final: prev: {
            llvm23 = final.callPackage ./nix/llvm23.nix { };
            mlir23 = final.callPackage ./nix/mlir23.nix { inherit (final) llvm23; };
          };

          lyPkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ llvm23Overlay ];
          };

          inherit (lyPkgs) llvm23 mlir23;

          lython = lyPkgs.callPackage ./nix/build-lython.nix { inherit llvm23 mlir23; };
        in
        {
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixfmt.enable = true;
            };

            settings.formatter = { };
          };

          pre-commit = {
            check.enable = true;
            settings = {
              hooks = {
                treefmt.enable = true;
                ripsecrets.enable = true;
                gitleaks = {
                  enable = true;
                  entry = "${pkgs.gitleaks}/bin/gitleaks protect --staged";
                  language = "system";
                };
              };
            };
          };

          packages = {
            default = lython;
            lython = lython;
            llvm23 = llvm23;
            mlir23 = mlir23;
          };

          devShells.default = pkgs.mkShell {
            packages = [
              # The built compiler itself, so `lyc jit` / `lyc foo.py` work
              # straight from `nix develop`.
              lython
              llvm23
              mlir23
              pkgs.cmake
              pkgs.ninja
              pkgs.gnumake
              pkgs.pkg-config
              pkgs.python3
              pkgs.uv
              pkgs.gdb
              # `lyc` shells out to the clang in llvm23/bin for AOT and WASI
              # linking; that clang locates the C++ runtime by finding gcc on
              # PATH.
              pkgs.gcc
              pkgs.binutils
            ];

            LLVM_DIR = "${llvm23}/lib/cmake/llvm";
            MLIR_DIR = "${mlir23}/lib/cmake/mlir";

            shellHook = ''
              export LLVM_DIR="${llvm23}/lib/cmake/llvm"
              export MLIR_DIR="${mlir23}/lib/cmake/mlir"
              echo "LLVM_DIR=$LLVM_DIR"
              echo "MLIR_DIR=$MLIR_DIR"
            '';
          };
        };
    };
}
