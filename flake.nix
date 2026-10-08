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

      # Lython hard-requires LLVM/MLIR major 23 (LYTHON_LLVM_MAJOR_VERSION
      # in CMakeLists.txt); nixpkgs ships 21. The overlay drops the
      # nixpkgs-built toolchain in under the name the rest of the flake
      # (and downstream users) expects.
      llvm23Overlay = final: prev: {
        llvm23 = final.callPackage ./nix/llvm23.nix { };
        mlir23 = final.callPackage ./nix/mlir23.nix { inherit (final) llvm23; };
      };

      lyPkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ llvm23Overlay ];
        };

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
          lyPkgs = lyPkgsFor system;
          lython = lyPkgs.callPackage ./nix/build-lython.nix {
            inherit (lyPkgs) llvm23 mlir23;
          };
        in
        {
          default = lython;
          inherit lython;
          llvm23 = lyPkgs.llvm23;
          mlir23 = lyPkgs.mlir23;
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
          lyPkgs = lyPkgsFor system;
        in
        {
          default = pkgs.mkShell {
            packages = [
              lyPkgs.llvm23
              lyPkgs.mlir23
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

            LLVM_DIR = "${lyPkgs.llvm23}/lib/cmake/llvm";
            MLIR_DIR = "${lyPkgs.mlir23}/lib/cmake/mlir";

            shellHook = ''
              export LLVM_DIR="${lyPkgs.llvm23}/lib/cmake/llvm"
              export MLIR_DIR="${lyPkgs.mlir23}/lib/cmake/mlir"
              echo "LLVM_DIR=$LLVM_DIR"
              echo "MLIR_DIR=$MLIR_DIR"
            '';
          };
        }
      );
    };
}
