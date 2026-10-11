{
  description = "Lython: an LLVM/MLIR 23 based Python compiler toolchain";

  # nixpkgs' own llvmPackages_23, which Hydra builds and cache.nixos.org
  # serves: no LLVM is built here and no other binary cache is needed.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs }:
    let
      # ⛔ No x86_64-darwin: nixpkgs no longer builds it, so its LLVM 23
      # would be a source build on every user's machine.
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: rec {
        lython = pkgs.callPackage ./nix/package.nix { };
        default = lython;
      });

      # `nix flake check` builds with the ctest suite and runs it, bench
      # tests aside.
      checks = forAllSystems (pkgs: {
        lython = pkgs.callPackage ./nix/package.nix { withTests = true; };
      });

      devShells = forAllSystems (
        pkgs:
        let
          llvm = pkgs.llvmPackages_23;
        in
        {
          # `cmake -B build -S .` as in the README; llvm-config on PATH is how
          # CMakeLists.txt finds LLVM, and MLIR_DIR names the other store path.
          default = pkgs.mkShell.override { stdenv = llvm.stdenv; } {
            packages = [
              llvm.llvm.dev
              llvm.mlir.dev
              llvm.tblgen
              llvm.clang
              llvm.lld
              pkgs.cmake
              pkgs.ninja
              pkgs.python3
              pkgs.uv
            ];
            MLIR_DIR = "${llvm.mlir.dev}/lib/cmake/mlir";
          };
        }
      );

      formatter = forAllSystems (pkgs: pkgs.nixfmt);
    };
}
