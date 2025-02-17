{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.11";
  };
    
  outputs = { self, nixpkgs }:
  let 
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
  in {
    packages.x86_64-linux.build = pkgs.stdenv.mkDerivation {
      name = "lython";
      nativeBuildInputs = with pkgs; [
        boehmgc
        bun
        deno
        llvmPackages_19.clangUseLLVM
        pkg-config
        python313FreeThreading
        uv
      ];
    };

    packages.x86_64-linux.default = self.packages.x86_64-linux.build;
  };
}
