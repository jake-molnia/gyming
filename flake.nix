{
  description = "dev-env";
  nixConfig = {
    # Use all available CPU cores for building
    max-jobs = "auto";
    cores = 0; # 0 means use all available cores

    # Increase download buffer for better network performance
    download-buffer-size = 538870932; # 256MB buffer

    # Additional performance optimizations
    keep-outputs = true;
    keep-derivations = true;
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.just
            pkgs.uv
          ];

          shellHook = ''
            echo "Ready!"
            echo "uv: $(uv --version)"
            echo "just: $(just --version)"
          '';
        };
      }
    );
}
