{
  description = "dev-env";
  nixConfig = {
    max-jobs = "auto";
    cores = 0;
    download-buffer-size = 538870932;
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
            pkgs.swig # required by box-2d gymnasium
          ];

          shellHook = ''
            echo "[ installed ] just - $(just --version | awk '{print $2}')"
            echo "[ installed ] uv - $(uv --version | awk '{print $2}')"
          '';
        };
      }
    );
}
