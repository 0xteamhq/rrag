# Vendored Schemars

This is a vendored copy of the `schemars` crate (https://github.com/GREsau/schemars).

## Why Vendored?

We vendor schemars locally to:
1. **Full Control**: Ability to customize and extend as needed
2. **Stability**: Lock to a specific version that works with our tooling
3. **Local Modifications**: Can add custom features specific to RSLLM
4. **Build Reproducibility**: No external dependency changes
5. **Offline Development**: Work without network access

## Version

Cloned from: https://github.com/GREsau/schemars
Version: 0.8.x (see schemars/Cargo.toml for exact version)
Date: 2025-10-28

## License

Schemars is MIT licensed (see LICENSE file in parent directory)

## Original Repository

For upstream updates and documentation, see:
- Repository: https://github.com/GREsau/schemars
- Docs: https://docs.rs/schemars
- Website: https://graham.cool/schemars

## Modifications

Currently using the original implementation without modifications.
Any future customizations will be documented here.

## Updating

To update to a newer version of schemars:
```bash
cd crates/schemars
git pull origin main  # If keeping .git
# OR
rm -rf crates/schemars
git clone --depth 1 https://github.com/GREsau/schemars.git crates/schemars
rm -rf crates/schemars/.git
```

## Integration

Used by:
- `crates/rsllm` - For automatic JSON Schema generation from Rust types
- `crates/rsllm-macros` - For #[tool] macro schema generation

## Benefits for RSLLM

- Automatic JSON Schema generation from `#[derive(JsonSchema)]`
- Doc comments become schema descriptions
- Validation attributes (#[schemars(range(...)])
- Enum support
- Optional field handling
- Default values integration
