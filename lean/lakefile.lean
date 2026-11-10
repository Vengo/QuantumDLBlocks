import Lake
open Lake DSL

package «QuantumDLBlocksLean» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib «QuantumDLBlocksLean» where
  roots := #[`QuantumDLBlocksLean]
