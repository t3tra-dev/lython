{-# OPTIONS --safe #-}

-- Backing storage is BYTES, not typed cells.
--
-- The reason is memref.view: it builds a differently-typed memref out of a flat
-- i8 buffer, so no typed-cell heap can express the programs this compiler
-- already emits. Lython's own boxed layout is the same shape -- one
-- memref<?xi8> allocation with the refcount, the class id and the payload at
-- fixed byte offsets inside it.

module Proof.Memory.Byte where

open import Data.Fin using (Fin)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ)

Byte : Set
Byte = Fin 256

-- Uninitialised is a first-class state rather than a distinguished byte value.
-- Without it, reading freshly allocated memory would yield some concrete byte
-- and the model could not tell "read what was written" from "read whatever was
-- there" -- which is the entire content of the uninitialized-read fault.
data StoredByte : Set where
  uninit : StoredByte
  init   : Byte → StoredByte

-- The read side of a stored byte. `nothing` is the uninitialised case, and it
-- is the caller's job to turn that into uninitialized-read rather than to
-- substitute a default -- substituting is how a real compiler ships a program
-- whose output depends on what the allocator last left there.
readByte : StoredByte → Maybe Byte
readByte uninit   = nothing
readByte (init b) = just b

data Initialised : StoredByte → Set where
  is-init : (b : Byte) → Initialised (init b)
