{-# OPTIONS --safe #-}

-- The build target. `make` typechecks this and therefore everything below it.
--
-- A module that is not reachable from here is not checked by `make`, and a
-- proof nobody checks is a comment. This file is the analogue of the compiler
-- suite's layer manifest: membership is the thing that makes a check run at all.

module Everything where

-- Shared vocabulary.
import Proof.Prelude

-- The memory model: identity, storage, descriptors.
import Proof.Memory.Fault
import Proof.Memory.Byte
import Proof.Memory.Element
import Proof.Memory.Index
import Proof.Memory.Heap
import Proof.Memory.Descriptor
import Proof.Memory.Resolve
import Proof.Memory.Ops
import Proof.Memory.Properties

-- The instantiation, and a trace that exercises it. Both are load-bearing: the
-- modules above are parameterised over an element signature, and every theorem
-- in them holds vacuously for a signature that is never supplied.
import Proof.Memory.Lython
import Proof.Memory.Trace
