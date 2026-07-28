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
import Proof.Memory.Properties

-- The memref dialect, transcribed op by op, and realloc.
import Proof.MemRef.Dialect
import Proof.MemRef.Realloc

-- Quantitative type theory's multiplicities, and the reference modes they do
-- not determine.
import Proof.QTT.Quantity

-- Reference counting: owner sites, the ghost count the runtime counter has to
-- implement, py.incref / py.decref, and the invariant tying them together.
import Proof.RC.Object
import Proof.RC.OwnerSite
import Proof.RC.Machine
import Proof.RC.Ops
import Proof.RC.Invariant
import Proof.RC.Properties
import Proof.RC.Trace

-- The instantiation, and a trace that exercises it. Both are load-bearing: the
-- modules above are parameterised over an element signature, and every theorem
-- in them holds vacuously for a signature that is never supplied.
import Proof.Memory.Lython
import Proof.Memory.Trace
