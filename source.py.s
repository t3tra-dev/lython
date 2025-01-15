	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:                                ; %entry
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh0:
	adrp	x0, l_.str.0@PAGE
Lloh1:
	add	x0, x0, l_.str.0@PAGEOFF
	bl	_print
Lloh2:
	adrp	x0, l_.str.1@PAGE
Lloh3:
	add	x0, x0, l_.str.1@PAGEOFF
	bl	_print
	mov	w0, wzr
	ldp	x29, x30, [sp], #16             ; 16-byte Folded Reload
	ret
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpAdd	Lloh0, Lloh1
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str.0:                               ; @.str.0
	.asciz	"Hello, world!"

l_.str.1:                               ; @.str.1
	.asciz	"multibite character: \343\201\202\343\201\204\343\201\206\343\201\210\343\201\212, \360\237\220\215"

	.section	__DATA,__data
	.globl	_Py_True                        ; @Py_True
_Py_True:
	.byte	1                               ; 0x1

	.globl	_Py_False                       ; @Py_False
.zerofill __DATA,__common,_Py_False,1,0
	.globl	_Py_None                        ; @Py_None
.zerofill __DATA,__common,_Py_None,8,3
.subsections_via_symbols
