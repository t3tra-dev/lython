	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #64
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	bl	_GC_init
	bl	_PyObject_InitSystem
Lloh0:
	adrp	x0, l_.str.0@PAGE
Lloh1:
	add	x0, x0, l_.str.0@PAGEOFF
	bl	_PyUnicode_FromString
	bl	_print
Lloh2:
	adrp	x21, _Py_None@GOTPAGE
Lloh3:
	ldr	x21, [x21, _Py_None@GOTPAGEOFF]
	ldr	x19, [x21]
	mov	x0, x19
	bl	_Py_INCREF
	mov	x0, x19
	bl	_Py_DECREF
	mov	w0, #6                          ; =0x6
	bl	_PyList_New
	mov	x19, x0
	mov	w0, wzr
	bl	_PyInt_FromI32
	mov	x20, x0
	bl	_Py_INCREF
	mov	x0, x19
	mov	x1, xzr
	mov	x2, x20
	bl	_PyList_SetItem
	mov	w0, #1                          ; =0x1
	bl	_PyInt_FromI32
	mov	x20, x0
	bl	_Py_INCREF
	mov	x0, x19
	mov	w1, #1                          ; =0x1
	mov	x2, x20
	bl	_PyList_SetItem
	mov	w0, #2                          ; =0x2
	bl	_PyInt_FromI32
	mov	x20, x0
	bl	_Py_INCREF
	mov	x0, x19
	mov	w1, #2                          ; =0x2
	mov	x2, x20
	bl	_PyList_SetItem
Lloh4:
	adrp	x0, l_.str.1@PAGE
Lloh5:
	add	x0, x0, l_.str.1@PAGEOFF
	bl	_PyUnicode_FromString
	mov	x20, x0
	bl	_Py_INCREF
	mov	x0, x19
	mov	w1, #3                          ; =0x3
	mov	x2, x20
	bl	_PyList_SetItem
Lloh6:
	adrp	x0, l_.str.2@PAGE
Lloh7:
	add	x0, x0, l_.str.2@PAGEOFF
	bl	_PyUnicode_FromString
	mov	x20, x0
	bl	_Py_INCREF
	mov	x0, x19
	mov	w1, #4                          ; =0x4
	mov	x2, x20
	bl	_PyList_SetItem
Lloh8:
	adrp	x0, l_.str.3@PAGE
Lloh9:
	add	x0, x0, l_.str.3@PAGEOFF
	bl	_PyUnicode_FromString
	mov	x20, x0
	bl	_Py_INCREF
	mov	x0, x19
	mov	w1, #5                          ; =0x5
	mov	x2, x20
	bl	_PyList_SetItem
	mov	x0, x19
	str	x19, [sp, #8]
	bl	_Py_INCREF
	bl	_PyDict_New
	mov	x19, x0
Lloh10:
	adrp	x0, l_.str.4@PAGE
Lloh11:
	add	x0, x0, l_.str.4@PAGEOFF
	bl	_PyUnicode_FromString
	mov	x20, x0
	mov	w0, #1                          ; =0x1
	bl	_PyInt_FromI32
	mov	x2, x0
	mov	x0, x19
	mov	x1, x20
	bl	_PyDict_SetItem
Lloh12:
	adrp	x0, l_.str.5@PAGE
Lloh13:
	add	x0, x0, l_.str.5@PAGEOFF
	bl	_PyUnicode_FromString
	mov	x20, x0
	mov	w0, #2                          ; =0x2
	bl	_PyInt_FromI32
	mov	x2, x0
	mov	x0, x19
	mov	x1, x20
	bl	_PyDict_SetItem
Lloh14:
	adrp	x0, l_.str.6@PAGE
Lloh15:
	add	x0, x0, l_.str.6@PAGEOFF
	bl	_PyUnicode_FromString
	mov	x20, x0
	mov	w0, #3                          ; =0x3
	bl	_PyInt_FromI32
	mov	x2, x0
	mov	x0, x19
	mov	x1, x20
	bl	_PyDict_SetItem
	mov	x0, x19
	str	x19, [sp]
	bl	_Py_INCREF
	add	x0, sp, #8
	bl	_Py_INCREF
	add	x0, sp, #8
	mov	w1, #3                          ; =0x3
	bl	_PyList_GetItem
	mov	x19, x0
	bl	_Py_INCREF
	mov	x0, x19
	bl	_PyObject_Str
	bl	_print
	ldr	x19, [x21]
	mov	x0, x19
	bl	_Py_INCREF
	mov	x0, x19
	bl	_Py_DECREF
	mov	x0, sp
	bl	_Py_INCREF
Lloh16:
	adrp	x0, l_.str.7@PAGE
Lloh17:
	add	x0, x0, l_.str.7@PAGEOFF
	bl	_PyUnicode_FromString
	mov	x19, x0
	mov	x0, sp
	mov	x1, x19
	bl	_PyDict_GetItem
	mov	x20, x0
	mov	x0, x19
	bl	_Py_DECREF
	mov	x0, x20
	bl	_Py_INCREF
	mov	x0, x20
	bl	_PyObject_Str
	bl	_print
	ldr	x19, [x21]
	mov	x0, x19
	bl	_Py_INCREF
	mov	x0, x19
	bl	_Py_DECREF
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	mov	w0, wzr
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh16, Lloh17
	.loh AdrpAdd	Lloh14, Lloh15
	.loh AdrpAdd	Lloh12, Lloh13
	.loh AdrpAdd	Lloh10, Lloh11
	.loh AdrpAdd	Lloh8, Lloh9
	.loh AdrpAdd	Lloh6, Lloh7
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpLdrGot	Lloh2, Lloh3
	.loh AdrpAdd	Lloh0, Lloh1
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str.0:                               ; @.str.0
	.asciz	"Hello, world!"

l_.str.1:                               ; @.str.1
	.asciz	"a"

l_.str.2:                               ; @.str.2
	.asciz	"b"

l_.str.3:                               ; @.str.3
	.asciz	"c"

l_.str.4:                               ; @.str.4
	.asciz	"key1"

l_.str.5:                               ; @.str.5
	.asciz	"key2"

l_.str.6:                               ; @.str.6
	.asciz	"key3"

l_.str.7:                               ; @.str.7
	.asciz	"key1"

.subsections_via_symbols
