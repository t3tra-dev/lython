; ModuleID = '<swift-imported-modules>'
source_filename = "<swift-imported-modules>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx15.0.0"

%swift.full_existential_type = type { ptr, %swift.type }
%swift.type = type { i64 }
%objc_class = type { ptr, ptr, ptr, ptr, ptr }
%swift.opaque = type opaque
%swift.method_descriptor = type { i32, i32 }
%swift.type_metadata_record = type { i32 }
%swift.metadata_response = type { ptr, i64 }
%Any = type { [24 x i8], ptr }
%TSS = type <{ %Ts11_StringGutsV }>
%Ts11_StringGutsV = type <{ %Ts13_StringObjectV }>
%Ts13_StringObjectV = type <{ %Ts6UInt64V, ptr }>
%Ts6UInt64V = type <{ i64 }>
%TSi = type <{ i64 }>
%TSa = type <{ %Ts12_ArrayBufferV }>
%Ts12_ArrayBufferV = type <{ %Ts14_BridgeStorageV }>
%Ts14_BridgeStorageV = type <{ ptr }>
%T4main6PersonC = type <{ %swift.refcounted, %TSS }>
%swift.refcounted = type { ptr, i64 }
%"$s4main6PersonC4nameSSvM.Frame" = type { [24 x i8] }

@"$s4main7person1AA6PersonCvp" = hidden global ptr null, align 8
@"$sypN" = external global %swift.full_existential_type
@"$sSSN" = external global %swift.type, align 8
@"$sSiN" = external global %swift.type, align 8
@.str.5.Alice = private unnamed_addr constant [6 x i8] c"Alice\00"
@"\01l_entry_point" = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @main to i64), i64 ptrtoint (ptr @"\01l_entry_point" to i64)) to i32), i32 0 }, section "__TEXT, __swift5_entry, regular, no_dead_strip", align 4
@"$s4main6PersonC4nameSSvpWvd" = hidden constant i64 16, align 8
@"$sBoWV" = external global ptr, align 8
@"$s4main6PersonCMm" = hidden global %objc_class { ptr @"OBJC_METACLASS_$__TtCs12_SwiftObject", ptr @"OBJC_METACLASS_$__TtCs12_SwiftObject", ptr @_objc_empty_cache, ptr null, ptr @_METACLASS_DATA__TtC4main6Person }, align 8
@"OBJC_CLASS_$__TtCs12_SwiftObject" = external global %objc_class, align 8
@_objc_empty_cache = external global %swift.opaque
@"OBJC_METACLASS_$__TtCs12_SwiftObject" = external global %objc_class, align 8
@.str.16._TtC4main6Person = private unnamed_addr constant [17 x i8] c"_TtC4main6Person\00"
@_METACLASS_DATA__TtC4main6Person = internal constant { i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { i32 129, i32 40, i32 40, i32 0, ptr null, ptr @.str.16._TtC4main6Person, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@.str.4.name = private unnamed_addr constant [5 x i8] c"name\00"
@.str.0. = private unnamed_addr constant [1 x i8] zeroinitializer
@_IVARS__TtC4main6Person = internal constant { i32, i32, [1 x { ptr, ptr, ptr, i32, i32 }] } { i32 32, i32 1, [1 x { ptr, ptr, ptr, i32, i32 }] [{ ptr, ptr, ptr, i32, i32 } { ptr @"$s4main6PersonC4nameSSvpWvd", ptr @.str.4.name, ptr @.str.0., i32 3, i32 16 }] }, section "__DATA, __objc_const", align 8
@_DATA__TtC4main6Person = internal constant { i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { i32 128, i32 16, i32 32, i32 0, ptr null, ptr @.str.16._TtC4main6Person, ptr null, ptr null, ptr @_IVARS__TtC4main6Person, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@.str.4.main = private constant [5 x i8] c"main\00"
@"$s4mainMXM" = linkonce_odr hidden constant <{ i32, i32, i32 }> <{ i32 0, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.4.main to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32 }>, ptr @"$s4mainMXM", i32 0, i32 2) to i64)) to i32) }>, section "__TEXT,__constg_swiftt", align 4
@.str.6.Person = private constant [7 x i8] c"Person\00"
@"$s4main6PersonCMn" = hidden constant <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }> <{ i32 -2147483568, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4mainMXM" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 1) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.6.Person to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonCMa" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonCMF" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 4) to i64)) to i32), i32 0, i32 3, i32 17, i32 7, i32 1, i32 10, i32 11, i32 6, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonC4nameSSvg" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 13, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonC4nameSSvs" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 14, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonC4nameSSvM" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 15, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonC3ageSivg" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 16, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 16, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonC9printNameyyF" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 17, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 1, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonCACycfC" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 18, i32 1) to i64)) to i32) } }>, section "__TEXT,__constg_swiftt", align 4
@"$s4main6PersonCMf" = internal global <{ ptr, ptr, ptr, i64, ptr, ptr, ptr, ptr, i32, i32, i32, i16, i16, i32, i32, ptr, ptr, i64, ptr, ptr, ptr, ptr, ptr, ptr }> <{ ptr null, ptr @"$s4main6PersonCfD", ptr @"$sBoWV", i64 ptrtoint (ptr @"$s4main6PersonCMm" to i64), ptr @"OBJC_CLASS_$__TtCs12_SwiftObject", ptr @_objc_empty_cache, ptr null, ptr getelementptr (i8, ptr @_DATA__TtC4main6Person, i64 2), i32 2, i32 0, i32 32, i16 7, i16 0, i32 160, i32 24, ptr @"$s4main6PersonCMn", ptr null, i64 16, ptr @"$s4main6PersonC4nameSSvg", ptr @"$s4main6PersonC4nameSSvs", ptr @"$s4main6PersonC4nameSSvM", ptr @"$s4main6PersonC3ageSivg", ptr @"$s4main6PersonC9printNameyyF", ptr @"$s4main6PersonCACycfC" }>, align 8
@"symbolic _____ 4main6PersonC" = linkonce_odr hidden constant <{ i8, i32, i8 }> <{ i8 1, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonCMn" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i8, i32, i8 }>, ptr @"symbolic _____ 4main6PersonC", i32 0, i32 1) to i64)) to i32), i8 0 }>, section "__TEXT,__swift5_typeref, regular", no_sanitize_address, align 2
@"symbolic SS" = linkonce_odr hidden constant <{ [2 x i8], i8 }> <{ [2 x i8] c"SS", i8 0 }>, section "__TEXT,__swift5_typeref, regular", no_sanitize_address, align 2
@0 = private constant [5 x i8] c"name\00", section "__TEXT,__swift5_reflstr, regular", no_sanitize_address
@"$s4main6PersonCMF" = internal constant { i32, i32, i16, i16, i32, i32, i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic _____ 4main6PersonC" to i64), i64 ptrtoint (ptr @"$s4main6PersonCMF" to i64)) to i32), i32 0, i16 1, i16 12, i32 1, i32 2, i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic SS" to i64), i64 ptrtoint (ptr getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32 }, ptr @"$s4main6PersonCMF", i32 0, i32 6) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @0 to i64), i64 ptrtoint (ptr getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32 }, ptr @"$s4main6PersonCMF", i32 0, i32 7) to i64)) to i32) }, section "__TEXT,__swift5_fieldmd, regular", no_sanitize_address, align 4
@".str.1.\0A" = private unnamed_addr constant [2 x i8] c"\0A\00"
@".str.1. " = private unnamed_addr constant [2 x i8] c" \00"
@"$s4main6PersonCHn" = private constant %swift.type_metadata_record { i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main6PersonCMn" to i64), i64 ptrtoint (ptr @"$s4main6PersonCHn" to i64)) to i32) }, section "__TEXT, __swift5_types, regular", no_sanitize_address, align 4
@__swift_reflection_version = linkonce_odr hidden constant i16 3
@"objc_classes_$s4main6PersonCN" = internal global ptr @"$s4main6PersonCN", section "__DATA,__objc_classlist,regular,no_dead_strip", no_sanitize_address, align 8
@llvm.used = appending global [11 x ptr] [ptr @"$s4main7person1AA6PersonCvp", ptr @main, ptr @"$s4main6PersonC4nameSSvg", ptr @"$s4main6PersonC4nameSSvs", ptr @"$s4main6PersonC3ageSivg", ptr @"$s4main6PersonC9printNameyyF", ptr @"\01l_entry_point", ptr @"$s4main6PersonCMF", ptr @"$s4main6PersonCHn", ptr @__swift_reflection_version, ptr @"objc_classes_$s4main6PersonCN"], section "llvm.metadata"
@llvm.compiler.used = appending global [8 x ptr] [ptr @"$s4main6PersonC4nameSSvgTq", ptr @"$s4main6PersonC4nameSSvsTq", ptr @"$s4main6PersonC4nameSSvMTq", ptr @"$s4main6PersonC3ageSivgTq", ptr @"$s4main6PersonC9printNameyyFTq", ptr @"$s4main6PersonCACycfCTq", ptr @"$s4main6PersonCMf", ptr @"$s4main6PersonCN"], section "llvm.metadata"

@"$s4main6PersonC4nameSSvgTq" = hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 13)
@"$s4main6PersonC4nameSSvsTq" = hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 14)
@"$s4main6PersonC4nameSSvMTq" = hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 15)
@"$s4main6PersonC3ageSivgTq" = hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 16)
@"$s4main6PersonC9printNameyyFTq" = hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 17)
@"$s4main6PersonCACycfCTq" = hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor }>, ptr @"$s4main6PersonCMn", i32 0, i32 18)
@"$s4main6PersonCN" = hidden alias %swift.type, getelementptr inbounds (<{ ptr, ptr, ptr, i64, ptr, ptr, ptr, ptr, i32, i32, i32, i16, i16, i32, i32, ptr, ptr, i64, ptr, ptr, ptr, ptr, ptr, ptr }>, ptr @"$s4main6PersonCMf", i32 0, i32 3)

define i32 @main(i32 %0, ptr %1) #0 {
entry:
  %2 = call swiftcc %swift.metadata_response @"$s4main6PersonCMa"(i64 0) #7
  %3 = extractvalue %swift.metadata_response %2, 0
  %4 = call swiftcc ptr @"$s4main6PersonCACycfC"(ptr swiftself %3)
  store ptr %4, ptr @"$s4main7person1AA6PersonCvp", align 8
  %5 = call swiftcc { ptr, ptr } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64 1, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %6 = extractvalue { ptr, ptr } %5, 0
  %7 = extractvalue { ptr, ptr } %5, 1
  %8 = load ptr, ptr @"$s4main7person1AA6PersonCvp", align 8
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds ptr, ptr %9, i64 11
  %11 = load ptr, ptr %10, align 8, !invariant.load !18
  %12 = call swiftcc { i64, ptr } %11(ptr swiftself %8)
  %13 = extractvalue { i64, ptr } %12, 0
  %14 = extractvalue { i64, ptr } %12, 1
  %15 = getelementptr inbounds %Any, ptr %7, i32 0, i32 1
  store ptr @"$sSSN", ptr %15, align 8
  %16 = getelementptr inbounds %Any, ptr %7, i32 0, i32 0
  %17 = getelementptr inbounds %Any, ptr %7, i32 0, i32 0
  %._guts = getelementptr inbounds %TSS, ptr %17, i32 0, i32 0
  %._guts._object = getelementptr inbounds %Ts11_StringGutsV, ptr %._guts, i32 0, i32 0
  %._guts._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 0
  %._guts._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, ptr %._guts._object._countAndFlagsBits, i32 0, i32 0
  store i64 %13, ptr %._guts._object._countAndFlagsBits._value, align 8
  %._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 1
  store ptr %14, ptr %._guts._object._object, align 8
  %18 = call swiftcc ptr @"$ss27_finalizeUninitializedArrayySayxGABnlF"(ptr %6, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %19 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"()
  %20 = extractvalue { i64, ptr } %19, 0
  %21 = extractvalue { i64, ptr } %19, 1
  %22 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"()
  %23 = extractvalue { i64, ptr } %22, 0
  %24 = extractvalue { i64, ptr } %22, 1
  call swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(ptr %18, i64 %20, ptr %21, i64 %23, ptr %24)
  call void @swift_bridgeObjectRelease(ptr %24) #2
  call void @swift_bridgeObjectRelease(ptr %21) #2
  call void @swift_bridgeObjectRelease(ptr %18) #2
  %25 = call swiftcc { ptr, ptr } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64 1, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %26 = extractvalue { ptr, ptr } %25, 0
  %27 = extractvalue { ptr, ptr } %25, 1
  %28 = load ptr, ptr @"$s4main7person1AA6PersonCvp", align 8
  %29 = load ptr, ptr %28, align 8
  %30 = getelementptr inbounds ptr, ptr %29, i64 14
  %31 = load ptr, ptr %30, align 8, !invariant.load !18
  %32 = call swiftcc i64 %31(ptr swiftself %28)
  %33 = getelementptr inbounds %Any, ptr %27, i32 0, i32 1
  store ptr @"$sSiN", ptr %33, align 8
  %34 = getelementptr inbounds %Any, ptr %27, i32 0, i32 0
  %35 = getelementptr inbounds %Any, ptr %27, i32 0, i32 0
  %._value = getelementptr inbounds %TSi, ptr %35, i32 0, i32 0
  store i64 %32, ptr %._value, align 8
  %36 = call swiftcc ptr @"$ss27_finalizeUninitializedArrayySayxGABnlF"(ptr %26, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %37 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"()
  %38 = extractvalue { i64, ptr } %37, 0
  %39 = extractvalue { i64, ptr } %37, 1
  %40 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"()
  %41 = extractvalue { i64, ptr } %40, 0
  %42 = extractvalue { i64, ptr } %40, 1
  call swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(ptr %36, i64 %38, ptr %39, i64 %41, ptr %42)
  call void @swift_bridgeObjectRelease(ptr %42) #2
  call void @swift_bridgeObjectRelease(ptr %39) #2
  call void @swift_bridgeObjectRelease(ptr %36) #2
  %43 = load ptr, ptr @"$s4main7person1AA6PersonCvp", align 8
  %44 = load ptr, ptr %43, align 8
  %45 = getelementptr inbounds ptr, ptr %44, i64 15
  %46 = load ptr, ptr %45, align 8, !invariant.load !18
  call swiftcc void %46(ptr swiftself %43)
  ret i32 0
}

; Function Attrs: noinline nounwind memory(none)
define hidden swiftcc %swift.metadata_response @"$s4main6PersonCMa"(i64 %0) #1 {
entry:
  %1 = call ptr @objc_opt_self(ptr getelementptr inbounds (<{ ptr, ptr, ptr, i64, ptr, ptr, ptr, ptr, i32, i32, i32, i16, i16, i32, i32, ptr, ptr, i64, ptr, ptr, ptr, ptr, ptr, ptr }>, ptr @"$s4main6PersonCMf", i32 0, i32 3)) #2
  %2 = insertvalue %swift.metadata_response undef, ptr %1, 0
  %3 = insertvalue %swift.metadata_response %2, i64 0, 1
  ret %swift.metadata_response %3
}

define hidden swiftcc ptr @"$s4main6PersonCACycfC"(ptr swiftself %0) #0 {
entry:
  %1 = call noalias ptr @swift_allocObject(ptr %0, i64 32, i64 7) #2
  %2 = call swiftcc ptr @"$s4main6PersonCACycfc"(ptr swiftself %1)
  ret ptr %2
}

declare swiftcc { ptr, ptr } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64, ptr) #0

define linkonce_odr hidden swiftcc ptr @"$ss27_finalizeUninitializedArrayySayxGABnlF"(ptr %0, ptr %Element) #0 {
entry:
  %Element1 = alloca ptr, align 8
  %1 = alloca %TSa, align 8
  store ptr %Element, ptr %Element1, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr %1)
  %._buffer = getelementptr inbounds %TSa, ptr %1, i32 0, i32 0
  %._buffer._storage = getelementptr inbounds %Ts12_ArrayBufferV, ptr %._buffer, i32 0, i32 0
  %._buffer._storage.rawValue = getelementptr inbounds %Ts14_BridgeStorageV, ptr %._buffer._storage, i32 0, i32 0
  store ptr %0, ptr %._buffer._storage.rawValue, align 8
  %2 = call swiftcc %swift.metadata_response @"$sSaMa"(i64 0, ptr %Element) #7
  %3 = extractvalue %swift.metadata_response %2, 0
  call swiftcc void @"$sSa12_endMutationyyF"(ptr %3, ptr nocapture swiftself dereferenceable(8) %1)
  %._buffer2 = getelementptr inbounds %TSa, ptr %1, i32 0, i32 0
  %._buffer2._storage = getelementptr inbounds %Ts12_ArrayBufferV, ptr %._buffer2, i32 0, i32 0
  %._buffer2._storage.rawValue = getelementptr inbounds %Ts14_BridgeStorageV, ptr %._buffer2._storage, i32 0, i32 0
  %4 = load ptr, ptr %._buffer2._storage.rawValue, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr %1)
  ret ptr %4
}

define linkonce_odr hidden swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"() #0 {
entry:
  %0 = call swiftcc { i64, ptr } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(ptr @".str.1. ", i64 1, i1 true)
  %1 = extractvalue { i64, ptr } %0, 0
  %2 = extractvalue { i64, ptr } %0, 1
  %3 = insertvalue { i64, ptr } undef, i64 %1, 0
  %4 = insertvalue { i64, ptr } %3, ptr %2, 1
  ret { i64, ptr } %4
}

define linkonce_odr hidden swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"() #0 {
entry:
  %0 = call swiftcc { i64, ptr } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(ptr @".str.1.\0A", i64 1, i1 true)
  %1 = extractvalue { i64, ptr } %0, 0
  %2 = extractvalue { i64, ptr } %0, 1
  %3 = insertvalue { i64, ptr } undef, i64 %1, 0
  %4 = insertvalue { i64, ptr } %3, ptr %2, 1
  ret { i64, ptr } %4
}

declare swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(ptr, i64, ptr, i64, ptr) #0

; Function Attrs: nounwind
declare void @swift_bridgeObjectRelease(ptr) #2

; Function Attrs: nounwind
declare ptr @swift_allocObject(ptr, i64, i64) #2

define hidden swiftcc { i64, ptr } @"$s4main6PersonC4nameSSvpfi"() #0 {
entry:
  %0 = call swiftcc { i64, ptr } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(ptr @.str.5.Alice, i64 5, i1 true)
  %1 = extractvalue { i64, ptr } %0, 0
  %2 = extractvalue { i64, ptr } %0, 1
  %3 = insertvalue { i64, ptr } undef, i64 %1, 0
  %4 = insertvalue { i64, ptr } %3, ptr %2, 1
  ret { i64, ptr } %4
}

define hidden swiftcc { i64, ptr } @"$s4main6PersonC4nameSSvg"(ptr swiftself %0) #0 {
entry:
  %access-scratch = alloca [24 x i8], align 8
  %1 = getelementptr inbounds %T4main6PersonC, ptr %0, i32 0, i32 1
  call void @llvm.lifetime.start.p0(i64 -1, ptr %access-scratch)
  call void @swift_beginAccess(ptr %1, ptr %access-scratch, i64 32, ptr null) #2
  %._guts = getelementptr inbounds %TSS, ptr %1, i32 0, i32 0
  %._guts._object = getelementptr inbounds %Ts11_StringGutsV, ptr %._guts, i32 0, i32 0
  %._guts._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 0
  %._guts._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, ptr %._guts._object._countAndFlagsBits, i32 0, i32 0
  %2 = load i64, ptr %._guts._object._countAndFlagsBits._value, align 8
  %._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 1
  %3 = load ptr, ptr %._guts._object._object, align 8
  %4 = call ptr @swift_bridgeObjectRetain(ptr returned %3) #2
  call void @swift_endAccess(ptr %access-scratch) #2
  call void @llvm.lifetime.end.p0(i64 -1, ptr %access-scratch)
  %5 = insertvalue { i64, ptr } undef, i64 %2, 0
  %6 = insertvalue { i64, ptr } %5, ptr %3, 1
  ret { i64, ptr } %6
}

define hidden swiftcc void @"$s4main6PersonC4nameSSvs"(i64 %0, ptr %1, ptr swiftself %2) #0 {
entry:
  %access-scratch = alloca [24 x i8], align 8
  %3 = call ptr @swift_bridgeObjectRetain(ptr returned %1) #2
  %4 = getelementptr inbounds %T4main6PersonC, ptr %2, i32 0, i32 1
  call void @llvm.lifetime.start.p0(i64 -1, ptr %access-scratch)
  call void @swift_beginAccess(ptr %4, ptr %access-scratch, i64 33, ptr null) #2
  %._guts = getelementptr inbounds %TSS, ptr %4, i32 0, i32 0
  %._guts._object = getelementptr inbounds %Ts11_StringGutsV, ptr %._guts, i32 0, i32 0
  %._guts._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 0
  %._guts._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, ptr %._guts._object._countAndFlagsBits, i32 0, i32 0
  %5 = load i64, ptr %._guts._object._countAndFlagsBits._value, align 8
  %._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 1
  %6 = load ptr, ptr %._guts._object._object, align 8
  %._guts1 = getelementptr inbounds %TSS, ptr %4, i32 0, i32 0
  %._guts1._object = getelementptr inbounds %Ts11_StringGutsV, ptr %._guts1, i32 0, i32 0
  %._guts1._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts1._object, i32 0, i32 0
  %._guts1._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, ptr %._guts1._object._countAndFlagsBits, i32 0, i32 0
  store i64 %0, ptr %._guts1._object._countAndFlagsBits._value, align 8
  %._guts1._object._object = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts1._object, i32 0, i32 1
  store ptr %1, ptr %._guts1._object._object, align 8
  call void @swift_bridgeObjectRelease(ptr %6) #2
  call void @swift_endAccess(ptr %access-scratch) #2
  call void @llvm.lifetime.end.p0(i64 -1, ptr %access-scratch)
  call void @swift_bridgeObjectRelease(ptr %1) #2
  ret void
}

; Function Attrs: noinline
define hidden swiftcc { ptr, ptr } @"$s4main6PersonC4nameSSvM"(ptr noalias dereferenceable(32) %0, ptr swiftself %1) #3 {
entry:
  %access-scratch = getelementptr inbounds %"$s4main6PersonC4nameSSvM.Frame", ptr %0, i32 0, i32 0
  %2 = getelementptr inbounds %T4main6PersonC, ptr %1, i32 0, i32 1
  call void @llvm.lifetime.start.p0(i64 -1, ptr %access-scratch)
  call void @swift_beginAccess(ptr %2, ptr %access-scratch, i64 33, ptr null) #2
  %3 = insertvalue { ptr, ptr } poison, ptr @"$s4main6PersonC4nameSSvM.resume.0", 0
  %4 = insertvalue { ptr, ptr } %3, ptr %2, 1
  ret { ptr, ptr } %4
}

define internal swiftcc void @"$s4main6PersonC4nameSSvM.resume.0"(ptr noalias noundef nonnull align 8 dereferenceable(32) %0, i1 %1) #0 {
entryresume.0:
  %access-scratch = getelementptr inbounds %"$s4main6PersonC4nameSSvM.Frame", ptr %0, i32 0, i32 0
  br i1 %1, label %3, label %2

2:                                                ; preds = %entryresume.0
  call void @swift_endAccess(ptr %access-scratch) #2
  call void @llvm.lifetime.end.p0(i64 -1, ptr %access-scratch)
  br label %CoroEnd

3:                                                ; preds = %entryresume.0
  call void @swift_endAccess(ptr %access-scratch) #2
  call void @llvm.lifetime.end.p0(i64 -1, ptr %access-scratch)
  br label %CoroEnd

CoroEnd:                                          ; preds = %2, %3
  ret void
}

define hidden swiftcc i64 @"$s4main6PersonC3ageSivg"(ptr swiftself %0) #0 {
entry:
  %self.debug = alloca ptr, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %self.debug, i8 0, i64 8, i1 false)
  store ptr %0, ptr %self.debug, align 8
  ret i64 10
}

define hidden swiftcc void @"$s4main6PersonC9printNameyyF"(ptr swiftself %0) #0 {
entry:
  %self.debug = alloca ptr, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %self.debug, i8 0, i64 8, i1 false)
  store ptr %0, ptr %self.debug, align 8
  %1 = call swiftcc { ptr, ptr } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64 1, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %2 = extractvalue { ptr, ptr } %1, 0
  %3 = extractvalue { ptr, ptr } %1, 1
  %4 = load ptr, ptr %0, align 8
  %5 = getelementptr inbounds ptr, ptr %4, i64 11
  %6 = load ptr, ptr %5, align 8, !invariant.load !18
  %7 = call swiftcc { i64, ptr } %6(ptr swiftself %0)
  %8 = extractvalue { i64, ptr } %7, 0
  %9 = extractvalue { i64, ptr } %7, 1
  %10 = getelementptr inbounds %Any, ptr %3, i32 0, i32 1
  store ptr @"$sSSN", ptr %10, align 8
  %11 = getelementptr inbounds %Any, ptr %3, i32 0, i32 0
  %12 = getelementptr inbounds %Any, ptr %3, i32 0, i32 0
  %._guts = getelementptr inbounds %TSS, ptr %12, i32 0, i32 0
  %._guts._object = getelementptr inbounds %Ts11_StringGutsV, ptr %._guts, i32 0, i32 0
  %._guts._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 0
  %._guts._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, ptr %._guts._object._countAndFlagsBits, i32 0, i32 0
  store i64 %8, ptr %._guts._object._countAndFlagsBits._value, align 8
  %._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 1
  store ptr %9, ptr %._guts._object._object, align 8
  %13 = call swiftcc ptr @"$ss27_finalizeUninitializedArrayySayxGABnlF"(ptr %2, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %14 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"()
  %15 = extractvalue { i64, ptr } %14, 0
  %16 = extractvalue { i64, ptr } %14, 1
  %17 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"()
  %18 = extractvalue { i64, ptr } %17, 0
  %19 = extractvalue { i64, ptr } %17, 1
  call swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(ptr %13, i64 %15, ptr %16, i64 %18, ptr %19)
  call void @swift_bridgeObjectRelease(ptr %19) #2
  call void @swift_bridgeObjectRelease(ptr %16) #2
  call void @swift_bridgeObjectRelease(ptr %13) #2
  ret void
}

define hidden swiftcc ptr @"$s4main6PersonCfd"(ptr swiftself %0) #0 {
entry:
  %self.debug = alloca ptr, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %self.debug, i8 0, i64 8, i1 false)
  store ptr %0, ptr %self.debug, align 8
  %1 = getelementptr inbounds %T4main6PersonC, ptr %0, i32 0, i32 1
  %2 = call ptr @"$sSSWOh"(ptr %1)
  ret ptr %0
}

define hidden swiftcc void @"$s4main6PersonCfD"(ptr swiftself %0) #0 {
entry:
  %self.debug = alloca ptr, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %self.debug, i8 0, i64 8, i1 false)
  store ptr %0, ptr %self.debug, align 8
  %1 = call swiftcc ptr @"$s4main6PersonCfd"(ptr swiftself %0)
  call void @swift_deallocClassInstance(ptr %1, i64 32, i64 7) #2
  ret void
}

define hidden swiftcc ptr @"$s4main6PersonCACycfc"(ptr swiftself %0) #0 {
entry:
  %self.debug = alloca ptr, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %self.debug, i8 0, i64 8, i1 false)
  store ptr %0, ptr %self.debug, align 8
  %1 = getelementptr inbounds %T4main6PersonC, ptr %0, i32 0, i32 1
  %2 = call swiftcc { i64, ptr } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(ptr @.str.5.Alice, i64 5, i1 true)
  %3 = extractvalue { i64, ptr } %2, 0
  %4 = extractvalue { i64, ptr } %2, 1
  %._guts = getelementptr inbounds %TSS, ptr %1, i32 0, i32 0
  %._guts._object = getelementptr inbounds %Ts11_StringGutsV, ptr %._guts, i32 0, i32 0
  %._guts._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 0
  %._guts._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, ptr %._guts._object._countAndFlagsBits, i32 0, i32 0
  store i64 %3, ptr %._guts._object._countAndFlagsBits._value, align 8
  %._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 1
  store ptr %4, ptr %._guts._object._object, align 8
  ret ptr %0
}

declare swiftcc { i64, ptr } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(ptr, i64, i1) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #4

; Function Attrs: nounwind
declare void @swift_beginAccess(ptr, ptr, i64, ptr) #2

; Function Attrs: nounwind
declare ptr @swift_bridgeObjectRetain(ptr returned) #2

; Function Attrs: nounwind
declare void @swift_endAccess(ptr) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #5

; Function Attrs: noinline nounwind
define linkonce_odr hidden ptr @"$sSSWOh"(ptr %0) #6 {
entry:
  %._guts = getelementptr inbounds %TSS, ptr %0, i32 0, i32 0
  %._guts._object = getelementptr inbounds %Ts11_StringGutsV, ptr %._guts, i32 0, i32 0
  %._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 1
  %toDestroy = load ptr, ptr %._guts._object._object, align 8
  call void @swift_bridgeObjectRelease(ptr %toDestroy) #2
  ret ptr %0
}

; Function Attrs: nounwind
declare void @swift_deallocClassInstance(ptr, i64, i64) #2

; Function Attrs: nounwind
declare ptr @objc_opt_self(ptr) #2

define linkonce_odr hidden swiftcc void @"$sSa12_endMutationyyF"(ptr %"Array<Element>", ptr nocapture swiftself dereferenceable(8) %0) #0 {
entry:
  %._buffer = getelementptr inbounds %TSa, ptr %0, i32 0, i32 0
  %._buffer._storage = getelementptr inbounds %Ts12_ArrayBufferV, ptr %._buffer, i32 0, i32 0
  %._buffer._storage.rawValue = getelementptr inbounds %Ts14_BridgeStorageV, ptr %._buffer._storage, i32 0, i32 0
  %1 = load ptr, ptr %._buffer._storage.rawValue, align 8
  %._buffer1 = getelementptr inbounds %TSa, ptr %0, i32 0, i32 0
  %._buffer1._storage = getelementptr inbounds %Ts12_ArrayBufferV, ptr %._buffer1, i32 0, i32 0
  %._buffer1._storage.rawValue = getelementptr inbounds %Ts14_BridgeStorageV, ptr %._buffer1._storage, i32 0, i32 0
  store ptr %1, ptr %._buffer1._storage.rawValue, align 8
  ret void
}

declare swiftcc %swift.metadata_response @"$sSaMa"(i64, ptr) #0

attributes #0 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #1 = { noinline nounwind memory(none) "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #2 = { nounwind }
attributes #3 = { noinline "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { noinline nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #7 = { nounwind memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11}
!swift.module.flags = !{!12}
!llvm.linker.options = !{!13, !14, !15, !16, !17}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 15, i32 1]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!4 = !{i32 4, !"Objective-C Garbage Collection", i32 100665088}
!5 = !{i32 1, !"Objective-C Class Properties", i32 64}
!6 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i8 0}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 8, !"PIC Level", i32 2}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 1}
!11 = !{i32 1, !"Swift Version", i32 7}
!12 = !{!"standard-library", i1 false}
!13 = !{!"-lswiftSwiftOnoneSupport"}
!14 = !{!"-lswiftCore"}
!15 = !{!"-lswift_Concurrency"}
!16 = !{!"-lswift_StringProcessing"}
!17 = !{!"-lobjc"}
!18 = !{}
