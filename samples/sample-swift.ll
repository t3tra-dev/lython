; ModuleID = '<swift-imported-modules>'
source_filename = "<swift-imported-modules>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx15.0.0"

%swift.full_existential_type = type { ptr, %swift.type }
%swift.type = type { i64 }
%Any = type { [24 x i8], ptr }
%TSS = type <{ %Ts11_StringGutsV }>
%Ts11_StringGutsV = type <{ %Ts13_StringObjectV }>
%Ts13_StringObjectV = type <{ %Ts6UInt64V, ptr }>
%Ts6UInt64V = type <{ i64 }>
%TSa = type <{ %Ts12_ArrayBufferV }>
%Ts12_ArrayBufferV = type <{ %Ts14_BridgeStorageV }>
%Ts14_BridgeStorageV = type <{ ptr }>
%swift.metadata_response = type { ptr, i64 }

@"$sypN" = external global %swift.full_existential_type
@".str.13.Hello, world!" = private unnamed_addr constant [14 x i8] c"Hello, world!\00"
@"$sSSN" = external global %swift.type, align 8
@"\01l_entry_point" = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @main to i64), i64 ptrtoint (ptr @"\01l_entry_point" to i64)) to i32), i32 0 }, section "__TEXT, __swift5_entry, regular, no_dead_strip", align 4
@".str.1.\0A" = private unnamed_addr constant [2 x i8] c"\0A\00"
@".str.1. " = private unnamed_addr constant [2 x i8] c" \00"
@__swift_reflection_version = linkonce_odr hidden constant i16 3
@llvm.used = appending global [3 x ptr] [ptr @main, ptr @"\01l_entry_point", ptr @__swift_reflection_version], section "llvm.metadata"

define i32 @main(i32 %0, ptr %1) #0 {
entry:
  %2 = call swiftcc { ptr, ptr } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64 1, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %3 = extractvalue { ptr, ptr } %2, 0
  %4 = extractvalue { ptr, ptr } %2, 1
  %5 = call swiftcc { i64, ptr } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(ptr @".str.13.Hello, world!", i64 13, i1 true)
  %6 = extractvalue { i64, ptr } %5, 0
  %7 = extractvalue { i64, ptr } %5, 1
  %8 = getelementptr inbounds %Any, ptr %4, i32 0, i32 1
  store ptr @"$sSSN", ptr %8, align 8
  %9 = getelementptr inbounds %Any, ptr %4, i32 0, i32 0
  %10 = getelementptr inbounds %Any, ptr %4, i32 0, i32 0
  %._guts = getelementptr inbounds %TSS, ptr %10, i32 0, i32 0
  %._guts._object = getelementptr inbounds %Ts11_StringGutsV, ptr %._guts, i32 0, i32 0
  %._guts._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 0
  %._guts._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, ptr %._guts._object._countAndFlagsBits, i32 0, i32 0
  store i64 %6, ptr %._guts._object._countAndFlagsBits._value, align 8
  %._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, ptr %._guts._object, i32 0, i32 1
  store ptr %7, ptr %._guts._object._object, align 8
  %11 = call swiftcc ptr @"$ss27_finalizeUninitializedArrayySayxGABnlF"(ptr %3, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %12 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"()
  %13 = extractvalue { i64, ptr } %12, 0
  %14 = extractvalue { i64, ptr } %12, 1
  %15 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"()
  %16 = extractvalue { i64, ptr } %15, 0
  %17 = extractvalue { i64, ptr } %15, 1
  call swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(ptr %11, i64 %13, ptr %14, i64 %16, ptr %17)
  call void @swift_bridgeObjectRelease(ptr %17) #1
  call void @swift_bridgeObjectRelease(ptr %14) #1
  call void @swift_bridgeObjectRelease(ptr %11) #1
  ret i32 0
}

declare swiftcc { ptr, ptr } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64, ptr) #0

declare swiftcc { i64, ptr } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(ptr, i64, i1) #0

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
  %2 = call swiftcc %swift.metadata_response @"$sSaMa"(i64 0, ptr %Element) #3
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
declare void @swift_bridgeObjectRelease(ptr) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

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

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

attributes #0 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #1 = { nounwind }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nounwind memory(none) }

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
