; 文字列定義
@str = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1

; 関数定義 
define i32 @main(i32 %argc, i8** %argv) {
; ラベル
entry:
  ; 関数呼び出し, GetElementPtr
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @str, i64 0, i64 0))
  ; 関数から値を返す
  ret i32 0
}
; 外部関数の宣言
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #1
