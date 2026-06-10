// CPython-compatible small integer cache for [-5, 256].
// The globals are mutable, not constant, because the refcount kernel uses
// atomic RMW even when the immortal refcount value is preserved.
module {
  memref.global "private" @__ly_small_int_m5_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_m5_meta : memref<2xi64> = dense<[-1, 1]>
  memref.global "private" @__ly_small_int_m5_digits : memref<1xi32> = dense<[5]>
  memref.global "private" @__ly_small_int_m4_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_m4_meta : memref<2xi64> = dense<[-1, 1]>
  memref.global "private" @__ly_small_int_m4_digits : memref<1xi32> = dense<[4]>
  memref.global "private" @__ly_small_int_m3_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_m3_meta : memref<2xi64> = dense<[-1, 1]>
  memref.global "private" @__ly_small_int_m3_digits : memref<1xi32> = dense<[3]>
  memref.global "private" @__ly_small_int_m2_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_m2_meta : memref<2xi64> = dense<[-1, 1]>
  memref.global "private" @__ly_small_int_m2_digits : memref<1xi32> = dense<[2]>
  memref.global "private" @__ly_small_int_m1_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_m1_meta : memref<2xi64> = dense<[-1, 1]>
  memref.global "private" @__ly_small_int_m1_digits : memref<1xi32> = dense<[1]>
  memref.global "private" @__ly_small_int_z_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_z_meta : memref<2xi64> = dense<[0, 0]>
  memref.global "private" @__ly_small_int_z_digits : memref<1xi32> = dense<[0]>
  memref.global "private" @__ly_small_int_1_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_1_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_1_digits : memref<1xi32> = dense<[1]>
  memref.global "private" @__ly_small_int_2_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_2_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_2_digits : memref<1xi32> = dense<[2]>
  memref.global "private" @__ly_small_int_3_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_3_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_3_digits : memref<1xi32> = dense<[3]>
  memref.global "private" @__ly_small_int_4_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_4_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_4_digits : memref<1xi32> = dense<[4]>
  memref.global "private" @__ly_small_int_5_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_5_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_5_digits : memref<1xi32> = dense<[5]>
  memref.global "private" @__ly_small_int_6_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_6_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_6_digits : memref<1xi32> = dense<[6]>
  memref.global "private" @__ly_small_int_7_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_7_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_7_digits : memref<1xi32> = dense<[7]>
  memref.global "private" @__ly_small_int_8_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_8_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_8_digits : memref<1xi32> = dense<[8]>
  memref.global "private" @__ly_small_int_9_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_9_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_9_digits : memref<1xi32> = dense<[9]>
  memref.global "private" @__ly_small_int_10_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_10_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_10_digits : memref<1xi32> = dense<[10]>
  memref.global "private" @__ly_small_int_11_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_11_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_11_digits : memref<1xi32> = dense<[11]>
  memref.global "private" @__ly_small_int_12_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_12_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_12_digits : memref<1xi32> = dense<[12]>
  memref.global "private" @__ly_small_int_13_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_13_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_13_digits : memref<1xi32> = dense<[13]>
  memref.global "private" @__ly_small_int_14_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_14_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_14_digits : memref<1xi32> = dense<[14]>
  memref.global "private" @__ly_small_int_15_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_15_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_15_digits : memref<1xi32> = dense<[15]>
  memref.global "private" @__ly_small_int_16_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_16_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_16_digits : memref<1xi32> = dense<[16]>
  memref.global "private" @__ly_small_int_17_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_17_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_17_digits : memref<1xi32> = dense<[17]>
  memref.global "private" @__ly_small_int_18_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_18_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_18_digits : memref<1xi32> = dense<[18]>
  memref.global "private" @__ly_small_int_19_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_19_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_19_digits : memref<1xi32> = dense<[19]>
  memref.global "private" @__ly_small_int_20_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_20_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_20_digits : memref<1xi32> = dense<[20]>
  memref.global "private" @__ly_small_int_21_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_21_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_21_digits : memref<1xi32> = dense<[21]>
  memref.global "private" @__ly_small_int_22_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_22_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_22_digits : memref<1xi32> = dense<[22]>
  memref.global "private" @__ly_small_int_23_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_23_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_23_digits : memref<1xi32> = dense<[23]>
  memref.global "private" @__ly_small_int_24_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_24_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_24_digits : memref<1xi32> = dense<[24]>
  memref.global "private" @__ly_small_int_25_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_25_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_25_digits : memref<1xi32> = dense<[25]>
  memref.global "private" @__ly_small_int_26_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_26_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_26_digits : memref<1xi32> = dense<[26]>
  memref.global "private" @__ly_small_int_27_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_27_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_27_digits : memref<1xi32> = dense<[27]>
  memref.global "private" @__ly_small_int_28_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_28_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_28_digits : memref<1xi32> = dense<[28]>
  memref.global "private" @__ly_small_int_29_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_29_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_29_digits : memref<1xi32> = dense<[29]>
  memref.global "private" @__ly_small_int_30_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_30_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_30_digits : memref<1xi32> = dense<[30]>
  memref.global "private" @__ly_small_int_31_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_31_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_31_digits : memref<1xi32> = dense<[31]>
  memref.global "private" @__ly_small_int_32_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_32_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_32_digits : memref<1xi32> = dense<[32]>
  memref.global "private" @__ly_small_int_33_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_33_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_33_digits : memref<1xi32> = dense<[33]>
  memref.global "private" @__ly_small_int_34_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_34_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_34_digits : memref<1xi32> = dense<[34]>
  memref.global "private" @__ly_small_int_35_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_35_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_35_digits : memref<1xi32> = dense<[35]>
  memref.global "private" @__ly_small_int_36_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_36_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_36_digits : memref<1xi32> = dense<[36]>
  memref.global "private" @__ly_small_int_37_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_37_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_37_digits : memref<1xi32> = dense<[37]>
  memref.global "private" @__ly_small_int_38_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_38_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_38_digits : memref<1xi32> = dense<[38]>
  memref.global "private" @__ly_small_int_39_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_39_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_39_digits : memref<1xi32> = dense<[39]>
  memref.global "private" @__ly_small_int_40_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_40_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_40_digits : memref<1xi32> = dense<[40]>
  memref.global "private" @__ly_small_int_41_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_41_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_41_digits : memref<1xi32> = dense<[41]>
  memref.global "private" @__ly_small_int_42_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_42_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_42_digits : memref<1xi32> = dense<[42]>
  memref.global "private" @__ly_small_int_43_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_43_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_43_digits : memref<1xi32> = dense<[43]>
  memref.global "private" @__ly_small_int_44_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_44_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_44_digits : memref<1xi32> = dense<[44]>
  memref.global "private" @__ly_small_int_45_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_45_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_45_digits : memref<1xi32> = dense<[45]>
  memref.global "private" @__ly_small_int_46_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_46_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_46_digits : memref<1xi32> = dense<[46]>
  memref.global "private" @__ly_small_int_47_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_47_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_47_digits : memref<1xi32> = dense<[47]>
  memref.global "private" @__ly_small_int_48_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_48_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_48_digits : memref<1xi32> = dense<[48]>
  memref.global "private" @__ly_small_int_49_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_49_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_49_digits : memref<1xi32> = dense<[49]>
  memref.global "private" @__ly_small_int_50_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_50_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_50_digits : memref<1xi32> = dense<[50]>
  memref.global "private" @__ly_small_int_51_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_51_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_51_digits : memref<1xi32> = dense<[51]>
  memref.global "private" @__ly_small_int_52_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_52_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_52_digits : memref<1xi32> = dense<[52]>
  memref.global "private" @__ly_small_int_53_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_53_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_53_digits : memref<1xi32> = dense<[53]>
  memref.global "private" @__ly_small_int_54_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_54_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_54_digits : memref<1xi32> = dense<[54]>
  memref.global "private" @__ly_small_int_55_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_55_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_55_digits : memref<1xi32> = dense<[55]>
  memref.global "private" @__ly_small_int_56_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_56_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_56_digits : memref<1xi32> = dense<[56]>
  memref.global "private" @__ly_small_int_57_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_57_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_57_digits : memref<1xi32> = dense<[57]>
  memref.global "private" @__ly_small_int_58_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_58_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_58_digits : memref<1xi32> = dense<[58]>
  memref.global "private" @__ly_small_int_59_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_59_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_59_digits : memref<1xi32> = dense<[59]>
  memref.global "private" @__ly_small_int_60_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_60_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_60_digits : memref<1xi32> = dense<[60]>
  memref.global "private" @__ly_small_int_61_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_61_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_61_digits : memref<1xi32> = dense<[61]>
  memref.global "private" @__ly_small_int_62_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_62_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_62_digits : memref<1xi32> = dense<[62]>
  memref.global "private" @__ly_small_int_63_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_63_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_63_digits : memref<1xi32> = dense<[63]>
  memref.global "private" @__ly_small_int_64_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_64_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_64_digits : memref<1xi32> = dense<[64]>
  memref.global "private" @__ly_small_int_65_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_65_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_65_digits : memref<1xi32> = dense<[65]>
  memref.global "private" @__ly_small_int_66_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_66_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_66_digits : memref<1xi32> = dense<[66]>
  memref.global "private" @__ly_small_int_67_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_67_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_67_digits : memref<1xi32> = dense<[67]>
  memref.global "private" @__ly_small_int_68_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_68_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_68_digits : memref<1xi32> = dense<[68]>
  memref.global "private" @__ly_small_int_69_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_69_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_69_digits : memref<1xi32> = dense<[69]>
  memref.global "private" @__ly_small_int_70_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_70_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_70_digits : memref<1xi32> = dense<[70]>
  memref.global "private" @__ly_small_int_71_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_71_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_71_digits : memref<1xi32> = dense<[71]>
  memref.global "private" @__ly_small_int_72_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_72_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_72_digits : memref<1xi32> = dense<[72]>
  memref.global "private" @__ly_small_int_73_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_73_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_73_digits : memref<1xi32> = dense<[73]>
  memref.global "private" @__ly_small_int_74_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_74_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_74_digits : memref<1xi32> = dense<[74]>
  memref.global "private" @__ly_small_int_75_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_75_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_75_digits : memref<1xi32> = dense<[75]>
  memref.global "private" @__ly_small_int_76_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_76_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_76_digits : memref<1xi32> = dense<[76]>
  memref.global "private" @__ly_small_int_77_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_77_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_77_digits : memref<1xi32> = dense<[77]>
  memref.global "private" @__ly_small_int_78_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_78_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_78_digits : memref<1xi32> = dense<[78]>
  memref.global "private" @__ly_small_int_79_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_79_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_79_digits : memref<1xi32> = dense<[79]>
  memref.global "private" @__ly_small_int_80_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_80_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_80_digits : memref<1xi32> = dense<[80]>
  memref.global "private" @__ly_small_int_81_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_81_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_81_digits : memref<1xi32> = dense<[81]>
  memref.global "private" @__ly_small_int_82_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_82_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_82_digits : memref<1xi32> = dense<[82]>
  memref.global "private" @__ly_small_int_83_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_83_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_83_digits : memref<1xi32> = dense<[83]>
  memref.global "private" @__ly_small_int_84_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_84_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_84_digits : memref<1xi32> = dense<[84]>
  memref.global "private" @__ly_small_int_85_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_85_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_85_digits : memref<1xi32> = dense<[85]>
  memref.global "private" @__ly_small_int_86_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_86_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_86_digits : memref<1xi32> = dense<[86]>
  memref.global "private" @__ly_small_int_87_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_87_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_87_digits : memref<1xi32> = dense<[87]>
  memref.global "private" @__ly_small_int_88_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_88_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_88_digits : memref<1xi32> = dense<[88]>
  memref.global "private" @__ly_small_int_89_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_89_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_89_digits : memref<1xi32> = dense<[89]>
  memref.global "private" @__ly_small_int_90_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_90_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_90_digits : memref<1xi32> = dense<[90]>
  memref.global "private" @__ly_small_int_91_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_91_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_91_digits : memref<1xi32> = dense<[91]>
  memref.global "private" @__ly_small_int_92_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_92_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_92_digits : memref<1xi32> = dense<[92]>
  memref.global "private" @__ly_small_int_93_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_93_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_93_digits : memref<1xi32> = dense<[93]>
  memref.global "private" @__ly_small_int_94_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_94_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_94_digits : memref<1xi32> = dense<[94]>
  memref.global "private" @__ly_small_int_95_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_95_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_95_digits : memref<1xi32> = dense<[95]>
  memref.global "private" @__ly_small_int_96_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_96_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_96_digits : memref<1xi32> = dense<[96]>
  memref.global "private" @__ly_small_int_97_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_97_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_97_digits : memref<1xi32> = dense<[97]>
  memref.global "private" @__ly_small_int_98_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_98_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_98_digits : memref<1xi32> = dense<[98]>
  memref.global "private" @__ly_small_int_99_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_99_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_99_digits : memref<1xi32> = dense<[99]>
  memref.global "private" @__ly_small_int_100_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_100_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_100_digits : memref<1xi32> = dense<[100]>
  memref.global "private" @__ly_small_int_101_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_101_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_101_digits : memref<1xi32> = dense<[101]>
  memref.global "private" @__ly_small_int_102_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_102_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_102_digits : memref<1xi32> = dense<[102]>
  memref.global "private" @__ly_small_int_103_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_103_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_103_digits : memref<1xi32> = dense<[103]>
  memref.global "private" @__ly_small_int_104_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_104_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_104_digits : memref<1xi32> = dense<[104]>
  memref.global "private" @__ly_small_int_105_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_105_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_105_digits : memref<1xi32> = dense<[105]>
  memref.global "private" @__ly_small_int_106_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_106_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_106_digits : memref<1xi32> = dense<[106]>
  memref.global "private" @__ly_small_int_107_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_107_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_107_digits : memref<1xi32> = dense<[107]>
  memref.global "private" @__ly_small_int_108_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_108_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_108_digits : memref<1xi32> = dense<[108]>
  memref.global "private" @__ly_small_int_109_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_109_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_109_digits : memref<1xi32> = dense<[109]>
  memref.global "private" @__ly_small_int_110_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_110_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_110_digits : memref<1xi32> = dense<[110]>
  memref.global "private" @__ly_small_int_111_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_111_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_111_digits : memref<1xi32> = dense<[111]>
  memref.global "private" @__ly_small_int_112_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_112_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_112_digits : memref<1xi32> = dense<[112]>
  memref.global "private" @__ly_small_int_113_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_113_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_113_digits : memref<1xi32> = dense<[113]>
  memref.global "private" @__ly_small_int_114_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_114_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_114_digits : memref<1xi32> = dense<[114]>
  memref.global "private" @__ly_small_int_115_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_115_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_115_digits : memref<1xi32> = dense<[115]>
  memref.global "private" @__ly_small_int_116_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_116_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_116_digits : memref<1xi32> = dense<[116]>
  memref.global "private" @__ly_small_int_117_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_117_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_117_digits : memref<1xi32> = dense<[117]>
  memref.global "private" @__ly_small_int_118_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_118_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_118_digits : memref<1xi32> = dense<[118]>
  memref.global "private" @__ly_small_int_119_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_119_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_119_digits : memref<1xi32> = dense<[119]>
  memref.global "private" @__ly_small_int_120_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_120_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_120_digits : memref<1xi32> = dense<[120]>
  memref.global "private" @__ly_small_int_121_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_121_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_121_digits : memref<1xi32> = dense<[121]>
  memref.global "private" @__ly_small_int_122_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_122_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_122_digits : memref<1xi32> = dense<[122]>
  memref.global "private" @__ly_small_int_123_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_123_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_123_digits : memref<1xi32> = dense<[123]>
  memref.global "private" @__ly_small_int_124_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_124_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_124_digits : memref<1xi32> = dense<[124]>
  memref.global "private" @__ly_small_int_125_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_125_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_125_digits : memref<1xi32> = dense<[125]>
  memref.global "private" @__ly_small_int_126_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_126_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_126_digits : memref<1xi32> = dense<[126]>
  memref.global "private" @__ly_small_int_127_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_127_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_127_digits : memref<1xi32> = dense<[127]>
  memref.global "private" @__ly_small_int_128_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_128_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_128_digits : memref<1xi32> = dense<[128]>
  memref.global "private" @__ly_small_int_129_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_129_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_129_digits : memref<1xi32> = dense<[129]>
  memref.global "private" @__ly_small_int_130_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_130_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_130_digits : memref<1xi32> = dense<[130]>
  memref.global "private" @__ly_small_int_131_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_131_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_131_digits : memref<1xi32> = dense<[131]>
  memref.global "private" @__ly_small_int_132_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_132_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_132_digits : memref<1xi32> = dense<[132]>
  memref.global "private" @__ly_small_int_133_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_133_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_133_digits : memref<1xi32> = dense<[133]>
  memref.global "private" @__ly_small_int_134_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_134_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_134_digits : memref<1xi32> = dense<[134]>
  memref.global "private" @__ly_small_int_135_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_135_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_135_digits : memref<1xi32> = dense<[135]>
  memref.global "private" @__ly_small_int_136_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_136_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_136_digits : memref<1xi32> = dense<[136]>
  memref.global "private" @__ly_small_int_137_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_137_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_137_digits : memref<1xi32> = dense<[137]>
  memref.global "private" @__ly_small_int_138_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_138_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_138_digits : memref<1xi32> = dense<[138]>
  memref.global "private" @__ly_small_int_139_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_139_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_139_digits : memref<1xi32> = dense<[139]>
  memref.global "private" @__ly_small_int_140_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_140_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_140_digits : memref<1xi32> = dense<[140]>
  memref.global "private" @__ly_small_int_141_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_141_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_141_digits : memref<1xi32> = dense<[141]>
  memref.global "private" @__ly_small_int_142_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_142_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_142_digits : memref<1xi32> = dense<[142]>
  memref.global "private" @__ly_small_int_143_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_143_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_143_digits : memref<1xi32> = dense<[143]>
  memref.global "private" @__ly_small_int_144_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_144_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_144_digits : memref<1xi32> = dense<[144]>
  memref.global "private" @__ly_small_int_145_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_145_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_145_digits : memref<1xi32> = dense<[145]>
  memref.global "private" @__ly_small_int_146_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_146_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_146_digits : memref<1xi32> = dense<[146]>
  memref.global "private" @__ly_small_int_147_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_147_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_147_digits : memref<1xi32> = dense<[147]>
  memref.global "private" @__ly_small_int_148_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_148_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_148_digits : memref<1xi32> = dense<[148]>
  memref.global "private" @__ly_small_int_149_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_149_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_149_digits : memref<1xi32> = dense<[149]>
  memref.global "private" @__ly_small_int_150_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_150_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_150_digits : memref<1xi32> = dense<[150]>
  memref.global "private" @__ly_small_int_151_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_151_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_151_digits : memref<1xi32> = dense<[151]>
  memref.global "private" @__ly_small_int_152_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_152_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_152_digits : memref<1xi32> = dense<[152]>
  memref.global "private" @__ly_small_int_153_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_153_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_153_digits : memref<1xi32> = dense<[153]>
  memref.global "private" @__ly_small_int_154_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_154_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_154_digits : memref<1xi32> = dense<[154]>
  memref.global "private" @__ly_small_int_155_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_155_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_155_digits : memref<1xi32> = dense<[155]>
  memref.global "private" @__ly_small_int_156_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_156_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_156_digits : memref<1xi32> = dense<[156]>
  memref.global "private" @__ly_small_int_157_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_157_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_157_digits : memref<1xi32> = dense<[157]>
  memref.global "private" @__ly_small_int_158_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_158_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_158_digits : memref<1xi32> = dense<[158]>
  memref.global "private" @__ly_small_int_159_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_159_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_159_digits : memref<1xi32> = dense<[159]>
  memref.global "private" @__ly_small_int_160_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_160_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_160_digits : memref<1xi32> = dense<[160]>
  memref.global "private" @__ly_small_int_161_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_161_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_161_digits : memref<1xi32> = dense<[161]>
  memref.global "private" @__ly_small_int_162_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_162_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_162_digits : memref<1xi32> = dense<[162]>
  memref.global "private" @__ly_small_int_163_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_163_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_163_digits : memref<1xi32> = dense<[163]>
  memref.global "private" @__ly_small_int_164_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_164_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_164_digits : memref<1xi32> = dense<[164]>
  memref.global "private" @__ly_small_int_165_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_165_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_165_digits : memref<1xi32> = dense<[165]>
  memref.global "private" @__ly_small_int_166_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_166_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_166_digits : memref<1xi32> = dense<[166]>
  memref.global "private" @__ly_small_int_167_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_167_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_167_digits : memref<1xi32> = dense<[167]>
  memref.global "private" @__ly_small_int_168_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_168_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_168_digits : memref<1xi32> = dense<[168]>
  memref.global "private" @__ly_small_int_169_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_169_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_169_digits : memref<1xi32> = dense<[169]>
  memref.global "private" @__ly_small_int_170_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_170_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_170_digits : memref<1xi32> = dense<[170]>
  memref.global "private" @__ly_small_int_171_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_171_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_171_digits : memref<1xi32> = dense<[171]>
  memref.global "private" @__ly_small_int_172_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_172_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_172_digits : memref<1xi32> = dense<[172]>
  memref.global "private" @__ly_small_int_173_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_173_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_173_digits : memref<1xi32> = dense<[173]>
  memref.global "private" @__ly_small_int_174_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_174_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_174_digits : memref<1xi32> = dense<[174]>
  memref.global "private" @__ly_small_int_175_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_175_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_175_digits : memref<1xi32> = dense<[175]>
  memref.global "private" @__ly_small_int_176_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_176_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_176_digits : memref<1xi32> = dense<[176]>
  memref.global "private" @__ly_small_int_177_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_177_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_177_digits : memref<1xi32> = dense<[177]>
  memref.global "private" @__ly_small_int_178_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_178_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_178_digits : memref<1xi32> = dense<[178]>
  memref.global "private" @__ly_small_int_179_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_179_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_179_digits : memref<1xi32> = dense<[179]>
  memref.global "private" @__ly_small_int_180_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_180_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_180_digits : memref<1xi32> = dense<[180]>
  memref.global "private" @__ly_small_int_181_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_181_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_181_digits : memref<1xi32> = dense<[181]>
  memref.global "private" @__ly_small_int_182_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_182_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_182_digits : memref<1xi32> = dense<[182]>
  memref.global "private" @__ly_small_int_183_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_183_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_183_digits : memref<1xi32> = dense<[183]>
  memref.global "private" @__ly_small_int_184_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_184_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_184_digits : memref<1xi32> = dense<[184]>
  memref.global "private" @__ly_small_int_185_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_185_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_185_digits : memref<1xi32> = dense<[185]>
  memref.global "private" @__ly_small_int_186_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_186_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_186_digits : memref<1xi32> = dense<[186]>
  memref.global "private" @__ly_small_int_187_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_187_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_187_digits : memref<1xi32> = dense<[187]>
  memref.global "private" @__ly_small_int_188_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_188_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_188_digits : memref<1xi32> = dense<[188]>
  memref.global "private" @__ly_small_int_189_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_189_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_189_digits : memref<1xi32> = dense<[189]>
  memref.global "private" @__ly_small_int_190_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_190_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_190_digits : memref<1xi32> = dense<[190]>
  memref.global "private" @__ly_small_int_191_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_191_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_191_digits : memref<1xi32> = dense<[191]>
  memref.global "private" @__ly_small_int_192_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_192_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_192_digits : memref<1xi32> = dense<[192]>
  memref.global "private" @__ly_small_int_193_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_193_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_193_digits : memref<1xi32> = dense<[193]>
  memref.global "private" @__ly_small_int_194_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_194_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_194_digits : memref<1xi32> = dense<[194]>
  memref.global "private" @__ly_small_int_195_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_195_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_195_digits : memref<1xi32> = dense<[195]>
  memref.global "private" @__ly_small_int_196_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_196_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_196_digits : memref<1xi32> = dense<[196]>
  memref.global "private" @__ly_small_int_197_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_197_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_197_digits : memref<1xi32> = dense<[197]>
  memref.global "private" @__ly_small_int_198_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_198_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_198_digits : memref<1xi32> = dense<[198]>
  memref.global "private" @__ly_small_int_199_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_199_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_199_digits : memref<1xi32> = dense<[199]>
  memref.global "private" @__ly_small_int_200_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_200_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_200_digits : memref<1xi32> = dense<[200]>
  memref.global "private" @__ly_small_int_201_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_201_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_201_digits : memref<1xi32> = dense<[201]>
  memref.global "private" @__ly_small_int_202_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_202_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_202_digits : memref<1xi32> = dense<[202]>
  memref.global "private" @__ly_small_int_203_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_203_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_203_digits : memref<1xi32> = dense<[203]>
  memref.global "private" @__ly_small_int_204_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_204_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_204_digits : memref<1xi32> = dense<[204]>
  memref.global "private" @__ly_small_int_205_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_205_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_205_digits : memref<1xi32> = dense<[205]>
  memref.global "private" @__ly_small_int_206_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_206_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_206_digits : memref<1xi32> = dense<[206]>
  memref.global "private" @__ly_small_int_207_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_207_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_207_digits : memref<1xi32> = dense<[207]>
  memref.global "private" @__ly_small_int_208_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_208_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_208_digits : memref<1xi32> = dense<[208]>
  memref.global "private" @__ly_small_int_209_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_209_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_209_digits : memref<1xi32> = dense<[209]>
  memref.global "private" @__ly_small_int_210_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_210_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_210_digits : memref<1xi32> = dense<[210]>
  memref.global "private" @__ly_small_int_211_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_211_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_211_digits : memref<1xi32> = dense<[211]>
  memref.global "private" @__ly_small_int_212_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_212_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_212_digits : memref<1xi32> = dense<[212]>
  memref.global "private" @__ly_small_int_213_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_213_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_213_digits : memref<1xi32> = dense<[213]>
  memref.global "private" @__ly_small_int_214_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_214_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_214_digits : memref<1xi32> = dense<[214]>
  memref.global "private" @__ly_small_int_215_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_215_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_215_digits : memref<1xi32> = dense<[215]>
  memref.global "private" @__ly_small_int_216_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_216_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_216_digits : memref<1xi32> = dense<[216]>
  memref.global "private" @__ly_small_int_217_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_217_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_217_digits : memref<1xi32> = dense<[217]>
  memref.global "private" @__ly_small_int_218_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_218_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_218_digits : memref<1xi32> = dense<[218]>
  memref.global "private" @__ly_small_int_219_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_219_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_219_digits : memref<1xi32> = dense<[219]>
  memref.global "private" @__ly_small_int_220_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_220_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_220_digits : memref<1xi32> = dense<[220]>
  memref.global "private" @__ly_small_int_221_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_221_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_221_digits : memref<1xi32> = dense<[221]>
  memref.global "private" @__ly_small_int_222_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_222_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_222_digits : memref<1xi32> = dense<[222]>
  memref.global "private" @__ly_small_int_223_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_223_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_223_digits : memref<1xi32> = dense<[223]>
  memref.global "private" @__ly_small_int_224_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_224_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_224_digits : memref<1xi32> = dense<[224]>
  memref.global "private" @__ly_small_int_225_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_225_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_225_digits : memref<1xi32> = dense<[225]>
  memref.global "private" @__ly_small_int_226_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_226_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_226_digits : memref<1xi32> = dense<[226]>
  memref.global "private" @__ly_small_int_227_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_227_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_227_digits : memref<1xi32> = dense<[227]>
  memref.global "private" @__ly_small_int_228_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_228_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_228_digits : memref<1xi32> = dense<[228]>
  memref.global "private" @__ly_small_int_229_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_229_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_229_digits : memref<1xi32> = dense<[229]>
  memref.global "private" @__ly_small_int_230_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_230_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_230_digits : memref<1xi32> = dense<[230]>
  memref.global "private" @__ly_small_int_231_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_231_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_231_digits : memref<1xi32> = dense<[231]>
  memref.global "private" @__ly_small_int_232_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_232_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_232_digits : memref<1xi32> = dense<[232]>
  memref.global "private" @__ly_small_int_233_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_233_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_233_digits : memref<1xi32> = dense<[233]>
  memref.global "private" @__ly_small_int_234_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_234_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_234_digits : memref<1xi32> = dense<[234]>
  memref.global "private" @__ly_small_int_235_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_235_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_235_digits : memref<1xi32> = dense<[235]>
  memref.global "private" @__ly_small_int_236_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_236_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_236_digits : memref<1xi32> = dense<[236]>
  memref.global "private" @__ly_small_int_237_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_237_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_237_digits : memref<1xi32> = dense<[237]>
  memref.global "private" @__ly_small_int_238_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_238_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_238_digits : memref<1xi32> = dense<[238]>
  memref.global "private" @__ly_small_int_239_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_239_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_239_digits : memref<1xi32> = dense<[239]>
  memref.global "private" @__ly_small_int_240_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_240_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_240_digits : memref<1xi32> = dense<[240]>
  memref.global "private" @__ly_small_int_241_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_241_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_241_digits : memref<1xi32> = dense<[241]>
  memref.global "private" @__ly_small_int_242_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_242_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_242_digits : memref<1xi32> = dense<[242]>
  memref.global "private" @__ly_small_int_243_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_243_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_243_digits : memref<1xi32> = dense<[243]>
  memref.global "private" @__ly_small_int_244_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_244_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_244_digits : memref<1xi32> = dense<[244]>
  memref.global "private" @__ly_small_int_245_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_245_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_245_digits : memref<1xi32> = dense<[245]>
  memref.global "private" @__ly_small_int_246_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_246_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_246_digits : memref<1xi32> = dense<[246]>
  memref.global "private" @__ly_small_int_247_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_247_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_247_digits : memref<1xi32> = dense<[247]>
  memref.global "private" @__ly_small_int_248_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_248_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_248_digits : memref<1xi32> = dense<[248]>
  memref.global "private" @__ly_small_int_249_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_249_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_249_digits : memref<1xi32> = dense<[249]>
  memref.global "private" @__ly_small_int_250_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_250_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_250_digits : memref<1xi32> = dense<[250]>
  memref.global "private" @__ly_small_int_251_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_251_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_251_digits : memref<1xi32> = dense<[251]>
  memref.global "private" @__ly_small_int_252_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_252_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_252_digits : memref<1xi32> = dense<[252]>
  memref.global "private" @__ly_small_int_253_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_253_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_253_digits : memref<1xi32> = dense<[253]>
  memref.global "private" @__ly_small_int_254_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_254_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_254_digits : memref<1xi32> = dense<[254]>
  memref.global "private" @__ly_small_int_255_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_255_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_255_digits : memref<1xi32> = dense<[255]>
  memref.global "private" @__ly_small_int_256_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" @__ly_small_int_256_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" @__ly_small_int_256_digits : memref<1xi32> = dense<[256]>
  func.func private @__ly_long_small_from_i64(%value: i64) -> (i1, memref<2xi64>, memref<2xi64>, memref<1xi32>) {
    cf.switch %value : i64, [
      default: ^default,
      -5: ^small_m5,
      -4: ^small_m4,
      -3: ^small_m3,
      -2: ^small_m2,
      -1: ^small_m1,
      0: ^small_z,
      1: ^small_1,
      2: ^small_2,
      3: ^small_3,
      4: ^small_4,
      5: ^small_5,
      6: ^small_6,
      7: ^small_7,
      8: ^small_8,
      9: ^small_9,
      10: ^small_10,
      11: ^small_11,
      12: ^small_12,
      13: ^small_13,
      14: ^small_14,
      15: ^small_15,
      16: ^small_16,
      17: ^small_17,
      18: ^small_18,
      19: ^small_19,
      20: ^small_20,
      21: ^small_21,
      22: ^small_22,
      23: ^small_23,
      24: ^small_24,
      25: ^small_25,
      26: ^small_26,
      27: ^small_27,
      28: ^small_28,
      29: ^small_29,
      30: ^small_30,
      31: ^small_31,
      32: ^small_32,
      33: ^small_33,
      34: ^small_34,
      35: ^small_35,
      36: ^small_36,
      37: ^small_37,
      38: ^small_38,
      39: ^small_39,
      40: ^small_40,
      41: ^small_41,
      42: ^small_42,
      43: ^small_43,
      44: ^small_44,
      45: ^small_45,
      46: ^small_46,
      47: ^small_47,
      48: ^small_48,
      49: ^small_49,
      50: ^small_50,
      51: ^small_51,
      52: ^small_52,
      53: ^small_53,
      54: ^small_54,
      55: ^small_55,
      56: ^small_56,
      57: ^small_57,
      58: ^small_58,
      59: ^small_59,
      60: ^small_60,
      61: ^small_61,
      62: ^small_62,
      63: ^small_63,
      64: ^small_64,
      65: ^small_65,
      66: ^small_66,
      67: ^small_67,
      68: ^small_68,
      69: ^small_69,
      70: ^small_70,
      71: ^small_71,
      72: ^small_72,
      73: ^small_73,
      74: ^small_74,
      75: ^small_75,
      76: ^small_76,
      77: ^small_77,
      78: ^small_78,
      79: ^small_79,
      80: ^small_80,
      81: ^small_81,
      82: ^small_82,
      83: ^small_83,
      84: ^small_84,
      85: ^small_85,
      86: ^small_86,
      87: ^small_87,
      88: ^small_88,
      89: ^small_89,
      90: ^small_90,
      91: ^small_91,
      92: ^small_92,
      93: ^small_93,
      94: ^small_94,
      95: ^small_95,
      96: ^small_96,
      97: ^small_97,
      98: ^small_98,
      99: ^small_99,
      100: ^small_100,
      101: ^small_101,
      102: ^small_102,
      103: ^small_103,
      104: ^small_104,
      105: ^small_105,
      106: ^small_106,
      107: ^small_107,
      108: ^small_108,
      109: ^small_109,
      110: ^small_110,
      111: ^small_111,
      112: ^small_112,
      113: ^small_113,
      114: ^small_114,
      115: ^small_115,
      116: ^small_116,
      117: ^small_117,
      118: ^small_118,
      119: ^small_119,
      120: ^small_120,
      121: ^small_121,
      122: ^small_122,
      123: ^small_123,
      124: ^small_124,
      125: ^small_125,
      126: ^small_126,
      127: ^small_127,
      128: ^small_128,
      129: ^small_129,
      130: ^small_130,
      131: ^small_131,
      132: ^small_132,
      133: ^small_133,
      134: ^small_134,
      135: ^small_135,
      136: ^small_136,
      137: ^small_137,
      138: ^small_138,
      139: ^small_139,
      140: ^small_140,
      141: ^small_141,
      142: ^small_142,
      143: ^small_143,
      144: ^small_144,
      145: ^small_145,
      146: ^small_146,
      147: ^small_147,
      148: ^small_148,
      149: ^small_149,
      150: ^small_150,
      151: ^small_151,
      152: ^small_152,
      153: ^small_153,
      154: ^small_154,
      155: ^small_155,
      156: ^small_156,
      157: ^small_157,
      158: ^small_158,
      159: ^small_159,
      160: ^small_160,
      161: ^small_161,
      162: ^small_162,
      163: ^small_163,
      164: ^small_164,
      165: ^small_165,
      166: ^small_166,
      167: ^small_167,
      168: ^small_168,
      169: ^small_169,
      170: ^small_170,
      171: ^small_171,
      172: ^small_172,
      173: ^small_173,
      174: ^small_174,
      175: ^small_175,
      176: ^small_176,
      177: ^small_177,
      178: ^small_178,
      179: ^small_179,
      180: ^small_180,
      181: ^small_181,
      182: ^small_182,
      183: ^small_183,
      184: ^small_184,
      185: ^small_185,
      186: ^small_186,
      187: ^small_187,
      188: ^small_188,
      189: ^small_189,
      190: ^small_190,
      191: ^small_191,
      192: ^small_192,
      193: ^small_193,
      194: ^small_194,
      195: ^small_195,
      196: ^small_196,
      197: ^small_197,
      198: ^small_198,
      199: ^small_199,
      200: ^small_200,
      201: ^small_201,
      202: ^small_202,
      203: ^small_203,
      204: ^small_204,
      205: ^small_205,
      206: ^small_206,
      207: ^small_207,
      208: ^small_208,
      209: ^small_209,
      210: ^small_210,
      211: ^small_211,
      212: ^small_212,
      213: ^small_213,
      214: ^small_214,
      215: ^small_215,
      216: ^small_216,
      217: ^small_217,
      218: ^small_218,
      219: ^small_219,
      220: ^small_220,
      221: ^small_221,
      222: ^small_222,
      223: ^small_223,
      224: ^small_224,
      225: ^small_225,
      226: ^small_226,
      227: ^small_227,
      228: ^small_228,
      229: ^small_229,
      230: ^small_230,
      231: ^small_231,
      232: ^small_232,
      233: ^small_233,
      234: ^small_234,
      235: ^small_235,
      236: ^small_236,
      237: ^small_237,
      238: ^small_238,
      239: ^small_239,
      240: ^small_240,
      241: ^small_241,
      242: ^small_242,
      243: ^small_243,
      244: ^small_244,
      245: ^small_245,
      246: ^small_246,
      247: ^small_247,
      248: ^small_248,
      249: ^small_249,
      250: ^small_250,
      251: ^small_251,
      252: ^small_252,
      253: ^small_253,
      254: ^small_254,
      255: ^small_255,
      256: ^small_256
    ]
  ^small_m5:
    %hit_m5 = arith.constant true
    %header_m5 = memref.get_global @__ly_small_int_m5_header : memref<2xi64>
    %meta_m5 = memref.get_global @__ly_small_int_m5_meta : memref<2xi64>
    %digits_m5 = memref.get_global @__ly_small_int_m5_digits : memref<1xi32>
    func.return %hit_m5, %header_m5, %meta_m5, %digits_m5 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_m4:
    %hit_m4 = arith.constant true
    %header_m4 = memref.get_global @__ly_small_int_m4_header : memref<2xi64>
    %meta_m4 = memref.get_global @__ly_small_int_m4_meta : memref<2xi64>
    %digits_m4 = memref.get_global @__ly_small_int_m4_digits : memref<1xi32>
    func.return %hit_m4, %header_m4, %meta_m4, %digits_m4 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_m3:
    %hit_m3 = arith.constant true
    %header_m3 = memref.get_global @__ly_small_int_m3_header : memref<2xi64>
    %meta_m3 = memref.get_global @__ly_small_int_m3_meta : memref<2xi64>
    %digits_m3 = memref.get_global @__ly_small_int_m3_digits : memref<1xi32>
    func.return %hit_m3, %header_m3, %meta_m3, %digits_m3 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_m2:
    %hit_m2 = arith.constant true
    %header_m2 = memref.get_global @__ly_small_int_m2_header : memref<2xi64>
    %meta_m2 = memref.get_global @__ly_small_int_m2_meta : memref<2xi64>
    %digits_m2 = memref.get_global @__ly_small_int_m2_digits : memref<1xi32>
    func.return %hit_m2, %header_m2, %meta_m2, %digits_m2 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_m1:
    %hit_m1 = arith.constant true
    %header_m1 = memref.get_global @__ly_small_int_m1_header : memref<2xi64>
    %meta_m1 = memref.get_global @__ly_small_int_m1_meta : memref<2xi64>
    %digits_m1 = memref.get_global @__ly_small_int_m1_digits : memref<1xi32>
    func.return %hit_m1, %header_m1, %meta_m1, %digits_m1 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_z:
    %hit_z = arith.constant true
    %header_z = memref.get_global @__ly_small_int_z_header : memref<2xi64>
    %meta_z = memref.get_global @__ly_small_int_z_meta : memref<2xi64>
    %digits_z = memref.get_global @__ly_small_int_z_digits : memref<1xi32>
    func.return %hit_z, %header_z, %meta_z, %digits_z : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_1:
    %hit_1 = arith.constant true
    %header_1 = memref.get_global @__ly_small_int_1_header : memref<2xi64>
    %meta_1 = memref.get_global @__ly_small_int_1_meta : memref<2xi64>
    %digits_1 = memref.get_global @__ly_small_int_1_digits : memref<1xi32>
    func.return %hit_1, %header_1, %meta_1, %digits_1 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_2:
    %hit_2 = arith.constant true
    %header_2 = memref.get_global @__ly_small_int_2_header : memref<2xi64>
    %meta_2 = memref.get_global @__ly_small_int_2_meta : memref<2xi64>
    %digits_2 = memref.get_global @__ly_small_int_2_digits : memref<1xi32>
    func.return %hit_2, %header_2, %meta_2, %digits_2 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_3:
    %hit_3 = arith.constant true
    %header_3 = memref.get_global @__ly_small_int_3_header : memref<2xi64>
    %meta_3 = memref.get_global @__ly_small_int_3_meta : memref<2xi64>
    %digits_3 = memref.get_global @__ly_small_int_3_digits : memref<1xi32>
    func.return %hit_3, %header_3, %meta_3, %digits_3 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_4:
    %hit_4 = arith.constant true
    %header_4 = memref.get_global @__ly_small_int_4_header : memref<2xi64>
    %meta_4 = memref.get_global @__ly_small_int_4_meta : memref<2xi64>
    %digits_4 = memref.get_global @__ly_small_int_4_digits : memref<1xi32>
    func.return %hit_4, %header_4, %meta_4, %digits_4 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_5:
    %hit_5 = arith.constant true
    %header_5 = memref.get_global @__ly_small_int_5_header : memref<2xi64>
    %meta_5 = memref.get_global @__ly_small_int_5_meta : memref<2xi64>
    %digits_5 = memref.get_global @__ly_small_int_5_digits : memref<1xi32>
    func.return %hit_5, %header_5, %meta_5, %digits_5 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_6:
    %hit_6 = arith.constant true
    %header_6 = memref.get_global @__ly_small_int_6_header : memref<2xi64>
    %meta_6 = memref.get_global @__ly_small_int_6_meta : memref<2xi64>
    %digits_6 = memref.get_global @__ly_small_int_6_digits : memref<1xi32>
    func.return %hit_6, %header_6, %meta_6, %digits_6 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_7:
    %hit_7 = arith.constant true
    %header_7 = memref.get_global @__ly_small_int_7_header : memref<2xi64>
    %meta_7 = memref.get_global @__ly_small_int_7_meta : memref<2xi64>
    %digits_7 = memref.get_global @__ly_small_int_7_digits : memref<1xi32>
    func.return %hit_7, %header_7, %meta_7, %digits_7 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_8:
    %hit_8 = arith.constant true
    %header_8 = memref.get_global @__ly_small_int_8_header : memref<2xi64>
    %meta_8 = memref.get_global @__ly_small_int_8_meta : memref<2xi64>
    %digits_8 = memref.get_global @__ly_small_int_8_digits : memref<1xi32>
    func.return %hit_8, %header_8, %meta_8, %digits_8 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_9:
    %hit_9 = arith.constant true
    %header_9 = memref.get_global @__ly_small_int_9_header : memref<2xi64>
    %meta_9 = memref.get_global @__ly_small_int_9_meta : memref<2xi64>
    %digits_9 = memref.get_global @__ly_small_int_9_digits : memref<1xi32>
    func.return %hit_9, %header_9, %meta_9, %digits_9 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_10:
    %hit_10 = arith.constant true
    %header_10 = memref.get_global @__ly_small_int_10_header : memref<2xi64>
    %meta_10 = memref.get_global @__ly_small_int_10_meta : memref<2xi64>
    %digits_10 = memref.get_global @__ly_small_int_10_digits : memref<1xi32>
    func.return %hit_10, %header_10, %meta_10, %digits_10 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_11:
    %hit_11 = arith.constant true
    %header_11 = memref.get_global @__ly_small_int_11_header : memref<2xi64>
    %meta_11 = memref.get_global @__ly_small_int_11_meta : memref<2xi64>
    %digits_11 = memref.get_global @__ly_small_int_11_digits : memref<1xi32>
    func.return %hit_11, %header_11, %meta_11, %digits_11 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_12:
    %hit_12 = arith.constant true
    %header_12 = memref.get_global @__ly_small_int_12_header : memref<2xi64>
    %meta_12 = memref.get_global @__ly_small_int_12_meta : memref<2xi64>
    %digits_12 = memref.get_global @__ly_small_int_12_digits : memref<1xi32>
    func.return %hit_12, %header_12, %meta_12, %digits_12 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_13:
    %hit_13 = arith.constant true
    %header_13 = memref.get_global @__ly_small_int_13_header : memref<2xi64>
    %meta_13 = memref.get_global @__ly_small_int_13_meta : memref<2xi64>
    %digits_13 = memref.get_global @__ly_small_int_13_digits : memref<1xi32>
    func.return %hit_13, %header_13, %meta_13, %digits_13 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_14:
    %hit_14 = arith.constant true
    %header_14 = memref.get_global @__ly_small_int_14_header : memref<2xi64>
    %meta_14 = memref.get_global @__ly_small_int_14_meta : memref<2xi64>
    %digits_14 = memref.get_global @__ly_small_int_14_digits : memref<1xi32>
    func.return %hit_14, %header_14, %meta_14, %digits_14 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_15:
    %hit_15 = arith.constant true
    %header_15 = memref.get_global @__ly_small_int_15_header : memref<2xi64>
    %meta_15 = memref.get_global @__ly_small_int_15_meta : memref<2xi64>
    %digits_15 = memref.get_global @__ly_small_int_15_digits : memref<1xi32>
    func.return %hit_15, %header_15, %meta_15, %digits_15 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_16:
    %hit_16 = arith.constant true
    %header_16 = memref.get_global @__ly_small_int_16_header : memref<2xi64>
    %meta_16 = memref.get_global @__ly_small_int_16_meta : memref<2xi64>
    %digits_16 = memref.get_global @__ly_small_int_16_digits : memref<1xi32>
    func.return %hit_16, %header_16, %meta_16, %digits_16 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_17:
    %hit_17 = arith.constant true
    %header_17 = memref.get_global @__ly_small_int_17_header : memref<2xi64>
    %meta_17 = memref.get_global @__ly_small_int_17_meta : memref<2xi64>
    %digits_17 = memref.get_global @__ly_small_int_17_digits : memref<1xi32>
    func.return %hit_17, %header_17, %meta_17, %digits_17 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_18:
    %hit_18 = arith.constant true
    %header_18 = memref.get_global @__ly_small_int_18_header : memref<2xi64>
    %meta_18 = memref.get_global @__ly_small_int_18_meta : memref<2xi64>
    %digits_18 = memref.get_global @__ly_small_int_18_digits : memref<1xi32>
    func.return %hit_18, %header_18, %meta_18, %digits_18 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_19:
    %hit_19 = arith.constant true
    %header_19 = memref.get_global @__ly_small_int_19_header : memref<2xi64>
    %meta_19 = memref.get_global @__ly_small_int_19_meta : memref<2xi64>
    %digits_19 = memref.get_global @__ly_small_int_19_digits : memref<1xi32>
    func.return %hit_19, %header_19, %meta_19, %digits_19 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_20:
    %hit_20 = arith.constant true
    %header_20 = memref.get_global @__ly_small_int_20_header : memref<2xi64>
    %meta_20 = memref.get_global @__ly_small_int_20_meta : memref<2xi64>
    %digits_20 = memref.get_global @__ly_small_int_20_digits : memref<1xi32>
    func.return %hit_20, %header_20, %meta_20, %digits_20 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_21:
    %hit_21 = arith.constant true
    %header_21 = memref.get_global @__ly_small_int_21_header : memref<2xi64>
    %meta_21 = memref.get_global @__ly_small_int_21_meta : memref<2xi64>
    %digits_21 = memref.get_global @__ly_small_int_21_digits : memref<1xi32>
    func.return %hit_21, %header_21, %meta_21, %digits_21 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_22:
    %hit_22 = arith.constant true
    %header_22 = memref.get_global @__ly_small_int_22_header : memref<2xi64>
    %meta_22 = memref.get_global @__ly_small_int_22_meta : memref<2xi64>
    %digits_22 = memref.get_global @__ly_small_int_22_digits : memref<1xi32>
    func.return %hit_22, %header_22, %meta_22, %digits_22 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_23:
    %hit_23 = arith.constant true
    %header_23 = memref.get_global @__ly_small_int_23_header : memref<2xi64>
    %meta_23 = memref.get_global @__ly_small_int_23_meta : memref<2xi64>
    %digits_23 = memref.get_global @__ly_small_int_23_digits : memref<1xi32>
    func.return %hit_23, %header_23, %meta_23, %digits_23 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_24:
    %hit_24 = arith.constant true
    %header_24 = memref.get_global @__ly_small_int_24_header : memref<2xi64>
    %meta_24 = memref.get_global @__ly_small_int_24_meta : memref<2xi64>
    %digits_24 = memref.get_global @__ly_small_int_24_digits : memref<1xi32>
    func.return %hit_24, %header_24, %meta_24, %digits_24 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_25:
    %hit_25 = arith.constant true
    %header_25 = memref.get_global @__ly_small_int_25_header : memref<2xi64>
    %meta_25 = memref.get_global @__ly_small_int_25_meta : memref<2xi64>
    %digits_25 = memref.get_global @__ly_small_int_25_digits : memref<1xi32>
    func.return %hit_25, %header_25, %meta_25, %digits_25 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_26:
    %hit_26 = arith.constant true
    %header_26 = memref.get_global @__ly_small_int_26_header : memref<2xi64>
    %meta_26 = memref.get_global @__ly_small_int_26_meta : memref<2xi64>
    %digits_26 = memref.get_global @__ly_small_int_26_digits : memref<1xi32>
    func.return %hit_26, %header_26, %meta_26, %digits_26 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_27:
    %hit_27 = arith.constant true
    %header_27 = memref.get_global @__ly_small_int_27_header : memref<2xi64>
    %meta_27 = memref.get_global @__ly_small_int_27_meta : memref<2xi64>
    %digits_27 = memref.get_global @__ly_small_int_27_digits : memref<1xi32>
    func.return %hit_27, %header_27, %meta_27, %digits_27 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_28:
    %hit_28 = arith.constant true
    %header_28 = memref.get_global @__ly_small_int_28_header : memref<2xi64>
    %meta_28 = memref.get_global @__ly_small_int_28_meta : memref<2xi64>
    %digits_28 = memref.get_global @__ly_small_int_28_digits : memref<1xi32>
    func.return %hit_28, %header_28, %meta_28, %digits_28 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_29:
    %hit_29 = arith.constant true
    %header_29 = memref.get_global @__ly_small_int_29_header : memref<2xi64>
    %meta_29 = memref.get_global @__ly_small_int_29_meta : memref<2xi64>
    %digits_29 = memref.get_global @__ly_small_int_29_digits : memref<1xi32>
    func.return %hit_29, %header_29, %meta_29, %digits_29 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_30:
    %hit_30 = arith.constant true
    %header_30 = memref.get_global @__ly_small_int_30_header : memref<2xi64>
    %meta_30 = memref.get_global @__ly_small_int_30_meta : memref<2xi64>
    %digits_30 = memref.get_global @__ly_small_int_30_digits : memref<1xi32>
    func.return %hit_30, %header_30, %meta_30, %digits_30 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_31:
    %hit_31 = arith.constant true
    %header_31 = memref.get_global @__ly_small_int_31_header : memref<2xi64>
    %meta_31 = memref.get_global @__ly_small_int_31_meta : memref<2xi64>
    %digits_31 = memref.get_global @__ly_small_int_31_digits : memref<1xi32>
    func.return %hit_31, %header_31, %meta_31, %digits_31 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_32:
    %hit_32 = arith.constant true
    %header_32 = memref.get_global @__ly_small_int_32_header : memref<2xi64>
    %meta_32 = memref.get_global @__ly_small_int_32_meta : memref<2xi64>
    %digits_32 = memref.get_global @__ly_small_int_32_digits : memref<1xi32>
    func.return %hit_32, %header_32, %meta_32, %digits_32 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_33:
    %hit_33 = arith.constant true
    %header_33 = memref.get_global @__ly_small_int_33_header : memref<2xi64>
    %meta_33 = memref.get_global @__ly_small_int_33_meta : memref<2xi64>
    %digits_33 = memref.get_global @__ly_small_int_33_digits : memref<1xi32>
    func.return %hit_33, %header_33, %meta_33, %digits_33 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_34:
    %hit_34 = arith.constant true
    %header_34 = memref.get_global @__ly_small_int_34_header : memref<2xi64>
    %meta_34 = memref.get_global @__ly_small_int_34_meta : memref<2xi64>
    %digits_34 = memref.get_global @__ly_small_int_34_digits : memref<1xi32>
    func.return %hit_34, %header_34, %meta_34, %digits_34 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_35:
    %hit_35 = arith.constant true
    %header_35 = memref.get_global @__ly_small_int_35_header : memref<2xi64>
    %meta_35 = memref.get_global @__ly_small_int_35_meta : memref<2xi64>
    %digits_35 = memref.get_global @__ly_small_int_35_digits : memref<1xi32>
    func.return %hit_35, %header_35, %meta_35, %digits_35 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_36:
    %hit_36 = arith.constant true
    %header_36 = memref.get_global @__ly_small_int_36_header : memref<2xi64>
    %meta_36 = memref.get_global @__ly_small_int_36_meta : memref<2xi64>
    %digits_36 = memref.get_global @__ly_small_int_36_digits : memref<1xi32>
    func.return %hit_36, %header_36, %meta_36, %digits_36 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_37:
    %hit_37 = arith.constant true
    %header_37 = memref.get_global @__ly_small_int_37_header : memref<2xi64>
    %meta_37 = memref.get_global @__ly_small_int_37_meta : memref<2xi64>
    %digits_37 = memref.get_global @__ly_small_int_37_digits : memref<1xi32>
    func.return %hit_37, %header_37, %meta_37, %digits_37 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_38:
    %hit_38 = arith.constant true
    %header_38 = memref.get_global @__ly_small_int_38_header : memref<2xi64>
    %meta_38 = memref.get_global @__ly_small_int_38_meta : memref<2xi64>
    %digits_38 = memref.get_global @__ly_small_int_38_digits : memref<1xi32>
    func.return %hit_38, %header_38, %meta_38, %digits_38 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_39:
    %hit_39 = arith.constant true
    %header_39 = memref.get_global @__ly_small_int_39_header : memref<2xi64>
    %meta_39 = memref.get_global @__ly_small_int_39_meta : memref<2xi64>
    %digits_39 = memref.get_global @__ly_small_int_39_digits : memref<1xi32>
    func.return %hit_39, %header_39, %meta_39, %digits_39 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_40:
    %hit_40 = arith.constant true
    %header_40 = memref.get_global @__ly_small_int_40_header : memref<2xi64>
    %meta_40 = memref.get_global @__ly_small_int_40_meta : memref<2xi64>
    %digits_40 = memref.get_global @__ly_small_int_40_digits : memref<1xi32>
    func.return %hit_40, %header_40, %meta_40, %digits_40 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_41:
    %hit_41 = arith.constant true
    %header_41 = memref.get_global @__ly_small_int_41_header : memref<2xi64>
    %meta_41 = memref.get_global @__ly_small_int_41_meta : memref<2xi64>
    %digits_41 = memref.get_global @__ly_small_int_41_digits : memref<1xi32>
    func.return %hit_41, %header_41, %meta_41, %digits_41 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_42:
    %hit_42 = arith.constant true
    %header_42 = memref.get_global @__ly_small_int_42_header : memref<2xi64>
    %meta_42 = memref.get_global @__ly_small_int_42_meta : memref<2xi64>
    %digits_42 = memref.get_global @__ly_small_int_42_digits : memref<1xi32>
    func.return %hit_42, %header_42, %meta_42, %digits_42 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_43:
    %hit_43 = arith.constant true
    %header_43 = memref.get_global @__ly_small_int_43_header : memref<2xi64>
    %meta_43 = memref.get_global @__ly_small_int_43_meta : memref<2xi64>
    %digits_43 = memref.get_global @__ly_small_int_43_digits : memref<1xi32>
    func.return %hit_43, %header_43, %meta_43, %digits_43 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_44:
    %hit_44 = arith.constant true
    %header_44 = memref.get_global @__ly_small_int_44_header : memref<2xi64>
    %meta_44 = memref.get_global @__ly_small_int_44_meta : memref<2xi64>
    %digits_44 = memref.get_global @__ly_small_int_44_digits : memref<1xi32>
    func.return %hit_44, %header_44, %meta_44, %digits_44 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_45:
    %hit_45 = arith.constant true
    %header_45 = memref.get_global @__ly_small_int_45_header : memref<2xi64>
    %meta_45 = memref.get_global @__ly_small_int_45_meta : memref<2xi64>
    %digits_45 = memref.get_global @__ly_small_int_45_digits : memref<1xi32>
    func.return %hit_45, %header_45, %meta_45, %digits_45 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_46:
    %hit_46 = arith.constant true
    %header_46 = memref.get_global @__ly_small_int_46_header : memref<2xi64>
    %meta_46 = memref.get_global @__ly_small_int_46_meta : memref<2xi64>
    %digits_46 = memref.get_global @__ly_small_int_46_digits : memref<1xi32>
    func.return %hit_46, %header_46, %meta_46, %digits_46 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_47:
    %hit_47 = arith.constant true
    %header_47 = memref.get_global @__ly_small_int_47_header : memref<2xi64>
    %meta_47 = memref.get_global @__ly_small_int_47_meta : memref<2xi64>
    %digits_47 = memref.get_global @__ly_small_int_47_digits : memref<1xi32>
    func.return %hit_47, %header_47, %meta_47, %digits_47 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_48:
    %hit_48 = arith.constant true
    %header_48 = memref.get_global @__ly_small_int_48_header : memref<2xi64>
    %meta_48 = memref.get_global @__ly_small_int_48_meta : memref<2xi64>
    %digits_48 = memref.get_global @__ly_small_int_48_digits : memref<1xi32>
    func.return %hit_48, %header_48, %meta_48, %digits_48 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_49:
    %hit_49 = arith.constant true
    %header_49 = memref.get_global @__ly_small_int_49_header : memref<2xi64>
    %meta_49 = memref.get_global @__ly_small_int_49_meta : memref<2xi64>
    %digits_49 = memref.get_global @__ly_small_int_49_digits : memref<1xi32>
    func.return %hit_49, %header_49, %meta_49, %digits_49 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_50:
    %hit_50 = arith.constant true
    %header_50 = memref.get_global @__ly_small_int_50_header : memref<2xi64>
    %meta_50 = memref.get_global @__ly_small_int_50_meta : memref<2xi64>
    %digits_50 = memref.get_global @__ly_small_int_50_digits : memref<1xi32>
    func.return %hit_50, %header_50, %meta_50, %digits_50 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_51:
    %hit_51 = arith.constant true
    %header_51 = memref.get_global @__ly_small_int_51_header : memref<2xi64>
    %meta_51 = memref.get_global @__ly_small_int_51_meta : memref<2xi64>
    %digits_51 = memref.get_global @__ly_small_int_51_digits : memref<1xi32>
    func.return %hit_51, %header_51, %meta_51, %digits_51 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_52:
    %hit_52 = arith.constant true
    %header_52 = memref.get_global @__ly_small_int_52_header : memref<2xi64>
    %meta_52 = memref.get_global @__ly_small_int_52_meta : memref<2xi64>
    %digits_52 = memref.get_global @__ly_small_int_52_digits : memref<1xi32>
    func.return %hit_52, %header_52, %meta_52, %digits_52 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_53:
    %hit_53 = arith.constant true
    %header_53 = memref.get_global @__ly_small_int_53_header : memref<2xi64>
    %meta_53 = memref.get_global @__ly_small_int_53_meta : memref<2xi64>
    %digits_53 = memref.get_global @__ly_small_int_53_digits : memref<1xi32>
    func.return %hit_53, %header_53, %meta_53, %digits_53 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_54:
    %hit_54 = arith.constant true
    %header_54 = memref.get_global @__ly_small_int_54_header : memref<2xi64>
    %meta_54 = memref.get_global @__ly_small_int_54_meta : memref<2xi64>
    %digits_54 = memref.get_global @__ly_small_int_54_digits : memref<1xi32>
    func.return %hit_54, %header_54, %meta_54, %digits_54 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_55:
    %hit_55 = arith.constant true
    %header_55 = memref.get_global @__ly_small_int_55_header : memref<2xi64>
    %meta_55 = memref.get_global @__ly_small_int_55_meta : memref<2xi64>
    %digits_55 = memref.get_global @__ly_small_int_55_digits : memref<1xi32>
    func.return %hit_55, %header_55, %meta_55, %digits_55 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_56:
    %hit_56 = arith.constant true
    %header_56 = memref.get_global @__ly_small_int_56_header : memref<2xi64>
    %meta_56 = memref.get_global @__ly_small_int_56_meta : memref<2xi64>
    %digits_56 = memref.get_global @__ly_small_int_56_digits : memref<1xi32>
    func.return %hit_56, %header_56, %meta_56, %digits_56 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_57:
    %hit_57 = arith.constant true
    %header_57 = memref.get_global @__ly_small_int_57_header : memref<2xi64>
    %meta_57 = memref.get_global @__ly_small_int_57_meta : memref<2xi64>
    %digits_57 = memref.get_global @__ly_small_int_57_digits : memref<1xi32>
    func.return %hit_57, %header_57, %meta_57, %digits_57 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_58:
    %hit_58 = arith.constant true
    %header_58 = memref.get_global @__ly_small_int_58_header : memref<2xi64>
    %meta_58 = memref.get_global @__ly_small_int_58_meta : memref<2xi64>
    %digits_58 = memref.get_global @__ly_small_int_58_digits : memref<1xi32>
    func.return %hit_58, %header_58, %meta_58, %digits_58 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_59:
    %hit_59 = arith.constant true
    %header_59 = memref.get_global @__ly_small_int_59_header : memref<2xi64>
    %meta_59 = memref.get_global @__ly_small_int_59_meta : memref<2xi64>
    %digits_59 = memref.get_global @__ly_small_int_59_digits : memref<1xi32>
    func.return %hit_59, %header_59, %meta_59, %digits_59 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_60:
    %hit_60 = arith.constant true
    %header_60 = memref.get_global @__ly_small_int_60_header : memref<2xi64>
    %meta_60 = memref.get_global @__ly_small_int_60_meta : memref<2xi64>
    %digits_60 = memref.get_global @__ly_small_int_60_digits : memref<1xi32>
    func.return %hit_60, %header_60, %meta_60, %digits_60 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_61:
    %hit_61 = arith.constant true
    %header_61 = memref.get_global @__ly_small_int_61_header : memref<2xi64>
    %meta_61 = memref.get_global @__ly_small_int_61_meta : memref<2xi64>
    %digits_61 = memref.get_global @__ly_small_int_61_digits : memref<1xi32>
    func.return %hit_61, %header_61, %meta_61, %digits_61 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_62:
    %hit_62 = arith.constant true
    %header_62 = memref.get_global @__ly_small_int_62_header : memref<2xi64>
    %meta_62 = memref.get_global @__ly_small_int_62_meta : memref<2xi64>
    %digits_62 = memref.get_global @__ly_small_int_62_digits : memref<1xi32>
    func.return %hit_62, %header_62, %meta_62, %digits_62 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_63:
    %hit_63 = arith.constant true
    %header_63 = memref.get_global @__ly_small_int_63_header : memref<2xi64>
    %meta_63 = memref.get_global @__ly_small_int_63_meta : memref<2xi64>
    %digits_63 = memref.get_global @__ly_small_int_63_digits : memref<1xi32>
    func.return %hit_63, %header_63, %meta_63, %digits_63 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_64:
    %hit_64 = arith.constant true
    %header_64 = memref.get_global @__ly_small_int_64_header : memref<2xi64>
    %meta_64 = memref.get_global @__ly_small_int_64_meta : memref<2xi64>
    %digits_64 = memref.get_global @__ly_small_int_64_digits : memref<1xi32>
    func.return %hit_64, %header_64, %meta_64, %digits_64 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_65:
    %hit_65 = arith.constant true
    %header_65 = memref.get_global @__ly_small_int_65_header : memref<2xi64>
    %meta_65 = memref.get_global @__ly_small_int_65_meta : memref<2xi64>
    %digits_65 = memref.get_global @__ly_small_int_65_digits : memref<1xi32>
    func.return %hit_65, %header_65, %meta_65, %digits_65 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_66:
    %hit_66 = arith.constant true
    %header_66 = memref.get_global @__ly_small_int_66_header : memref<2xi64>
    %meta_66 = memref.get_global @__ly_small_int_66_meta : memref<2xi64>
    %digits_66 = memref.get_global @__ly_small_int_66_digits : memref<1xi32>
    func.return %hit_66, %header_66, %meta_66, %digits_66 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_67:
    %hit_67 = arith.constant true
    %header_67 = memref.get_global @__ly_small_int_67_header : memref<2xi64>
    %meta_67 = memref.get_global @__ly_small_int_67_meta : memref<2xi64>
    %digits_67 = memref.get_global @__ly_small_int_67_digits : memref<1xi32>
    func.return %hit_67, %header_67, %meta_67, %digits_67 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_68:
    %hit_68 = arith.constant true
    %header_68 = memref.get_global @__ly_small_int_68_header : memref<2xi64>
    %meta_68 = memref.get_global @__ly_small_int_68_meta : memref<2xi64>
    %digits_68 = memref.get_global @__ly_small_int_68_digits : memref<1xi32>
    func.return %hit_68, %header_68, %meta_68, %digits_68 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_69:
    %hit_69 = arith.constant true
    %header_69 = memref.get_global @__ly_small_int_69_header : memref<2xi64>
    %meta_69 = memref.get_global @__ly_small_int_69_meta : memref<2xi64>
    %digits_69 = memref.get_global @__ly_small_int_69_digits : memref<1xi32>
    func.return %hit_69, %header_69, %meta_69, %digits_69 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_70:
    %hit_70 = arith.constant true
    %header_70 = memref.get_global @__ly_small_int_70_header : memref<2xi64>
    %meta_70 = memref.get_global @__ly_small_int_70_meta : memref<2xi64>
    %digits_70 = memref.get_global @__ly_small_int_70_digits : memref<1xi32>
    func.return %hit_70, %header_70, %meta_70, %digits_70 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_71:
    %hit_71 = arith.constant true
    %header_71 = memref.get_global @__ly_small_int_71_header : memref<2xi64>
    %meta_71 = memref.get_global @__ly_small_int_71_meta : memref<2xi64>
    %digits_71 = memref.get_global @__ly_small_int_71_digits : memref<1xi32>
    func.return %hit_71, %header_71, %meta_71, %digits_71 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_72:
    %hit_72 = arith.constant true
    %header_72 = memref.get_global @__ly_small_int_72_header : memref<2xi64>
    %meta_72 = memref.get_global @__ly_small_int_72_meta : memref<2xi64>
    %digits_72 = memref.get_global @__ly_small_int_72_digits : memref<1xi32>
    func.return %hit_72, %header_72, %meta_72, %digits_72 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_73:
    %hit_73 = arith.constant true
    %header_73 = memref.get_global @__ly_small_int_73_header : memref<2xi64>
    %meta_73 = memref.get_global @__ly_small_int_73_meta : memref<2xi64>
    %digits_73 = memref.get_global @__ly_small_int_73_digits : memref<1xi32>
    func.return %hit_73, %header_73, %meta_73, %digits_73 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_74:
    %hit_74 = arith.constant true
    %header_74 = memref.get_global @__ly_small_int_74_header : memref<2xi64>
    %meta_74 = memref.get_global @__ly_small_int_74_meta : memref<2xi64>
    %digits_74 = memref.get_global @__ly_small_int_74_digits : memref<1xi32>
    func.return %hit_74, %header_74, %meta_74, %digits_74 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_75:
    %hit_75 = arith.constant true
    %header_75 = memref.get_global @__ly_small_int_75_header : memref<2xi64>
    %meta_75 = memref.get_global @__ly_small_int_75_meta : memref<2xi64>
    %digits_75 = memref.get_global @__ly_small_int_75_digits : memref<1xi32>
    func.return %hit_75, %header_75, %meta_75, %digits_75 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_76:
    %hit_76 = arith.constant true
    %header_76 = memref.get_global @__ly_small_int_76_header : memref<2xi64>
    %meta_76 = memref.get_global @__ly_small_int_76_meta : memref<2xi64>
    %digits_76 = memref.get_global @__ly_small_int_76_digits : memref<1xi32>
    func.return %hit_76, %header_76, %meta_76, %digits_76 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_77:
    %hit_77 = arith.constant true
    %header_77 = memref.get_global @__ly_small_int_77_header : memref<2xi64>
    %meta_77 = memref.get_global @__ly_small_int_77_meta : memref<2xi64>
    %digits_77 = memref.get_global @__ly_small_int_77_digits : memref<1xi32>
    func.return %hit_77, %header_77, %meta_77, %digits_77 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_78:
    %hit_78 = arith.constant true
    %header_78 = memref.get_global @__ly_small_int_78_header : memref<2xi64>
    %meta_78 = memref.get_global @__ly_small_int_78_meta : memref<2xi64>
    %digits_78 = memref.get_global @__ly_small_int_78_digits : memref<1xi32>
    func.return %hit_78, %header_78, %meta_78, %digits_78 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_79:
    %hit_79 = arith.constant true
    %header_79 = memref.get_global @__ly_small_int_79_header : memref<2xi64>
    %meta_79 = memref.get_global @__ly_small_int_79_meta : memref<2xi64>
    %digits_79 = memref.get_global @__ly_small_int_79_digits : memref<1xi32>
    func.return %hit_79, %header_79, %meta_79, %digits_79 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_80:
    %hit_80 = arith.constant true
    %header_80 = memref.get_global @__ly_small_int_80_header : memref<2xi64>
    %meta_80 = memref.get_global @__ly_small_int_80_meta : memref<2xi64>
    %digits_80 = memref.get_global @__ly_small_int_80_digits : memref<1xi32>
    func.return %hit_80, %header_80, %meta_80, %digits_80 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_81:
    %hit_81 = arith.constant true
    %header_81 = memref.get_global @__ly_small_int_81_header : memref<2xi64>
    %meta_81 = memref.get_global @__ly_small_int_81_meta : memref<2xi64>
    %digits_81 = memref.get_global @__ly_small_int_81_digits : memref<1xi32>
    func.return %hit_81, %header_81, %meta_81, %digits_81 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_82:
    %hit_82 = arith.constant true
    %header_82 = memref.get_global @__ly_small_int_82_header : memref<2xi64>
    %meta_82 = memref.get_global @__ly_small_int_82_meta : memref<2xi64>
    %digits_82 = memref.get_global @__ly_small_int_82_digits : memref<1xi32>
    func.return %hit_82, %header_82, %meta_82, %digits_82 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_83:
    %hit_83 = arith.constant true
    %header_83 = memref.get_global @__ly_small_int_83_header : memref<2xi64>
    %meta_83 = memref.get_global @__ly_small_int_83_meta : memref<2xi64>
    %digits_83 = memref.get_global @__ly_small_int_83_digits : memref<1xi32>
    func.return %hit_83, %header_83, %meta_83, %digits_83 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_84:
    %hit_84 = arith.constant true
    %header_84 = memref.get_global @__ly_small_int_84_header : memref<2xi64>
    %meta_84 = memref.get_global @__ly_small_int_84_meta : memref<2xi64>
    %digits_84 = memref.get_global @__ly_small_int_84_digits : memref<1xi32>
    func.return %hit_84, %header_84, %meta_84, %digits_84 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_85:
    %hit_85 = arith.constant true
    %header_85 = memref.get_global @__ly_small_int_85_header : memref<2xi64>
    %meta_85 = memref.get_global @__ly_small_int_85_meta : memref<2xi64>
    %digits_85 = memref.get_global @__ly_small_int_85_digits : memref<1xi32>
    func.return %hit_85, %header_85, %meta_85, %digits_85 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_86:
    %hit_86 = arith.constant true
    %header_86 = memref.get_global @__ly_small_int_86_header : memref<2xi64>
    %meta_86 = memref.get_global @__ly_small_int_86_meta : memref<2xi64>
    %digits_86 = memref.get_global @__ly_small_int_86_digits : memref<1xi32>
    func.return %hit_86, %header_86, %meta_86, %digits_86 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_87:
    %hit_87 = arith.constant true
    %header_87 = memref.get_global @__ly_small_int_87_header : memref<2xi64>
    %meta_87 = memref.get_global @__ly_small_int_87_meta : memref<2xi64>
    %digits_87 = memref.get_global @__ly_small_int_87_digits : memref<1xi32>
    func.return %hit_87, %header_87, %meta_87, %digits_87 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_88:
    %hit_88 = arith.constant true
    %header_88 = memref.get_global @__ly_small_int_88_header : memref<2xi64>
    %meta_88 = memref.get_global @__ly_small_int_88_meta : memref<2xi64>
    %digits_88 = memref.get_global @__ly_small_int_88_digits : memref<1xi32>
    func.return %hit_88, %header_88, %meta_88, %digits_88 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_89:
    %hit_89 = arith.constant true
    %header_89 = memref.get_global @__ly_small_int_89_header : memref<2xi64>
    %meta_89 = memref.get_global @__ly_small_int_89_meta : memref<2xi64>
    %digits_89 = memref.get_global @__ly_small_int_89_digits : memref<1xi32>
    func.return %hit_89, %header_89, %meta_89, %digits_89 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_90:
    %hit_90 = arith.constant true
    %header_90 = memref.get_global @__ly_small_int_90_header : memref<2xi64>
    %meta_90 = memref.get_global @__ly_small_int_90_meta : memref<2xi64>
    %digits_90 = memref.get_global @__ly_small_int_90_digits : memref<1xi32>
    func.return %hit_90, %header_90, %meta_90, %digits_90 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_91:
    %hit_91 = arith.constant true
    %header_91 = memref.get_global @__ly_small_int_91_header : memref<2xi64>
    %meta_91 = memref.get_global @__ly_small_int_91_meta : memref<2xi64>
    %digits_91 = memref.get_global @__ly_small_int_91_digits : memref<1xi32>
    func.return %hit_91, %header_91, %meta_91, %digits_91 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_92:
    %hit_92 = arith.constant true
    %header_92 = memref.get_global @__ly_small_int_92_header : memref<2xi64>
    %meta_92 = memref.get_global @__ly_small_int_92_meta : memref<2xi64>
    %digits_92 = memref.get_global @__ly_small_int_92_digits : memref<1xi32>
    func.return %hit_92, %header_92, %meta_92, %digits_92 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_93:
    %hit_93 = arith.constant true
    %header_93 = memref.get_global @__ly_small_int_93_header : memref<2xi64>
    %meta_93 = memref.get_global @__ly_small_int_93_meta : memref<2xi64>
    %digits_93 = memref.get_global @__ly_small_int_93_digits : memref<1xi32>
    func.return %hit_93, %header_93, %meta_93, %digits_93 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_94:
    %hit_94 = arith.constant true
    %header_94 = memref.get_global @__ly_small_int_94_header : memref<2xi64>
    %meta_94 = memref.get_global @__ly_small_int_94_meta : memref<2xi64>
    %digits_94 = memref.get_global @__ly_small_int_94_digits : memref<1xi32>
    func.return %hit_94, %header_94, %meta_94, %digits_94 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_95:
    %hit_95 = arith.constant true
    %header_95 = memref.get_global @__ly_small_int_95_header : memref<2xi64>
    %meta_95 = memref.get_global @__ly_small_int_95_meta : memref<2xi64>
    %digits_95 = memref.get_global @__ly_small_int_95_digits : memref<1xi32>
    func.return %hit_95, %header_95, %meta_95, %digits_95 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_96:
    %hit_96 = arith.constant true
    %header_96 = memref.get_global @__ly_small_int_96_header : memref<2xi64>
    %meta_96 = memref.get_global @__ly_small_int_96_meta : memref<2xi64>
    %digits_96 = memref.get_global @__ly_small_int_96_digits : memref<1xi32>
    func.return %hit_96, %header_96, %meta_96, %digits_96 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_97:
    %hit_97 = arith.constant true
    %header_97 = memref.get_global @__ly_small_int_97_header : memref<2xi64>
    %meta_97 = memref.get_global @__ly_small_int_97_meta : memref<2xi64>
    %digits_97 = memref.get_global @__ly_small_int_97_digits : memref<1xi32>
    func.return %hit_97, %header_97, %meta_97, %digits_97 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_98:
    %hit_98 = arith.constant true
    %header_98 = memref.get_global @__ly_small_int_98_header : memref<2xi64>
    %meta_98 = memref.get_global @__ly_small_int_98_meta : memref<2xi64>
    %digits_98 = memref.get_global @__ly_small_int_98_digits : memref<1xi32>
    func.return %hit_98, %header_98, %meta_98, %digits_98 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_99:
    %hit_99 = arith.constant true
    %header_99 = memref.get_global @__ly_small_int_99_header : memref<2xi64>
    %meta_99 = memref.get_global @__ly_small_int_99_meta : memref<2xi64>
    %digits_99 = memref.get_global @__ly_small_int_99_digits : memref<1xi32>
    func.return %hit_99, %header_99, %meta_99, %digits_99 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_100:
    %hit_100 = arith.constant true
    %header_100 = memref.get_global @__ly_small_int_100_header : memref<2xi64>
    %meta_100 = memref.get_global @__ly_small_int_100_meta : memref<2xi64>
    %digits_100 = memref.get_global @__ly_small_int_100_digits : memref<1xi32>
    func.return %hit_100, %header_100, %meta_100, %digits_100 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_101:
    %hit_101 = arith.constant true
    %header_101 = memref.get_global @__ly_small_int_101_header : memref<2xi64>
    %meta_101 = memref.get_global @__ly_small_int_101_meta : memref<2xi64>
    %digits_101 = memref.get_global @__ly_small_int_101_digits : memref<1xi32>
    func.return %hit_101, %header_101, %meta_101, %digits_101 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_102:
    %hit_102 = arith.constant true
    %header_102 = memref.get_global @__ly_small_int_102_header : memref<2xi64>
    %meta_102 = memref.get_global @__ly_small_int_102_meta : memref<2xi64>
    %digits_102 = memref.get_global @__ly_small_int_102_digits : memref<1xi32>
    func.return %hit_102, %header_102, %meta_102, %digits_102 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_103:
    %hit_103 = arith.constant true
    %header_103 = memref.get_global @__ly_small_int_103_header : memref<2xi64>
    %meta_103 = memref.get_global @__ly_small_int_103_meta : memref<2xi64>
    %digits_103 = memref.get_global @__ly_small_int_103_digits : memref<1xi32>
    func.return %hit_103, %header_103, %meta_103, %digits_103 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_104:
    %hit_104 = arith.constant true
    %header_104 = memref.get_global @__ly_small_int_104_header : memref<2xi64>
    %meta_104 = memref.get_global @__ly_small_int_104_meta : memref<2xi64>
    %digits_104 = memref.get_global @__ly_small_int_104_digits : memref<1xi32>
    func.return %hit_104, %header_104, %meta_104, %digits_104 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_105:
    %hit_105 = arith.constant true
    %header_105 = memref.get_global @__ly_small_int_105_header : memref<2xi64>
    %meta_105 = memref.get_global @__ly_small_int_105_meta : memref<2xi64>
    %digits_105 = memref.get_global @__ly_small_int_105_digits : memref<1xi32>
    func.return %hit_105, %header_105, %meta_105, %digits_105 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_106:
    %hit_106 = arith.constant true
    %header_106 = memref.get_global @__ly_small_int_106_header : memref<2xi64>
    %meta_106 = memref.get_global @__ly_small_int_106_meta : memref<2xi64>
    %digits_106 = memref.get_global @__ly_small_int_106_digits : memref<1xi32>
    func.return %hit_106, %header_106, %meta_106, %digits_106 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_107:
    %hit_107 = arith.constant true
    %header_107 = memref.get_global @__ly_small_int_107_header : memref<2xi64>
    %meta_107 = memref.get_global @__ly_small_int_107_meta : memref<2xi64>
    %digits_107 = memref.get_global @__ly_small_int_107_digits : memref<1xi32>
    func.return %hit_107, %header_107, %meta_107, %digits_107 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_108:
    %hit_108 = arith.constant true
    %header_108 = memref.get_global @__ly_small_int_108_header : memref<2xi64>
    %meta_108 = memref.get_global @__ly_small_int_108_meta : memref<2xi64>
    %digits_108 = memref.get_global @__ly_small_int_108_digits : memref<1xi32>
    func.return %hit_108, %header_108, %meta_108, %digits_108 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_109:
    %hit_109 = arith.constant true
    %header_109 = memref.get_global @__ly_small_int_109_header : memref<2xi64>
    %meta_109 = memref.get_global @__ly_small_int_109_meta : memref<2xi64>
    %digits_109 = memref.get_global @__ly_small_int_109_digits : memref<1xi32>
    func.return %hit_109, %header_109, %meta_109, %digits_109 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_110:
    %hit_110 = arith.constant true
    %header_110 = memref.get_global @__ly_small_int_110_header : memref<2xi64>
    %meta_110 = memref.get_global @__ly_small_int_110_meta : memref<2xi64>
    %digits_110 = memref.get_global @__ly_small_int_110_digits : memref<1xi32>
    func.return %hit_110, %header_110, %meta_110, %digits_110 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_111:
    %hit_111 = arith.constant true
    %header_111 = memref.get_global @__ly_small_int_111_header : memref<2xi64>
    %meta_111 = memref.get_global @__ly_small_int_111_meta : memref<2xi64>
    %digits_111 = memref.get_global @__ly_small_int_111_digits : memref<1xi32>
    func.return %hit_111, %header_111, %meta_111, %digits_111 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_112:
    %hit_112 = arith.constant true
    %header_112 = memref.get_global @__ly_small_int_112_header : memref<2xi64>
    %meta_112 = memref.get_global @__ly_small_int_112_meta : memref<2xi64>
    %digits_112 = memref.get_global @__ly_small_int_112_digits : memref<1xi32>
    func.return %hit_112, %header_112, %meta_112, %digits_112 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_113:
    %hit_113 = arith.constant true
    %header_113 = memref.get_global @__ly_small_int_113_header : memref<2xi64>
    %meta_113 = memref.get_global @__ly_small_int_113_meta : memref<2xi64>
    %digits_113 = memref.get_global @__ly_small_int_113_digits : memref<1xi32>
    func.return %hit_113, %header_113, %meta_113, %digits_113 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_114:
    %hit_114 = arith.constant true
    %header_114 = memref.get_global @__ly_small_int_114_header : memref<2xi64>
    %meta_114 = memref.get_global @__ly_small_int_114_meta : memref<2xi64>
    %digits_114 = memref.get_global @__ly_small_int_114_digits : memref<1xi32>
    func.return %hit_114, %header_114, %meta_114, %digits_114 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_115:
    %hit_115 = arith.constant true
    %header_115 = memref.get_global @__ly_small_int_115_header : memref<2xi64>
    %meta_115 = memref.get_global @__ly_small_int_115_meta : memref<2xi64>
    %digits_115 = memref.get_global @__ly_small_int_115_digits : memref<1xi32>
    func.return %hit_115, %header_115, %meta_115, %digits_115 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_116:
    %hit_116 = arith.constant true
    %header_116 = memref.get_global @__ly_small_int_116_header : memref<2xi64>
    %meta_116 = memref.get_global @__ly_small_int_116_meta : memref<2xi64>
    %digits_116 = memref.get_global @__ly_small_int_116_digits : memref<1xi32>
    func.return %hit_116, %header_116, %meta_116, %digits_116 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_117:
    %hit_117 = arith.constant true
    %header_117 = memref.get_global @__ly_small_int_117_header : memref<2xi64>
    %meta_117 = memref.get_global @__ly_small_int_117_meta : memref<2xi64>
    %digits_117 = memref.get_global @__ly_small_int_117_digits : memref<1xi32>
    func.return %hit_117, %header_117, %meta_117, %digits_117 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_118:
    %hit_118 = arith.constant true
    %header_118 = memref.get_global @__ly_small_int_118_header : memref<2xi64>
    %meta_118 = memref.get_global @__ly_small_int_118_meta : memref<2xi64>
    %digits_118 = memref.get_global @__ly_small_int_118_digits : memref<1xi32>
    func.return %hit_118, %header_118, %meta_118, %digits_118 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_119:
    %hit_119 = arith.constant true
    %header_119 = memref.get_global @__ly_small_int_119_header : memref<2xi64>
    %meta_119 = memref.get_global @__ly_small_int_119_meta : memref<2xi64>
    %digits_119 = memref.get_global @__ly_small_int_119_digits : memref<1xi32>
    func.return %hit_119, %header_119, %meta_119, %digits_119 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_120:
    %hit_120 = arith.constant true
    %header_120 = memref.get_global @__ly_small_int_120_header : memref<2xi64>
    %meta_120 = memref.get_global @__ly_small_int_120_meta : memref<2xi64>
    %digits_120 = memref.get_global @__ly_small_int_120_digits : memref<1xi32>
    func.return %hit_120, %header_120, %meta_120, %digits_120 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_121:
    %hit_121 = arith.constant true
    %header_121 = memref.get_global @__ly_small_int_121_header : memref<2xi64>
    %meta_121 = memref.get_global @__ly_small_int_121_meta : memref<2xi64>
    %digits_121 = memref.get_global @__ly_small_int_121_digits : memref<1xi32>
    func.return %hit_121, %header_121, %meta_121, %digits_121 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_122:
    %hit_122 = arith.constant true
    %header_122 = memref.get_global @__ly_small_int_122_header : memref<2xi64>
    %meta_122 = memref.get_global @__ly_small_int_122_meta : memref<2xi64>
    %digits_122 = memref.get_global @__ly_small_int_122_digits : memref<1xi32>
    func.return %hit_122, %header_122, %meta_122, %digits_122 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_123:
    %hit_123 = arith.constant true
    %header_123 = memref.get_global @__ly_small_int_123_header : memref<2xi64>
    %meta_123 = memref.get_global @__ly_small_int_123_meta : memref<2xi64>
    %digits_123 = memref.get_global @__ly_small_int_123_digits : memref<1xi32>
    func.return %hit_123, %header_123, %meta_123, %digits_123 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_124:
    %hit_124 = arith.constant true
    %header_124 = memref.get_global @__ly_small_int_124_header : memref<2xi64>
    %meta_124 = memref.get_global @__ly_small_int_124_meta : memref<2xi64>
    %digits_124 = memref.get_global @__ly_small_int_124_digits : memref<1xi32>
    func.return %hit_124, %header_124, %meta_124, %digits_124 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_125:
    %hit_125 = arith.constant true
    %header_125 = memref.get_global @__ly_small_int_125_header : memref<2xi64>
    %meta_125 = memref.get_global @__ly_small_int_125_meta : memref<2xi64>
    %digits_125 = memref.get_global @__ly_small_int_125_digits : memref<1xi32>
    func.return %hit_125, %header_125, %meta_125, %digits_125 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_126:
    %hit_126 = arith.constant true
    %header_126 = memref.get_global @__ly_small_int_126_header : memref<2xi64>
    %meta_126 = memref.get_global @__ly_small_int_126_meta : memref<2xi64>
    %digits_126 = memref.get_global @__ly_small_int_126_digits : memref<1xi32>
    func.return %hit_126, %header_126, %meta_126, %digits_126 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_127:
    %hit_127 = arith.constant true
    %header_127 = memref.get_global @__ly_small_int_127_header : memref<2xi64>
    %meta_127 = memref.get_global @__ly_small_int_127_meta : memref<2xi64>
    %digits_127 = memref.get_global @__ly_small_int_127_digits : memref<1xi32>
    func.return %hit_127, %header_127, %meta_127, %digits_127 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_128:
    %hit_128 = arith.constant true
    %header_128 = memref.get_global @__ly_small_int_128_header : memref<2xi64>
    %meta_128 = memref.get_global @__ly_small_int_128_meta : memref<2xi64>
    %digits_128 = memref.get_global @__ly_small_int_128_digits : memref<1xi32>
    func.return %hit_128, %header_128, %meta_128, %digits_128 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_129:
    %hit_129 = arith.constant true
    %header_129 = memref.get_global @__ly_small_int_129_header : memref<2xi64>
    %meta_129 = memref.get_global @__ly_small_int_129_meta : memref<2xi64>
    %digits_129 = memref.get_global @__ly_small_int_129_digits : memref<1xi32>
    func.return %hit_129, %header_129, %meta_129, %digits_129 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_130:
    %hit_130 = arith.constant true
    %header_130 = memref.get_global @__ly_small_int_130_header : memref<2xi64>
    %meta_130 = memref.get_global @__ly_small_int_130_meta : memref<2xi64>
    %digits_130 = memref.get_global @__ly_small_int_130_digits : memref<1xi32>
    func.return %hit_130, %header_130, %meta_130, %digits_130 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_131:
    %hit_131 = arith.constant true
    %header_131 = memref.get_global @__ly_small_int_131_header : memref<2xi64>
    %meta_131 = memref.get_global @__ly_small_int_131_meta : memref<2xi64>
    %digits_131 = memref.get_global @__ly_small_int_131_digits : memref<1xi32>
    func.return %hit_131, %header_131, %meta_131, %digits_131 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_132:
    %hit_132 = arith.constant true
    %header_132 = memref.get_global @__ly_small_int_132_header : memref<2xi64>
    %meta_132 = memref.get_global @__ly_small_int_132_meta : memref<2xi64>
    %digits_132 = memref.get_global @__ly_small_int_132_digits : memref<1xi32>
    func.return %hit_132, %header_132, %meta_132, %digits_132 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_133:
    %hit_133 = arith.constant true
    %header_133 = memref.get_global @__ly_small_int_133_header : memref<2xi64>
    %meta_133 = memref.get_global @__ly_small_int_133_meta : memref<2xi64>
    %digits_133 = memref.get_global @__ly_small_int_133_digits : memref<1xi32>
    func.return %hit_133, %header_133, %meta_133, %digits_133 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_134:
    %hit_134 = arith.constant true
    %header_134 = memref.get_global @__ly_small_int_134_header : memref<2xi64>
    %meta_134 = memref.get_global @__ly_small_int_134_meta : memref<2xi64>
    %digits_134 = memref.get_global @__ly_small_int_134_digits : memref<1xi32>
    func.return %hit_134, %header_134, %meta_134, %digits_134 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_135:
    %hit_135 = arith.constant true
    %header_135 = memref.get_global @__ly_small_int_135_header : memref<2xi64>
    %meta_135 = memref.get_global @__ly_small_int_135_meta : memref<2xi64>
    %digits_135 = memref.get_global @__ly_small_int_135_digits : memref<1xi32>
    func.return %hit_135, %header_135, %meta_135, %digits_135 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_136:
    %hit_136 = arith.constant true
    %header_136 = memref.get_global @__ly_small_int_136_header : memref<2xi64>
    %meta_136 = memref.get_global @__ly_small_int_136_meta : memref<2xi64>
    %digits_136 = memref.get_global @__ly_small_int_136_digits : memref<1xi32>
    func.return %hit_136, %header_136, %meta_136, %digits_136 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_137:
    %hit_137 = arith.constant true
    %header_137 = memref.get_global @__ly_small_int_137_header : memref<2xi64>
    %meta_137 = memref.get_global @__ly_small_int_137_meta : memref<2xi64>
    %digits_137 = memref.get_global @__ly_small_int_137_digits : memref<1xi32>
    func.return %hit_137, %header_137, %meta_137, %digits_137 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_138:
    %hit_138 = arith.constant true
    %header_138 = memref.get_global @__ly_small_int_138_header : memref<2xi64>
    %meta_138 = memref.get_global @__ly_small_int_138_meta : memref<2xi64>
    %digits_138 = memref.get_global @__ly_small_int_138_digits : memref<1xi32>
    func.return %hit_138, %header_138, %meta_138, %digits_138 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_139:
    %hit_139 = arith.constant true
    %header_139 = memref.get_global @__ly_small_int_139_header : memref<2xi64>
    %meta_139 = memref.get_global @__ly_small_int_139_meta : memref<2xi64>
    %digits_139 = memref.get_global @__ly_small_int_139_digits : memref<1xi32>
    func.return %hit_139, %header_139, %meta_139, %digits_139 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_140:
    %hit_140 = arith.constant true
    %header_140 = memref.get_global @__ly_small_int_140_header : memref<2xi64>
    %meta_140 = memref.get_global @__ly_small_int_140_meta : memref<2xi64>
    %digits_140 = memref.get_global @__ly_small_int_140_digits : memref<1xi32>
    func.return %hit_140, %header_140, %meta_140, %digits_140 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_141:
    %hit_141 = arith.constant true
    %header_141 = memref.get_global @__ly_small_int_141_header : memref<2xi64>
    %meta_141 = memref.get_global @__ly_small_int_141_meta : memref<2xi64>
    %digits_141 = memref.get_global @__ly_small_int_141_digits : memref<1xi32>
    func.return %hit_141, %header_141, %meta_141, %digits_141 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_142:
    %hit_142 = arith.constant true
    %header_142 = memref.get_global @__ly_small_int_142_header : memref<2xi64>
    %meta_142 = memref.get_global @__ly_small_int_142_meta : memref<2xi64>
    %digits_142 = memref.get_global @__ly_small_int_142_digits : memref<1xi32>
    func.return %hit_142, %header_142, %meta_142, %digits_142 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_143:
    %hit_143 = arith.constant true
    %header_143 = memref.get_global @__ly_small_int_143_header : memref<2xi64>
    %meta_143 = memref.get_global @__ly_small_int_143_meta : memref<2xi64>
    %digits_143 = memref.get_global @__ly_small_int_143_digits : memref<1xi32>
    func.return %hit_143, %header_143, %meta_143, %digits_143 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_144:
    %hit_144 = arith.constant true
    %header_144 = memref.get_global @__ly_small_int_144_header : memref<2xi64>
    %meta_144 = memref.get_global @__ly_small_int_144_meta : memref<2xi64>
    %digits_144 = memref.get_global @__ly_small_int_144_digits : memref<1xi32>
    func.return %hit_144, %header_144, %meta_144, %digits_144 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_145:
    %hit_145 = arith.constant true
    %header_145 = memref.get_global @__ly_small_int_145_header : memref<2xi64>
    %meta_145 = memref.get_global @__ly_small_int_145_meta : memref<2xi64>
    %digits_145 = memref.get_global @__ly_small_int_145_digits : memref<1xi32>
    func.return %hit_145, %header_145, %meta_145, %digits_145 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_146:
    %hit_146 = arith.constant true
    %header_146 = memref.get_global @__ly_small_int_146_header : memref<2xi64>
    %meta_146 = memref.get_global @__ly_small_int_146_meta : memref<2xi64>
    %digits_146 = memref.get_global @__ly_small_int_146_digits : memref<1xi32>
    func.return %hit_146, %header_146, %meta_146, %digits_146 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_147:
    %hit_147 = arith.constant true
    %header_147 = memref.get_global @__ly_small_int_147_header : memref<2xi64>
    %meta_147 = memref.get_global @__ly_small_int_147_meta : memref<2xi64>
    %digits_147 = memref.get_global @__ly_small_int_147_digits : memref<1xi32>
    func.return %hit_147, %header_147, %meta_147, %digits_147 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_148:
    %hit_148 = arith.constant true
    %header_148 = memref.get_global @__ly_small_int_148_header : memref<2xi64>
    %meta_148 = memref.get_global @__ly_small_int_148_meta : memref<2xi64>
    %digits_148 = memref.get_global @__ly_small_int_148_digits : memref<1xi32>
    func.return %hit_148, %header_148, %meta_148, %digits_148 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_149:
    %hit_149 = arith.constant true
    %header_149 = memref.get_global @__ly_small_int_149_header : memref<2xi64>
    %meta_149 = memref.get_global @__ly_small_int_149_meta : memref<2xi64>
    %digits_149 = memref.get_global @__ly_small_int_149_digits : memref<1xi32>
    func.return %hit_149, %header_149, %meta_149, %digits_149 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_150:
    %hit_150 = arith.constant true
    %header_150 = memref.get_global @__ly_small_int_150_header : memref<2xi64>
    %meta_150 = memref.get_global @__ly_small_int_150_meta : memref<2xi64>
    %digits_150 = memref.get_global @__ly_small_int_150_digits : memref<1xi32>
    func.return %hit_150, %header_150, %meta_150, %digits_150 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_151:
    %hit_151 = arith.constant true
    %header_151 = memref.get_global @__ly_small_int_151_header : memref<2xi64>
    %meta_151 = memref.get_global @__ly_small_int_151_meta : memref<2xi64>
    %digits_151 = memref.get_global @__ly_small_int_151_digits : memref<1xi32>
    func.return %hit_151, %header_151, %meta_151, %digits_151 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_152:
    %hit_152 = arith.constant true
    %header_152 = memref.get_global @__ly_small_int_152_header : memref<2xi64>
    %meta_152 = memref.get_global @__ly_small_int_152_meta : memref<2xi64>
    %digits_152 = memref.get_global @__ly_small_int_152_digits : memref<1xi32>
    func.return %hit_152, %header_152, %meta_152, %digits_152 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_153:
    %hit_153 = arith.constant true
    %header_153 = memref.get_global @__ly_small_int_153_header : memref<2xi64>
    %meta_153 = memref.get_global @__ly_small_int_153_meta : memref<2xi64>
    %digits_153 = memref.get_global @__ly_small_int_153_digits : memref<1xi32>
    func.return %hit_153, %header_153, %meta_153, %digits_153 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_154:
    %hit_154 = arith.constant true
    %header_154 = memref.get_global @__ly_small_int_154_header : memref<2xi64>
    %meta_154 = memref.get_global @__ly_small_int_154_meta : memref<2xi64>
    %digits_154 = memref.get_global @__ly_small_int_154_digits : memref<1xi32>
    func.return %hit_154, %header_154, %meta_154, %digits_154 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_155:
    %hit_155 = arith.constant true
    %header_155 = memref.get_global @__ly_small_int_155_header : memref<2xi64>
    %meta_155 = memref.get_global @__ly_small_int_155_meta : memref<2xi64>
    %digits_155 = memref.get_global @__ly_small_int_155_digits : memref<1xi32>
    func.return %hit_155, %header_155, %meta_155, %digits_155 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_156:
    %hit_156 = arith.constant true
    %header_156 = memref.get_global @__ly_small_int_156_header : memref<2xi64>
    %meta_156 = memref.get_global @__ly_small_int_156_meta : memref<2xi64>
    %digits_156 = memref.get_global @__ly_small_int_156_digits : memref<1xi32>
    func.return %hit_156, %header_156, %meta_156, %digits_156 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_157:
    %hit_157 = arith.constant true
    %header_157 = memref.get_global @__ly_small_int_157_header : memref<2xi64>
    %meta_157 = memref.get_global @__ly_small_int_157_meta : memref<2xi64>
    %digits_157 = memref.get_global @__ly_small_int_157_digits : memref<1xi32>
    func.return %hit_157, %header_157, %meta_157, %digits_157 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_158:
    %hit_158 = arith.constant true
    %header_158 = memref.get_global @__ly_small_int_158_header : memref<2xi64>
    %meta_158 = memref.get_global @__ly_small_int_158_meta : memref<2xi64>
    %digits_158 = memref.get_global @__ly_small_int_158_digits : memref<1xi32>
    func.return %hit_158, %header_158, %meta_158, %digits_158 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_159:
    %hit_159 = arith.constant true
    %header_159 = memref.get_global @__ly_small_int_159_header : memref<2xi64>
    %meta_159 = memref.get_global @__ly_small_int_159_meta : memref<2xi64>
    %digits_159 = memref.get_global @__ly_small_int_159_digits : memref<1xi32>
    func.return %hit_159, %header_159, %meta_159, %digits_159 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_160:
    %hit_160 = arith.constant true
    %header_160 = memref.get_global @__ly_small_int_160_header : memref<2xi64>
    %meta_160 = memref.get_global @__ly_small_int_160_meta : memref<2xi64>
    %digits_160 = memref.get_global @__ly_small_int_160_digits : memref<1xi32>
    func.return %hit_160, %header_160, %meta_160, %digits_160 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_161:
    %hit_161 = arith.constant true
    %header_161 = memref.get_global @__ly_small_int_161_header : memref<2xi64>
    %meta_161 = memref.get_global @__ly_small_int_161_meta : memref<2xi64>
    %digits_161 = memref.get_global @__ly_small_int_161_digits : memref<1xi32>
    func.return %hit_161, %header_161, %meta_161, %digits_161 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_162:
    %hit_162 = arith.constant true
    %header_162 = memref.get_global @__ly_small_int_162_header : memref<2xi64>
    %meta_162 = memref.get_global @__ly_small_int_162_meta : memref<2xi64>
    %digits_162 = memref.get_global @__ly_small_int_162_digits : memref<1xi32>
    func.return %hit_162, %header_162, %meta_162, %digits_162 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_163:
    %hit_163 = arith.constant true
    %header_163 = memref.get_global @__ly_small_int_163_header : memref<2xi64>
    %meta_163 = memref.get_global @__ly_small_int_163_meta : memref<2xi64>
    %digits_163 = memref.get_global @__ly_small_int_163_digits : memref<1xi32>
    func.return %hit_163, %header_163, %meta_163, %digits_163 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_164:
    %hit_164 = arith.constant true
    %header_164 = memref.get_global @__ly_small_int_164_header : memref<2xi64>
    %meta_164 = memref.get_global @__ly_small_int_164_meta : memref<2xi64>
    %digits_164 = memref.get_global @__ly_small_int_164_digits : memref<1xi32>
    func.return %hit_164, %header_164, %meta_164, %digits_164 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_165:
    %hit_165 = arith.constant true
    %header_165 = memref.get_global @__ly_small_int_165_header : memref<2xi64>
    %meta_165 = memref.get_global @__ly_small_int_165_meta : memref<2xi64>
    %digits_165 = memref.get_global @__ly_small_int_165_digits : memref<1xi32>
    func.return %hit_165, %header_165, %meta_165, %digits_165 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_166:
    %hit_166 = arith.constant true
    %header_166 = memref.get_global @__ly_small_int_166_header : memref<2xi64>
    %meta_166 = memref.get_global @__ly_small_int_166_meta : memref<2xi64>
    %digits_166 = memref.get_global @__ly_small_int_166_digits : memref<1xi32>
    func.return %hit_166, %header_166, %meta_166, %digits_166 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_167:
    %hit_167 = arith.constant true
    %header_167 = memref.get_global @__ly_small_int_167_header : memref<2xi64>
    %meta_167 = memref.get_global @__ly_small_int_167_meta : memref<2xi64>
    %digits_167 = memref.get_global @__ly_small_int_167_digits : memref<1xi32>
    func.return %hit_167, %header_167, %meta_167, %digits_167 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_168:
    %hit_168 = arith.constant true
    %header_168 = memref.get_global @__ly_small_int_168_header : memref<2xi64>
    %meta_168 = memref.get_global @__ly_small_int_168_meta : memref<2xi64>
    %digits_168 = memref.get_global @__ly_small_int_168_digits : memref<1xi32>
    func.return %hit_168, %header_168, %meta_168, %digits_168 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_169:
    %hit_169 = arith.constant true
    %header_169 = memref.get_global @__ly_small_int_169_header : memref<2xi64>
    %meta_169 = memref.get_global @__ly_small_int_169_meta : memref<2xi64>
    %digits_169 = memref.get_global @__ly_small_int_169_digits : memref<1xi32>
    func.return %hit_169, %header_169, %meta_169, %digits_169 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_170:
    %hit_170 = arith.constant true
    %header_170 = memref.get_global @__ly_small_int_170_header : memref<2xi64>
    %meta_170 = memref.get_global @__ly_small_int_170_meta : memref<2xi64>
    %digits_170 = memref.get_global @__ly_small_int_170_digits : memref<1xi32>
    func.return %hit_170, %header_170, %meta_170, %digits_170 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_171:
    %hit_171 = arith.constant true
    %header_171 = memref.get_global @__ly_small_int_171_header : memref<2xi64>
    %meta_171 = memref.get_global @__ly_small_int_171_meta : memref<2xi64>
    %digits_171 = memref.get_global @__ly_small_int_171_digits : memref<1xi32>
    func.return %hit_171, %header_171, %meta_171, %digits_171 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_172:
    %hit_172 = arith.constant true
    %header_172 = memref.get_global @__ly_small_int_172_header : memref<2xi64>
    %meta_172 = memref.get_global @__ly_small_int_172_meta : memref<2xi64>
    %digits_172 = memref.get_global @__ly_small_int_172_digits : memref<1xi32>
    func.return %hit_172, %header_172, %meta_172, %digits_172 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_173:
    %hit_173 = arith.constant true
    %header_173 = memref.get_global @__ly_small_int_173_header : memref<2xi64>
    %meta_173 = memref.get_global @__ly_small_int_173_meta : memref<2xi64>
    %digits_173 = memref.get_global @__ly_small_int_173_digits : memref<1xi32>
    func.return %hit_173, %header_173, %meta_173, %digits_173 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_174:
    %hit_174 = arith.constant true
    %header_174 = memref.get_global @__ly_small_int_174_header : memref<2xi64>
    %meta_174 = memref.get_global @__ly_small_int_174_meta : memref<2xi64>
    %digits_174 = memref.get_global @__ly_small_int_174_digits : memref<1xi32>
    func.return %hit_174, %header_174, %meta_174, %digits_174 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_175:
    %hit_175 = arith.constant true
    %header_175 = memref.get_global @__ly_small_int_175_header : memref<2xi64>
    %meta_175 = memref.get_global @__ly_small_int_175_meta : memref<2xi64>
    %digits_175 = memref.get_global @__ly_small_int_175_digits : memref<1xi32>
    func.return %hit_175, %header_175, %meta_175, %digits_175 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_176:
    %hit_176 = arith.constant true
    %header_176 = memref.get_global @__ly_small_int_176_header : memref<2xi64>
    %meta_176 = memref.get_global @__ly_small_int_176_meta : memref<2xi64>
    %digits_176 = memref.get_global @__ly_small_int_176_digits : memref<1xi32>
    func.return %hit_176, %header_176, %meta_176, %digits_176 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_177:
    %hit_177 = arith.constant true
    %header_177 = memref.get_global @__ly_small_int_177_header : memref<2xi64>
    %meta_177 = memref.get_global @__ly_small_int_177_meta : memref<2xi64>
    %digits_177 = memref.get_global @__ly_small_int_177_digits : memref<1xi32>
    func.return %hit_177, %header_177, %meta_177, %digits_177 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_178:
    %hit_178 = arith.constant true
    %header_178 = memref.get_global @__ly_small_int_178_header : memref<2xi64>
    %meta_178 = memref.get_global @__ly_small_int_178_meta : memref<2xi64>
    %digits_178 = memref.get_global @__ly_small_int_178_digits : memref<1xi32>
    func.return %hit_178, %header_178, %meta_178, %digits_178 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_179:
    %hit_179 = arith.constant true
    %header_179 = memref.get_global @__ly_small_int_179_header : memref<2xi64>
    %meta_179 = memref.get_global @__ly_small_int_179_meta : memref<2xi64>
    %digits_179 = memref.get_global @__ly_small_int_179_digits : memref<1xi32>
    func.return %hit_179, %header_179, %meta_179, %digits_179 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_180:
    %hit_180 = arith.constant true
    %header_180 = memref.get_global @__ly_small_int_180_header : memref<2xi64>
    %meta_180 = memref.get_global @__ly_small_int_180_meta : memref<2xi64>
    %digits_180 = memref.get_global @__ly_small_int_180_digits : memref<1xi32>
    func.return %hit_180, %header_180, %meta_180, %digits_180 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_181:
    %hit_181 = arith.constant true
    %header_181 = memref.get_global @__ly_small_int_181_header : memref<2xi64>
    %meta_181 = memref.get_global @__ly_small_int_181_meta : memref<2xi64>
    %digits_181 = memref.get_global @__ly_small_int_181_digits : memref<1xi32>
    func.return %hit_181, %header_181, %meta_181, %digits_181 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_182:
    %hit_182 = arith.constant true
    %header_182 = memref.get_global @__ly_small_int_182_header : memref<2xi64>
    %meta_182 = memref.get_global @__ly_small_int_182_meta : memref<2xi64>
    %digits_182 = memref.get_global @__ly_small_int_182_digits : memref<1xi32>
    func.return %hit_182, %header_182, %meta_182, %digits_182 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_183:
    %hit_183 = arith.constant true
    %header_183 = memref.get_global @__ly_small_int_183_header : memref<2xi64>
    %meta_183 = memref.get_global @__ly_small_int_183_meta : memref<2xi64>
    %digits_183 = memref.get_global @__ly_small_int_183_digits : memref<1xi32>
    func.return %hit_183, %header_183, %meta_183, %digits_183 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_184:
    %hit_184 = arith.constant true
    %header_184 = memref.get_global @__ly_small_int_184_header : memref<2xi64>
    %meta_184 = memref.get_global @__ly_small_int_184_meta : memref<2xi64>
    %digits_184 = memref.get_global @__ly_small_int_184_digits : memref<1xi32>
    func.return %hit_184, %header_184, %meta_184, %digits_184 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_185:
    %hit_185 = arith.constant true
    %header_185 = memref.get_global @__ly_small_int_185_header : memref<2xi64>
    %meta_185 = memref.get_global @__ly_small_int_185_meta : memref<2xi64>
    %digits_185 = memref.get_global @__ly_small_int_185_digits : memref<1xi32>
    func.return %hit_185, %header_185, %meta_185, %digits_185 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_186:
    %hit_186 = arith.constant true
    %header_186 = memref.get_global @__ly_small_int_186_header : memref<2xi64>
    %meta_186 = memref.get_global @__ly_small_int_186_meta : memref<2xi64>
    %digits_186 = memref.get_global @__ly_small_int_186_digits : memref<1xi32>
    func.return %hit_186, %header_186, %meta_186, %digits_186 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_187:
    %hit_187 = arith.constant true
    %header_187 = memref.get_global @__ly_small_int_187_header : memref<2xi64>
    %meta_187 = memref.get_global @__ly_small_int_187_meta : memref<2xi64>
    %digits_187 = memref.get_global @__ly_small_int_187_digits : memref<1xi32>
    func.return %hit_187, %header_187, %meta_187, %digits_187 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_188:
    %hit_188 = arith.constant true
    %header_188 = memref.get_global @__ly_small_int_188_header : memref<2xi64>
    %meta_188 = memref.get_global @__ly_small_int_188_meta : memref<2xi64>
    %digits_188 = memref.get_global @__ly_small_int_188_digits : memref<1xi32>
    func.return %hit_188, %header_188, %meta_188, %digits_188 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_189:
    %hit_189 = arith.constant true
    %header_189 = memref.get_global @__ly_small_int_189_header : memref<2xi64>
    %meta_189 = memref.get_global @__ly_small_int_189_meta : memref<2xi64>
    %digits_189 = memref.get_global @__ly_small_int_189_digits : memref<1xi32>
    func.return %hit_189, %header_189, %meta_189, %digits_189 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_190:
    %hit_190 = arith.constant true
    %header_190 = memref.get_global @__ly_small_int_190_header : memref<2xi64>
    %meta_190 = memref.get_global @__ly_small_int_190_meta : memref<2xi64>
    %digits_190 = memref.get_global @__ly_small_int_190_digits : memref<1xi32>
    func.return %hit_190, %header_190, %meta_190, %digits_190 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_191:
    %hit_191 = arith.constant true
    %header_191 = memref.get_global @__ly_small_int_191_header : memref<2xi64>
    %meta_191 = memref.get_global @__ly_small_int_191_meta : memref<2xi64>
    %digits_191 = memref.get_global @__ly_small_int_191_digits : memref<1xi32>
    func.return %hit_191, %header_191, %meta_191, %digits_191 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_192:
    %hit_192 = arith.constant true
    %header_192 = memref.get_global @__ly_small_int_192_header : memref<2xi64>
    %meta_192 = memref.get_global @__ly_small_int_192_meta : memref<2xi64>
    %digits_192 = memref.get_global @__ly_small_int_192_digits : memref<1xi32>
    func.return %hit_192, %header_192, %meta_192, %digits_192 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_193:
    %hit_193 = arith.constant true
    %header_193 = memref.get_global @__ly_small_int_193_header : memref<2xi64>
    %meta_193 = memref.get_global @__ly_small_int_193_meta : memref<2xi64>
    %digits_193 = memref.get_global @__ly_small_int_193_digits : memref<1xi32>
    func.return %hit_193, %header_193, %meta_193, %digits_193 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_194:
    %hit_194 = arith.constant true
    %header_194 = memref.get_global @__ly_small_int_194_header : memref<2xi64>
    %meta_194 = memref.get_global @__ly_small_int_194_meta : memref<2xi64>
    %digits_194 = memref.get_global @__ly_small_int_194_digits : memref<1xi32>
    func.return %hit_194, %header_194, %meta_194, %digits_194 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_195:
    %hit_195 = arith.constant true
    %header_195 = memref.get_global @__ly_small_int_195_header : memref<2xi64>
    %meta_195 = memref.get_global @__ly_small_int_195_meta : memref<2xi64>
    %digits_195 = memref.get_global @__ly_small_int_195_digits : memref<1xi32>
    func.return %hit_195, %header_195, %meta_195, %digits_195 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_196:
    %hit_196 = arith.constant true
    %header_196 = memref.get_global @__ly_small_int_196_header : memref<2xi64>
    %meta_196 = memref.get_global @__ly_small_int_196_meta : memref<2xi64>
    %digits_196 = memref.get_global @__ly_small_int_196_digits : memref<1xi32>
    func.return %hit_196, %header_196, %meta_196, %digits_196 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_197:
    %hit_197 = arith.constant true
    %header_197 = memref.get_global @__ly_small_int_197_header : memref<2xi64>
    %meta_197 = memref.get_global @__ly_small_int_197_meta : memref<2xi64>
    %digits_197 = memref.get_global @__ly_small_int_197_digits : memref<1xi32>
    func.return %hit_197, %header_197, %meta_197, %digits_197 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_198:
    %hit_198 = arith.constant true
    %header_198 = memref.get_global @__ly_small_int_198_header : memref<2xi64>
    %meta_198 = memref.get_global @__ly_small_int_198_meta : memref<2xi64>
    %digits_198 = memref.get_global @__ly_small_int_198_digits : memref<1xi32>
    func.return %hit_198, %header_198, %meta_198, %digits_198 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_199:
    %hit_199 = arith.constant true
    %header_199 = memref.get_global @__ly_small_int_199_header : memref<2xi64>
    %meta_199 = memref.get_global @__ly_small_int_199_meta : memref<2xi64>
    %digits_199 = memref.get_global @__ly_small_int_199_digits : memref<1xi32>
    func.return %hit_199, %header_199, %meta_199, %digits_199 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_200:
    %hit_200 = arith.constant true
    %header_200 = memref.get_global @__ly_small_int_200_header : memref<2xi64>
    %meta_200 = memref.get_global @__ly_small_int_200_meta : memref<2xi64>
    %digits_200 = memref.get_global @__ly_small_int_200_digits : memref<1xi32>
    func.return %hit_200, %header_200, %meta_200, %digits_200 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_201:
    %hit_201 = arith.constant true
    %header_201 = memref.get_global @__ly_small_int_201_header : memref<2xi64>
    %meta_201 = memref.get_global @__ly_small_int_201_meta : memref<2xi64>
    %digits_201 = memref.get_global @__ly_small_int_201_digits : memref<1xi32>
    func.return %hit_201, %header_201, %meta_201, %digits_201 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_202:
    %hit_202 = arith.constant true
    %header_202 = memref.get_global @__ly_small_int_202_header : memref<2xi64>
    %meta_202 = memref.get_global @__ly_small_int_202_meta : memref<2xi64>
    %digits_202 = memref.get_global @__ly_small_int_202_digits : memref<1xi32>
    func.return %hit_202, %header_202, %meta_202, %digits_202 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_203:
    %hit_203 = arith.constant true
    %header_203 = memref.get_global @__ly_small_int_203_header : memref<2xi64>
    %meta_203 = memref.get_global @__ly_small_int_203_meta : memref<2xi64>
    %digits_203 = memref.get_global @__ly_small_int_203_digits : memref<1xi32>
    func.return %hit_203, %header_203, %meta_203, %digits_203 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_204:
    %hit_204 = arith.constant true
    %header_204 = memref.get_global @__ly_small_int_204_header : memref<2xi64>
    %meta_204 = memref.get_global @__ly_small_int_204_meta : memref<2xi64>
    %digits_204 = memref.get_global @__ly_small_int_204_digits : memref<1xi32>
    func.return %hit_204, %header_204, %meta_204, %digits_204 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_205:
    %hit_205 = arith.constant true
    %header_205 = memref.get_global @__ly_small_int_205_header : memref<2xi64>
    %meta_205 = memref.get_global @__ly_small_int_205_meta : memref<2xi64>
    %digits_205 = memref.get_global @__ly_small_int_205_digits : memref<1xi32>
    func.return %hit_205, %header_205, %meta_205, %digits_205 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_206:
    %hit_206 = arith.constant true
    %header_206 = memref.get_global @__ly_small_int_206_header : memref<2xi64>
    %meta_206 = memref.get_global @__ly_small_int_206_meta : memref<2xi64>
    %digits_206 = memref.get_global @__ly_small_int_206_digits : memref<1xi32>
    func.return %hit_206, %header_206, %meta_206, %digits_206 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_207:
    %hit_207 = arith.constant true
    %header_207 = memref.get_global @__ly_small_int_207_header : memref<2xi64>
    %meta_207 = memref.get_global @__ly_small_int_207_meta : memref<2xi64>
    %digits_207 = memref.get_global @__ly_small_int_207_digits : memref<1xi32>
    func.return %hit_207, %header_207, %meta_207, %digits_207 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_208:
    %hit_208 = arith.constant true
    %header_208 = memref.get_global @__ly_small_int_208_header : memref<2xi64>
    %meta_208 = memref.get_global @__ly_small_int_208_meta : memref<2xi64>
    %digits_208 = memref.get_global @__ly_small_int_208_digits : memref<1xi32>
    func.return %hit_208, %header_208, %meta_208, %digits_208 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_209:
    %hit_209 = arith.constant true
    %header_209 = memref.get_global @__ly_small_int_209_header : memref<2xi64>
    %meta_209 = memref.get_global @__ly_small_int_209_meta : memref<2xi64>
    %digits_209 = memref.get_global @__ly_small_int_209_digits : memref<1xi32>
    func.return %hit_209, %header_209, %meta_209, %digits_209 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_210:
    %hit_210 = arith.constant true
    %header_210 = memref.get_global @__ly_small_int_210_header : memref<2xi64>
    %meta_210 = memref.get_global @__ly_small_int_210_meta : memref<2xi64>
    %digits_210 = memref.get_global @__ly_small_int_210_digits : memref<1xi32>
    func.return %hit_210, %header_210, %meta_210, %digits_210 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_211:
    %hit_211 = arith.constant true
    %header_211 = memref.get_global @__ly_small_int_211_header : memref<2xi64>
    %meta_211 = memref.get_global @__ly_small_int_211_meta : memref<2xi64>
    %digits_211 = memref.get_global @__ly_small_int_211_digits : memref<1xi32>
    func.return %hit_211, %header_211, %meta_211, %digits_211 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_212:
    %hit_212 = arith.constant true
    %header_212 = memref.get_global @__ly_small_int_212_header : memref<2xi64>
    %meta_212 = memref.get_global @__ly_small_int_212_meta : memref<2xi64>
    %digits_212 = memref.get_global @__ly_small_int_212_digits : memref<1xi32>
    func.return %hit_212, %header_212, %meta_212, %digits_212 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_213:
    %hit_213 = arith.constant true
    %header_213 = memref.get_global @__ly_small_int_213_header : memref<2xi64>
    %meta_213 = memref.get_global @__ly_small_int_213_meta : memref<2xi64>
    %digits_213 = memref.get_global @__ly_small_int_213_digits : memref<1xi32>
    func.return %hit_213, %header_213, %meta_213, %digits_213 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_214:
    %hit_214 = arith.constant true
    %header_214 = memref.get_global @__ly_small_int_214_header : memref<2xi64>
    %meta_214 = memref.get_global @__ly_small_int_214_meta : memref<2xi64>
    %digits_214 = memref.get_global @__ly_small_int_214_digits : memref<1xi32>
    func.return %hit_214, %header_214, %meta_214, %digits_214 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_215:
    %hit_215 = arith.constant true
    %header_215 = memref.get_global @__ly_small_int_215_header : memref<2xi64>
    %meta_215 = memref.get_global @__ly_small_int_215_meta : memref<2xi64>
    %digits_215 = memref.get_global @__ly_small_int_215_digits : memref<1xi32>
    func.return %hit_215, %header_215, %meta_215, %digits_215 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_216:
    %hit_216 = arith.constant true
    %header_216 = memref.get_global @__ly_small_int_216_header : memref<2xi64>
    %meta_216 = memref.get_global @__ly_small_int_216_meta : memref<2xi64>
    %digits_216 = memref.get_global @__ly_small_int_216_digits : memref<1xi32>
    func.return %hit_216, %header_216, %meta_216, %digits_216 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_217:
    %hit_217 = arith.constant true
    %header_217 = memref.get_global @__ly_small_int_217_header : memref<2xi64>
    %meta_217 = memref.get_global @__ly_small_int_217_meta : memref<2xi64>
    %digits_217 = memref.get_global @__ly_small_int_217_digits : memref<1xi32>
    func.return %hit_217, %header_217, %meta_217, %digits_217 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_218:
    %hit_218 = arith.constant true
    %header_218 = memref.get_global @__ly_small_int_218_header : memref<2xi64>
    %meta_218 = memref.get_global @__ly_small_int_218_meta : memref<2xi64>
    %digits_218 = memref.get_global @__ly_small_int_218_digits : memref<1xi32>
    func.return %hit_218, %header_218, %meta_218, %digits_218 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_219:
    %hit_219 = arith.constant true
    %header_219 = memref.get_global @__ly_small_int_219_header : memref<2xi64>
    %meta_219 = memref.get_global @__ly_small_int_219_meta : memref<2xi64>
    %digits_219 = memref.get_global @__ly_small_int_219_digits : memref<1xi32>
    func.return %hit_219, %header_219, %meta_219, %digits_219 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_220:
    %hit_220 = arith.constant true
    %header_220 = memref.get_global @__ly_small_int_220_header : memref<2xi64>
    %meta_220 = memref.get_global @__ly_small_int_220_meta : memref<2xi64>
    %digits_220 = memref.get_global @__ly_small_int_220_digits : memref<1xi32>
    func.return %hit_220, %header_220, %meta_220, %digits_220 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_221:
    %hit_221 = arith.constant true
    %header_221 = memref.get_global @__ly_small_int_221_header : memref<2xi64>
    %meta_221 = memref.get_global @__ly_small_int_221_meta : memref<2xi64>
    %digits_221 = memref.get_global @__ly_small_int_221_digits : memref<1xi32>
    func.return %hit_221, %header_221, %meta_221, %digits_221 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_222:
    %hit_222 = arith.constant true
    %header_222 = memref.get_global @__ly_small_int_222_header : memref<2xi64>
    %meta_222 = memref.get_global @__ly_small_int_222_meta : memref<2xi64>
    %digits_222 = memref.get_global @__ly_small_int_222_digits : memref<1xi32>
    func.return %hit_222, %header_222, %meta_222, %digits_222 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_223:
    %hit_223 = arith.constant true
    %header_223 = memref.get_global @__ly_small_int_223_header : memref<2xi64>
    %meta_223 = memref.get_global @__ly_small_int_223_meta : memref<2xi64>
    %digits_223 = memref.get_global @__ly_small_int_223_digits : memref<1xi32>
    func.return %hit_223, %header_223, %meta_223, %digits_223 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_224:
    %hit_224 = arith.constant true
    %header_224 = memref.get_global @__ly_small_int_224_header : memref<2xi64>
    %meta_224 = memref.get_global @__ly_small_int_224_meta : memref<2xi64>
    %digits_224 = memref.get_global @__ly_small_int_224_digits : memref<1xi32>
    func.return %hit_224, %header_224, %meta_224, %digits_224 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_225:
    %hit_225 = arith.constant true
    %header_225 = memref.get_global @__ly_small_int_225_header : memref<2xi64>
    %meta_225 = memref.get_global @__ly_small_int_225_meta : memref<2xi64>
    %digits_225 = memref.get_global @__ly_small_int_225_digits : memref<1xi32>
    func.return %hit_225, %header_225, %meta_225, %digits_225 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_226:
    %hit_226 = arith.constant true
    %header_226 = memref.get_global @__ly_small_int_226_header : memref<2xi64>
    %meta_226 = memref.get_global @__ly_small_int_226_meta : memref<2xi64>
    %digits_226 = memref.get_global @__ly_small_int_226_digits : memref<1xi32>
    func.return %hit_226, %header_226, %meta_226, %digits_226 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_227:
    %hit_227 = arith.constant true
    %header_227 = memref.get_global @__ly_small_int_227_header : memref<2xi64>
    %meta_227 = memref.get_global @__ly_small_int_227_meta : memref<2xi64>
    %digits_227 = memref.get_global @__ly_small_int_227_digits : memref<1xi32>
    func.return %hit_227, %header_227, %meta_227, %digits_227 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_228:
    %hit_228 = arith.constant true
    %header_228 = memref.get_global @__ly_small_int_228_header : memref<2xi64>
    %meta_228 = memref.get_global @__ly_small_int_228_meta : memref<2xi64>
    %digits_228 = memref.get_global @__ly_small_int_228_digits : memref<1xi32>
    func.return %hit_228, %header_228, %meta_228, %digits_228 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_229:
    %hit_229 = arith.constant true
    %header_229 = memref.get_global @__ly_small_int_229_header : memref<2xi64>
    %meta_229 = memref.get_global @__ly_small_int_229_meta : memref<2xi64>
    %digits_229 = memref.get_global @__ly_small_int_229_digits : memref<1xi32>
    func.return %hit_229, %header_229, %meta_229, %digits_229 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_230:
    %hit_230 = arith.constant true
    %header_230 = memref.get_global @__ly_small_int_230_header : memref<2xi64>
    %meta_230 = memref.get_global @__ly_small_int_230_meta : memref<2xi64>
    %digits_230 = memref.get_global @__ly_small_int_230_digits : memref<1xi32>
    func.return %hit_230, %header_230, %meta_230, %digits_230 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_231:
    %hit_231 = arith.constant true
    %header_231 = memref.get_global @__ly_small_int_231_header : memref<2xi64>
    %meta_231 = memref.get_global @__ly_small_int_231_meta : memref<2xi64>
    %digits_231 = memref.get_global @__ly_small_int_231_digits : memref<1xi32>
    func.return %hit_231, %header_231, %meta_231, %digits_231 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_232:
    %hit_232 = arith.constant true
    %header_232 = memref.get_global @__ly_small_int_232_header : memref<2xi64>
    %meta_232 = memref.get_global @__ly_small_int_232_meta : memref<2xi64>
    %digits_232 = memref.get_global @__ly_small_int_232_digits : memref<1xi32>
    func.return %hit_232, %header_232, %meta_232, %digits_232 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_233:
    %hit_233 = arith.constant true
    %header_233 = memref.get_global @__ly_small_int_233_header : memref<2xi64>
    %meta_233 = memref.get_global @__ly_small_int_233_meta : memref<2xi64>
    %digits_233 = memref.get_global @__ly_small_int_233_digits : memref<1xi32>
    func.return %hit_233, %header_233, %meta_233, %digits_233 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_234:
    %hit_234 = arith.constant true
    %header_234 = memref.get_global @__ly_small_int_234_header : memref<2xi64>
    %meta_234 = memref.get_global @__ly_small_int_234_meta : memref<2xi64>
    %digits_234 = memref.get_global @__ly_small_int_234_digits : memref<1xi32>
    func.return %hit_234, %header_234, %meta_234, %digits_234 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_235:
    %hit_235 = arith.constant true
    %header_235 = memref.get_global @__ly_small_int_235_header : memref<2xi64>
    %meta_235 = memref.get_global @__ly_small_int_235_meta : memref<2xi64>
    %digits_235 = memref.get_global @__ly_small_int_235_digits : memref<1xi32>
    func.return %hit_235, %header_235, %meta_235, %digits_235 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_236:
    %hit_236 = arith.constant true
    %header_236 = memref.get_global @__ly_small_int_236_header : memref<2xi64>
    %meta_236 = memref.get_global @__ly_small_int_236_meta : memref<2xi64>
    %digits_236 = memref.get_global @__ly_small_int_236_digits : memref<1xi32>
    func.return %hit_236, %header_236, %meta_236, %digits_236 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_237:
    %hit_237 = arith.constant true
    %header_237 = memref.get_global @__ly_small_int_237_header : memref<2xi64>
    %meta_237 = memref.get_global @__ly_small_int_237_meta : memref<2xi64>
    %digits_237 = memref.get_global @__ly_small_int_237_digits : memref<1xi32>
    func.return %hit_237, %header_237, %meta_237, %digits_237 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_238:
    %hit_238 = arith.constant true
    %header_238 = memref.get_global @__ly_small_int_238_header : memref<2xi64>
    %meta_238 = memref.get_global @__ly_small_int_238_meta : memref<2xi64>
    %digits_238 = memref.get_global @__ly_small_int_238_digits : memref<1xi32>
    func.return %hit_238, %header_238, %meta_238, %digits_238 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_239:
    %hit_239 = arith.constant true
    %header_239 = memref.get_global @__ly_small_int_239_header : memref<2xi64>
    %meta_239 = memref.get_global @__ly_small_int_239_meta : memref<2xi64>
    %digits_239 = memref.get_global @__ly_small_int_239_digits : memref<1xi32>
    func.return %hit_239, %header_239, %meta_239, %digits_239 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_240:
    %hit_240 = arith.constant true
    %header_240 = memref.get_global @__ly_small_int_240_header : memref<2xi64>
    %meta_240 = memref.get_global @__ly_small_int_240_meta : memref<2xi64>
    %digits_240 = memref.get_global @__ly_small_int_240_digits : memref<1xi32>
    func.return %hit_240, %header_240, %meta_240, %digits_240 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_241:
    %hit_241 = arith.constant true
    %header_241 = memref.get_global @__ly_small_int_241_header : memref<2xi64>
    %meta_241 = memref.get_global @__ly_small_int_241_meta : memref<2xi64>
    %digits_241 = memref.get_global @__ly_small_int_241_digits : memref<1xi32>
    func.return %hit_241, %header_241, %meta_241, %digits_241 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_242:
    %hit_242 = arith.constant true
    %header_242 = memref.get_global @__ly_small_int_242_header : memref<2xi64>
    %meta_242 = memref.get_global @__ly_small_int_242_meta : memref<2xi64>
    %digits_242 = memref.get_global @__ly_small_int_242_digits : memref<1xi32>
    func.return %hit_242, %header_242, %meta_242, %digits_242 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_243:
    %hit_243 = arith.constant true
    %header_243 = memref.get_global @__ly_small_int_243_header : memref<2xi64>
    %meta_243 = memref.get_global @__ly_small_int_243_meta : memref<2xi64>
    %digits_243 = memref.get_global @__ly_small_int_243_digits : memref<1xi32>
    func.return %hit_243, %header_243, %meta_243, %digits_243 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_244:
    %hit_244 = arith.constant true
    %header_244 = memref.get_global @__ly_small_int_244_header : memref<2xi64>
    %meta_244 = memref.get_global @__ly_small_int_244_meta : memref<2xi64>
    %digits_244 = memref.get_global @__ly_small_int_244_digits : memref<1xi32>
    func.return %hit_244, %header_244, %meta_244, %digits_244 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_245:
    %hit_245 = arith.constant true
    %header_245 = memref.get_global @__ly_small_int_245_header : memref<2xi64>
    %meta_245 = memref.get_global @__ly_small_int_245_meta : memref<2xi64>
    %digits_245 = memref.get_global @__ly_small_int_245_digits : memref<1xi32>
    func.return %hit_245, %header_245, %meta_245, %digits_245 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_246:
    %hit_246 = arith.constant true
    %header_246 = memref.get_global @__ly_small_int_246_header : memref<2xi64>
    %meta_246 = memref.get_global @__ly_small_int_246_meta : memref<2xi64>
    %digits_246 = memref.get_global @__ly_small_int_246_digits : memref<1xi32>
    func.return %hit_246, %header_246, %meta_246, %digits_246 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_247:
    %hit_247 = arith.constant true
    %header_247 = memref.get_global @__ly_small_int_247_header : memref<2xi64>
    %meta_247 = memref.get_global @__ly_small_int_247_meta : memref<2xi64>
    %digits_247 = memref.get_global @__ly_small_int_247_digits : memref<1xi32>
    func.return %hit_247, %header_247, %meta_247, %digits_247 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_248:
    %hit_248 = arith.constant true
    %header_248 = memref.get_global @__ly_small_int_248_header : memref<2xi64>
    %meta_248 = memref.get_global @__ly_small_int_248_meta : memref<2xi64>
    %digits_248 = memref.get_global @__ly_small_int_248_digits : memref<1xi32>
    func.return %hit_248, %header_248, %meta_248, %digits_248 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_249:
    %hit_249 = arith.constant true
    %header_249 = memref.get_global @__ly_small_int_249_header : memref<2xi64>
    %meta_249 = memref.get_global @__ly_small_int_249_meta : memref<2xi64>
    %digits_249 = memref.get_global @__ly_small_int_249_digits : memref<1xi32>
    func.return %hit_249, %header_249, %meta_249, %digits_249 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_250:
    %hit_250 = arith.constant true
    %header_250 = memref.get_global @__ly_small_int_250_header : memref<2xi64>
    %meta_250 = memref.get_global @__ly_small_int_250_meta : memref<2xi64>
    %digits_250 = memref.get_global @__ly_small_int_250_digits : memref<1xi32>
    func.return %hit_250, %header_250, %meta_250, %digits_250 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_251:
    %hit_251 = arith.constant true
    %header_251 = memref.get_global @__ly_small_int_251_header : memref<2xi64>
    %meta_251 = memref.get_global @__ly_small_int_251_meta : memref<2xi64>
    %digits_251 = memref.get_global @__ly_small_int_251_digits : memref<1xi32>
    func.return %hit_251, %header_251, %meta_251, %digits_251 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_252:
    %hit_252 = arith.constant true
    %header_252 = memref.get_global @__ly_small_int_252_header : memref<2xi64>
    %meta_252 = memref.get_global @__ly_small_int_252_meta : memref<2xi64>
    %digits_252 = memref.get_global @__ly_small_int_252_digits : memref<1xi32>
    func.return %hit_252, %header_252, %meta_252, %digits_252 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_253:
    %hit_253 = arith.constant true
    %header_253 = memref.get_global @__ly_small_int_253_header : memref<2xi64>
    %meta_253 = memref.get_global @__ly_small_int_253_meta : memref<2xi64>
    %digits_253 = memref.get_global @__ly_small_int_253_digits : memref<1xi32>
    func.return %hit_253, %header_253, %meta_253, %digits_253 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_254:
    %hit_254 = arith.constant true
    %header_254 = memref.get_global @__ly_small_int_254_header : memref<2xi64>
    %meta_254 = memref.get_global @__ly_small_int_254_meta : memref<2xi64>
    %digits_254 = memref.get_global @__ly_small_int_254_digits : memref<1xi32>
    func.return %hit_254, %header_254, %meta_254, %digits_254 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_255:
    %hit_255 = arith.constant true
    %header_255 = memref.get_global @__ly_small_int_255_header : memref<2xi64>
    %meta_255 = memref.get_global @__ly_small_int_255_meta : memref<2xi64>
    %digits_255 = memref.get_global @__ly_small_int_255_digits : memref<1xi32>
    func.return %hit_255, %header_255, %meta_255, %digits_255 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^small_256:
    %hit_256 = arith.constant true
    %header_256 = memref.get_global @__ly_small_int_256_header : memref<2xi64>
    %meta_256 = memref.get_global @__ly_small_int_256_meta : memref<2xi64>
    %digits_256 = memref.get_global @__ly_small_int_256_digits : memref<1xi32>
    func.return %hit_256, %header_256, %meta_256, %digits_256 : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>

  ^default:
    %miss = arith.constant false
    %header = memref.get_global @__ly_small_int_z_header : memref<2xi64>
    %meta = memref.get_global @__ly_small_int_z_meta : memref<2xi64>
    %digits = memref.get_global @__ly_small_int_z_digits : memref<1xi32>
    func.return %miss, %header, %meta, %digits : i1, memref<2xi64>, memref<2xi64>, memref<1xi32>
  }
}
