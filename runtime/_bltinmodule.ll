; ModuleID = 'Python/bltinmodule.c'
source_filename = "Python/bltinmodule.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

%struct._typeobject = type { %struct.PyVarObject, ptr, i64, i64, ptr, i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr, ptr, ptr, ptr, i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, ptr, i8, i16 }
%struct.PyVarObject = type { %struct._object, i64 }
%struct._object = type { %union.anon, ptr }
%union.anon = type { i64 }
%struct.PyMethodDef = type { ptr, ptr, i32, ptr }
%struct.PyModuleDef = type { %struct.PyModuleDef_Base, ptr, ptr, i64, ptr, ptr, ptr, ptr, ptr }
%struct.PyModuleDef_Base = type { %struct._object, ptr, i64, ptr }
%struct._longobject = type { %struct._object, %struct._PyLongValue }
%struct._PyLongValue = type { i64, [1 x i32] }
%struct.pyruntimestate = type { %struct._Py_DebugOffsets, i32, i32, i32, i32, i32, ptr, i64, %struct.pyinterpreters, i64, ptr, %struct._PyXI_global_state_t, %struct._pymem_allocators, %struct._obmalloc_global_state, %struct.pyhash_runtime_state, %struct._pythread_runtime_state, %struct._signals_runtime_state, %struct._Py_tss_t, %struct._Py_tss_t, %struct.PyWideStringList, %struct._parser_runtime_state, %struct._atexit_runtime_state, %struct._import_runtime_state, %struct._ceval_runtime_state, %struct._gilstate_runtime_state, %struct._getargs_runtime_state, %struct._fileutils_state, %struct._faulthandler_runtime_state, %struct._tracemalloc_runtime_state, %struct._reftracer_runtime_state, %struct._PyRWMutex, %struct._stoptheworld_state, %struct.PyPreConfig, ptr, ptr, %struct.anon.44, %struct._py_object_runtime_state, %struct._Py_float_runtime_state, %struct._Py_unicode_runtime_state, %struct._types_runtime_state, %struct._Py_cached_objects, %struct._Py_static_objects, %struct._is }
%struct._Py_DebugOffsets = type { [8 x i8], i64, i64, %struct._runtime_state, %struct._interpreter_state, %struct._thread_state, %struct._interpreter_frame, %struct._code_object, %struct._pyobject, %struct._type_object, %struct._tuple_object, %struct._list_object, %struct._dict_object, %struct._float_object, %struct._long_object, %struct._bytes_object, %struct._unicode_object, %struct._gc }
%struct._runtime_state = type { i64, i64, i64 }
%struct._interpreter_state = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 }
%struct._thread_state = type { i64, i64, i64, i64, i64, i64, i64, i64, i64 }
%struct._interpreter_frame = type { i64, i64, i64, i64, i64, i64 }
%struct._code_object = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 }
%struct._pyobject = type { i64, i64 }
%struct._type_object = type { i64, i64, i64, i64 }
%struct._tuple_object = type { i64, i64, i64 }
%struct._list_object = type { i64, i64, i64 }
%struct._dict_object = type { i64, i64, i64 }
%struct._float_object = type { i64, i64 }
%struct._long_object = type { i64, i64, i64 }
%struct._bytes_object = type { i64, i64, i64 }
%struct._unicode_object = type { i64, i64, i64, i64 }
%struct._gc = type { i64, i64 }
%struct.pyinterpreters = type { %struct.PyMutex, ptr, ptr, i64 }
%struct.PyMutex = type { i8 }
%struct._PyXI_global_state_t = type { %struct._xid_lookup_state }
%struct._xid_lookup_state = type { %struct._PyXIData_registry_t }
%struct._PyXIData_registry_t = type { i32, i32, %struct.PyMutex, ptr }
%struct._pymem_allocators = type { %struct.PyMutex, %struct.anon.5, %struct.anon.6, i32, %struct.PyObjectArenaAllocator }
%struct.anon.5 = type { %struct.PyMemAllocatorEx, %struct.PyMemAllocatorEx, %struct.PyMemAllocatorEx }
%struct.PyMemAllocatorEx = type { ptr, ptr, ptr, ptr, ptr }
%struct.anon.6 = type { %struct.debug_alloc_api_t, %struct.debug_alloc_api_t, %struct.debug_alloc_api_t }
%struct.debug_alloc_api_t = type { i8, %struct.PyMemAllocatorEx }
%struct.PyObjectArenaAllocator = type { ptr, ptr, ptr }
%struct._obmalloc_global_state = type { i32, i64 }
%struct.pyhash_runtime_state = type { %struct.anon.7 }
%struct.anon.7 = type { i32, i32, i64 }
%struct._pythread_runtime_state = type { i32, %struct.anon.8, %struct.llist_node }
%struct.anon.8 = type { ptr }
%struct.llist_node = type { ptr, ptr }
%struct._signals_runtime_state = type { [32 x %struct.anon.9], %struct.anon.10, i32, ptr, ptr, i32 }
%struct.anon.9 = type { i32, ptr }
%struct.anon.10 = type { i32, i32 }
%struct._Py_tss_t = type { i32, i64 }
%struct.PyWideStringList = type { i64, ptr }
%struct._parser_runtime_state = type { i32, %struct._expr }
%struct._expr = type { i32, %union.anon.11, i32, i32, i32, i32 }
%union.anon.11 = type { %struct.anon.14 }
%struct.anon.14 = type { ptr, i32, ptr }
%struct._atexit_runtime_state = type { %struct.PyMutex, [32 x ptr], i32 }
%struct._import_runtime_state = type { ptr, i64, %struct.anon.39, ptr }
%struct.anon.39 = type { %struct.PyMutex, ptr }
%struct._ceval_runtime_state = type { %struct.anon.40, %struct._pending_calls, %struct.PyMutex }
%struct.anon.40 = type { i32 }
%struct._pending_calls = type { ptr, %struct.PyMutex, i32, i32, i32, [300 x %struct._pending_call], i32, i32 }
%struct._pending_call = type { ptr, ptr, i32 }
%struct._gilstate_runtime_state = type { i32, ptr }
%struct._getargs_runtime_state = type { ptr }
%struct._fileutils_state = type { i32 }
%struct._faulthandler_runtime_state = type { %struct.anon.41, %struct.anon.42, ptr, %struct.__darwin_sigaltstack, %struct.__darwin_sigaltstack }
%struct.anon.41 = type { i32, ptr, i32, i32, ptr }
%struct.anon.42 = type { ptr, i32, i64, i32, ptr, i32, ptr, i64, ptr, ptr }
%struct.__darwin_sigaltstack = type { ptr, i64, i32 }
%struct._tracemalloc_runtime_state = type { %struct._PyTraceMalloc_Config, %struct.anon.43, ptr, i64, i64, ptr, ptr, ptr, ptr, ptr, %struct.tracemalloc_traceback, %struct._Py_tss_t }
%struct._PyTraceMalloc_Config = type { i32, i32, i32 }
%struct.anon.43 = type { %struct.PyMemAllocatorEx, %struct.PyMemAllocatorEx, %struct.PyMemAllocatorEx }
%struct.tracemalloc_traceback = type { i64, i16, i16, [1 x %struct.tracemalloc_frame] }
%struct.tracemalloc_frame = type <{ ptr, i32 }>
%struct._reftracer_runtime_state = type { ptr, ptr }
%struct._PyRWMutex = type { i64 }
%struct._stoptheworld_state = type { %struct.PyMutex, i8, i8, i8, %struct.PyEvent, i64, ptr }
%struct.PyEvent = type { i8 }
%struct.PyPreConfig = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.anon.44 = type { %struct.PyMutex, ptr }
%struct._py_object_runtime_state = type { i32 }
%struct._Py_float_runtime_state = type { i32, i32 }
%struct._Py_unicode_runtime_state = type { %struct._Py_unicode_runtime_ids }
%struct._Py_unicode_runtime_ids = type { %struct.PyMutex, i64 }
%struct._types_runtime_state = type { i32, %struct.anon.45 }
%struct.anon.45 = type { [210 x %struct.anon.46] }
%struct.anon.46 = type { ptr, i64 }
%struct._Py_cached_objects = type { ptr }
%struct._Py_static_objects = type { %struct.anon.47 }
%struct.anon.47 = type { [262 x %struct._longobject], %struct.PyBytesObject, [256 x %struct.anon.48], %struct._Py_global_strings, %struct.PyGC_Head, %struct.PyTupleObject, %struct.PyGC_Head, %struct.PyHamtNode_Bitmap, %struct._PyContextTokenMissing }
%struct.PyBytesObject = type { %struct.PyVarObject, i64, [1 x i8] }
%struct.anon.48 = type { %struct.PyBytesObject, i8 }
%struct._Py_global_strings = type { %struct.anon.49, %struct.anon.74, [128 x %struct.anon.803], [128 x %struct.anon.804] }
%struct.anon.49 = type { %struct.anon.50, %struct.anon.52, %struct.anon.53, %struct.anon.54, %struct.anon.55, %struct.anon.56, %struct.anon.57, %struct.anon.58, %struct.anon.59, %struct.anon.60, %struct.anon.61, %struct.anon.62, %struct.anon.63, %struct.anon.64, %struct.anon.65, %struct.anon.66, %struct.anon.67, %struct.anon.68, %struct.anon.69, %struct.anon.70, %struct.anon.71, %struct.anon.72, %struct.anon.73 }
%struct.anon.50 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.PyASCIIObject = type { %struct._object, i64, i64, %struct.anon.51 }
%struct.anon.51 = type { i16, i16 }
%struct.anon.52 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.53 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.54 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.55 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.56 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.57 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.58 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.59 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.60 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.61 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.62 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.63 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.64 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.65 = type { %struct.PyASCIIObject, [1 x i8] }
%struct.anon.66 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.67 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.68 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.69 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.70 = type { %struct.PyASCIIObject, [24 x i8] }
%struct.anon.71 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.72 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.73 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.74 = type { %struct.anon.75, %struct.anon.76, %struct.anon.77, %struct.anon.78, %struct.anon.79, %struct.anon.80, %struct.anon.81, %struct.anon.82, %struct.anon.83, %struct.anon.84, %struct.anon.85, %struct.anon.86, %struct.anon.87, %struct.anon.88, %struct.anon.89, %struct.anon.90, %struct.anon.91, %struct.anon.92, %struct.anon.93, %struct.anon.94, %struct.anon.95, %struct.anon.96, %struct.anon.97, %struct.anon.98, %struct.anon.99, %struct.anon.100, %struct.anon.101, %struct.anon.102, %struct.anon.103, %struct.anon.104, %struct.anon.105, %struct.anon.106, %struct.anon.107, %struct.anon.108, %struct.anon.109, %struct.anon.110, %struct.anon.111, %struct.anon.112, %struct.anon.113, %struct.anon.114, %struct.anon.115, %struct.anon.116, %struct.anon.117, %struct.anon.118, %struct.anon.119, %struct.anon.120, %struct.anon.121, %struct.anon.122, %struct.anon.123, %struct.anon.124, %struct.anon.125, %struct.anon.126, %struct.anon.127, %struct.anon.128, %struct.anon.129, %struct.anon.130, %struct.anon.131, %struct.anon.132, %struct.anon.133, %struct.anon.134, %struct.anon.135, %struct.anon.136, %struct.anon.137, %struct.anon.138, %struct.anon.139, %struct.anon.140, %struct.anon.141, %struct.anon.142, %struct.anon.143, %struct.anon.144, %struct.anon.145, %struct.anon.146, %struct.anon.147, %struct.anon.148, %struct.anon.149, %struct.anon.150, %struct.anon.151, %struct.anon.152, %struct.anon.153, %struct.anon.154, %struct.anon.155, %struct.anon.156, %struct.anon.157, %struct.anon.158, %struct.anon.159, %struct.anon.160, %struct.anon.161, %struct.anon.162, %struct.anon.163, %struct.anon.164, %struct.anon.165, %struct.anon.166, %struct.anon.167, %struct.anon.168, %struct.anon.169, %struct.anon.170, %struct.anon.171, %struct.anon.172, %struct.anon.173, %struct.anon.174, %struct.anon.175, %struct.anon.176, %struct.anon.177, %struct.anon.178, %struct.anon.179, %struct.anon.180, %struct.anon.181, %struct.anon.182, %struct.anon.183, %struct.anon.184, %struct.anon.185, %struct.anon.186, %struct.anon.187, %struct.anon.188, %struct.anon.189, %struct.anon.190, %struct.anon.191, %struct.anon.192, %struct.anon.193, %struct.anon.194, %struct.anon.195, %struct.anon.196, %struct.anon.197, %struct.anon.198, %struct.anon.199, %struct.anon.200, %struct.anon.201, %struct.anon.202, %struct.anon.203, %struct.anon.204, %struct.anon.205, %struct.anon.206, %struct.anon.207, %struct.anon.208, %struct.anon.209, %struct.anon.210, %struct.anon.211, %struct.anon.212, %struct.anon.213, %struct.anon.214, %struct.anon.215, %struct.anon.216, %struct.anon.217, %struct.anon.218, %struct.anon.219, %struct.anon.220, %struct.anon.221, %struct.anon.222, %struct.anon.223, %struct.anon.224, %struct.anon.225, %struct.anon.226, %struct.anon.227, %struct.anon.228, %struct.anon.229, %struct.anon.230, %struct.anon.231, %struct.anon.232, %struct.anon.233, %struct.anon.234, %struct.anon.235, %struct.anon.236, %struct.anon.237, %struct.anon.238, %struct.anon.239, %struct.anon.240, %struct.anon.241, %struct.anon.242, %struct.anon.243, %struct.anon.244, %struct.anon.245, %struct.anon.246, %struct.anon.247, %struct.anon.248, %struct.anon.249, %struct.anon.250, %struct.anon.251, %struct.anon.252, %struct.anon.253, %struct.anon.254, %struct.anon.255, %struct.anon.256, %struct.anon.257, %struct.anon.258, %struct.anon.259, %struct.anon.260, %struct.anon.261, %struct.anon.262, %struct.anon.263, %struct.anon.264, %struct.anon.265, %struct.anon.266, %struct.anon.267, %struct.anon.268, %struct.anon.269, %struct.anon.270, %struct.anon.271, %struct.anon.272, %struct.anon.273, %struct.anon.274, %struct.anon.275, %struct.anon.276, %struct.anon.277, %struct.anon.278, %struct.anon.279, %struct.anon.280, %struct.anon.281, %struct.anon.282, %struct.anon.283, %struct.anon.284, %struct.anon.285, %struct.anon.286, %struct.anon.287, %struct.anon.288, %struct.anon.289, %struct.anon.290, %struct.anon.291, %struct.anon.292, %struct.anon.293, %struct.anon.294, %struct.anon.295, %struct.anon.296, %struct.anon.297, %struct.anon.298, %struct.anon.299, %struct.anon.300, %struct.anon.301, %struct.anon.302, %struct.anon.303, %struct.anon.304, %struct.anon.305, %struct.anon.306, %struct.anon.307, %struct.anon.308, %struct.anon.309, %struct.anon.310, %struct.anon.311, %struct.anon.312, %struct.anon.313, %struct.anon.314, %struct.anon.315, %struct.anon.316, %struct.anon.317, %struct.anon.318, %struct.anon.319, %struct.anon.320, %struct.anon.321, %struct.anon.322, %struct.anon.323, %struct.anon.324, %struct.anon.325, %struct.anon.326, %struct.anon.327, %struct.anon.328, %struct.anon.329, %struct.anon.330, %struct.anon.331, %struct.anon.332, %struct.anon.333, %struct.anon.334, %struct.anon.335, %struct.anon.336, %struct.anon.337, %struct.anon.338, %struct.anon.339, %struct.anon.340, %struct.anon.341, %struct.anon.342, %struct.anon.343, %struct.anon.344, %struct.anon.345, %struct.anon.346, %struct.anon.347, %struct.anon.348, %struct.anon.349, %struct.anon.350, %struct.anon.351, %struct.anon.352, %struct.anon.353, %struct.anon.354, %struct.anon.355, %struct.anon.356, %struct.anon.357, %struct.anon.358, %struct.anon.359, %struct.anon.360, %struct.anon.361, %struct.anon.362, %struct.anon.363, %struct.anon.364, %struct.anon.365, %struct.anon.366, %struct.anon.367, %struct.anon.368, %struct.anon.369, %struct.anon.370, %struct.anon.371, %struct.anon.372, %struct.anon.373, %struct.anon.374, %struct.anon.375, %struct.anon.376, %struct.anon.377, %struct.anon.378, %struct.anon.379, %struct.anon.380, %struct.anon.381, %struct.anon.382, %struct.anon.383, %struct.anon.384, %struct.anon.385, %struct.anon.386, %struct.anon.387, %struct.anon.388, %struct.anon.389, %struct.anon.390, %struct.anon.391, %struct.anon.392, %struct.anon.393, %struct.anon.394, %struct.anon.395, %struct.anon.396, %struct.anon.397, %struct.anon.398, %struct.anon.399, %struct.anon.400, %struct.anon.401, %struct.anon.402, %struct.anon.403, %struct.anon.404, %struct.anon.405, %struct.anon.406, %struct.anon.407, %struct.anon.408, %struct.anon.409, %struct.anon.410, %struct.anon.411, %struct.anon.412, %struct.anon.413, %struct.anon.414, %struct.anon.415, %struct.anon.416, %struct.anon.417, %struct.anon.418, %struct.anon.419, %struct.anon.420, %struct.anon.421, %struct.anon.422, %struct.anon.423, %struct.anon.424, %struct.anon.425, %struct.anon.426, %struct.anon.427, %struct.anon.428, %struct.anon.429, %struct.anon.430, %struct.anon.431, %struct.anon.432, %struct.anon.433, %struct.anon.434, %struct.anon.435, %struct.anon.436, %struct.anon.437, %struct.anon.438, %struct.anon.439, %struct.anon.440, %struct.anon.441, %struct.anon.442, %struct.anon.443, %struct.anon.444, %struct.anon.445, %struct.anon.446, %struct.anon.447, %struct.anon.448, %struct.anon.449, %struct.anon.450, %struct.anon.451, %struct.anon.452, %struct.anon.453, %struct.anon.454, %struct.anon.455, %struct.anon.456, %struct.anon.457, %struct.anon.458, %struct.anon.459, %struct.anon.460, %struct.anon.461, %struct.anon.462, %struct.anon.463, %struct.anon.464, %struct.anon.465, %struct.anon.466, %struct.anon.467, %struct.anon.468, %struct.anon.469, %struct.anon.470, %struct.anon.471, %struct.anon.472, %struct.anon.473, %struct.anon.474, %struct.anon.475, %struct.anon.476, %struct.anon.477, %struct.anon.478, %struct.anon.479, %struct.anon.480, %struct.anon.481, %struct.anon.482, %struct.anon.483, %struct.anon.484, %struct.anon.485, %struct.anon.486, %struct.anon.487, %struct.anon.488, %struct.anon.489, %struct.anon.490, %struct.anon.491, %struct.anon.492, %struct.anon.493, %struct.anon.494, %struct.anon.495, %struct.anon.496, %struct.anon.497, %struct.anon.498, %struct.anon.499, %struct.anon.500, %struct.anon.501, %struct.anon.502, %struct.anon.503, %struct.anon.504, %struct.anon.505, %struct.anon.506, %struct.anon.507, %struct.anon.508, %struct.anon.509, %struct.anon.510, %struct.anon.511, %struct.anon.512, %struct.anon.513, %struct.anon.514, %struct.anon.515, %struct.anon.516, %struct.anon.517, %struct.anon.518, %struct.anon.519, %struct.anon.520, %struct.anon.521, %struct.anon.522, %struct.anon.523, %struct.anon.524, %struct.anon.525, %struct.anon.526, %struct.anon.527, %struct.anon.528, %struct.anon.529, %struct.anon.530, %struct.anon.531, %struct.anon.532, %struct.anon.533, %struct.anon.534, %struct.anon.535, %struct.anon.536, %struct.anon.537, %struct.anon.538, %struct.anon.539, %struct.anon.540, %struct.anon.541, %struct.anon.542, %struct.anon.543, %struct.anon.544, %struct.anon.545, %struct.anon.546, %struct.anon.547, %struct.anon.548, %struct.anon.549, %struct.anon.550, %struct.anon.551, %struct.anon.552, %struct.anon.553, %struct.anon.554, %struct.anon.555, %struct.anon.556, %struct.anon.557, %struct.anon.558, %struct.anon.559, %struct.anon.560, %struct.anon.561, %struct.anon.562, %struct.anon.563, %struct.anon.564, %struct.anon.565, %struct.anon.566, %struct.anon.567, %struct.anon.568, %struct.anon.569, %struct.anon.570, %struct.anon.571, %struct.anon.572, %struct.anon.573, %struct.anon.574, %struct.anon.575, %struct.anon.576, %struct.anon.577, %struct.anon.578, %struct.anon.579, %struct.anon.580, %struct.anon.581, %struct.anon.582, %struct.anon.583, %struct.anon.584, %struct.anon.585, %struct.anon.586, %struct.anon.587, %struct.anon.588, %struct.anon.589, %struct.anon.590, %struct.anon.591, %struct.anon.592, %struct.anon.593, %struct.anon.594, %struct.anon.595, %struct.anon.596, %struct.anon.597, %struct.anon.598, %struct.anon.599, %struct.anon.600, %struct.anon.601, %struct.anon.602, %struct.anon.603, %struct.anon.604, %struct.anon.605, %struct.anon.606, %struct.anon.607, %struct.anon.608, %struct.anon.609, %struct.anon.610, %struct.anon.611, %struct.anon.612, %struct.anon.613, %struct.anon.614, %struct.anon.615, %struct.anon.616, %struct.anon.617, %struct.anon.618, %struct.anon.619, %struct.anon.620, %struct.anon.621, %struct.anon.622, %struct.anon.623, %struct.anon.624, %struct.anon.625, %struct.anon.626, %struct.anon.627, %struct.anon.628, %struct.anon.629, %struct.anon.630, %struct.anon.631, %struct.anon.632, %struct.anon.633, %struct.anon.634, %struct.anon.635, %struct.anon.636, %struct.anon.637, %struct.anon.638, %struct.anon.639, %struct.anon.640, %struct.anon.641, %struct.anon.642, %struct.anon.643, %struct.anon.644, %struct.anon.645, %struct.anon.646, %struct.anon.647, %struct.anon.648, %struct.anon.649, %struct.anon.650, %struct.anon.651, %struct.anon.652, %struct.anon.653, %struct.anon.654, %struct.anon.655, %struct.anon.656, %struct.anon.657, %struct.anon.658, %struct.anon.659, %struct.anon.660, %struct.anon.661, %struct.anon.662, %struct.anon.663, %struct.anon.664, %struct.anon.665, %struct.anon.666, %struct.anon.667, %struct.anon.668, %struct.anon.669, %struct.anon.670, %struct.anon.671, %struct.anon.672, %struct.anon.673, %struct.anon.674, %struct.anon.675, %struct.anon.676, %struct.anon.677, %struct.anon.678, %struct.anon.679, %struct.anon.680, %struct.anon.681, %struct.anon.682, %struct.anon.683, %struct.anon.684, %struct.anon.685, %struct.anon.686, %struct.anon.687, %struct.anon.688, %struct.anon.689, %struct.anon.690, %struct.anon.691, %struct.anon.692, %struct.anon.693, %struct.anon.694, %struct.anon.695, %struct.anon.696, %struct.anon.697, %struct.anon.698, %struct.anon.699, %struct.anon.700, %struct.anon.701, %struct.anon.702, %struct.anon.703, %struct.anon.704, %struct.anon.705, %struct.anon.706, %struct.anon.707, %struct.anon.708, %struct.anon.709, %struct.anon.710, %struct.anon.711, %struct.anon.712, %struct.anon.713, %struct.anon.714, %struct.anon.715, %struct.anon.716, %struct.anon.717, %struct.anon.718, %struct.anon.719, %struct.anon.720, %struct.anon.721, %struct.anon.722, %struct.anon.723, %struct.anon.724, %struct.anon.725, %struct.anon.726, %struct.anon.727, %struct.anon.728, %struct.anon.729, %struct.anon.730, %struct.anon.731, %struct.anon.732, %struct.anon.733, %struct.anon.734, %struct.anon.735, %struct.anon.736, %struct.anon.737, %struct.anon.738, %struct.anon.739, %struct.anon.740, %struct.anon.741, %struct.anon.742, %struct.anon.743, %struct.anon.744, %struct.anon.745, %struct.anon.746, %struct.anon.747, %struct.anon.748, %struct.anon.749, %struct.anon.750, %struct.anon.751, %struct.anon.752, %struct.anon.753, %struct.anon.754, %struct.anon.755, %struct.anon.756, %struct.anon.757, %struct.anon.758, %struct.anon.759, %struct.anon.760, %struct.anon.761, %struct.anon.762, %struct.anon.763, %struct.anon.764, %struct.anon.765, %struct.anon.766, %struct.anon.767, %struct.anon.768, %struct.anon.769, %struct.anon.770, %struct.anon.771, %struct.anon.772, %struct.anon.773, %struct.anon.774, %struct.anon.775, %struct.anon.776, %struct.anon.777, %struct.anon.778, %struct.anon.779, %struct.anon.780, %struct.anon.781, %struct.anon.782, %struct.anon.783, %struct.anon.784, %struct.anon.785, %struct.anon.786, %struct.anon.787, %struct.anon.788, %struct.anon.789, %struct.anon.790, %struct.anon.791, %struct.anon.792, %struct.anon.793, %struct.anon.794, %struct.anon.795, %struct.anon.796, %struct.anon.797, %struct.anon.798, %struct.anon.799, %struct.anon.800, %struct.anon.801, %struct.anon.802 }
%struct.anon.75 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.76 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.77 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.78 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.79 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.80 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.81 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.82 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.83 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.84 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.85 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.86 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.87 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.88 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.89 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.90 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.91 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.92 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.93 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.94 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.95 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.96 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.97 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.98 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.99 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.100 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.101 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.102 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.103 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.104 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.105 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.106 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.107 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.108 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.109 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.110 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.111 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.112 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.113 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.114 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.115 = type { %struct.PyASCIIObject, [25 x i8] }
%struct.anon.116 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.117 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.118 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.119 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.120 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.121 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.122 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.123 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.124 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.125 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.126 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.127 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.128 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.129 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.130 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.131 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.132 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.133 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.134 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.135 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.136 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.137 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.138 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.139 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.140 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.141 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.142 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.143 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.144 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.145 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.146 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.147 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.148 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.149 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.150 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.151 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.152 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.153 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.154 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.155 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.156 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.157 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.158 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.159 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.160 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.161 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.162 = type { %struct.PyASCIIObject, [21 x i8] }
%struct.anon.163 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.164 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.165 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.166 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.167 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.168 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.169 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.170 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.171 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.172 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.173 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.174 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.175 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.176 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.177 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.178 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.179 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.180 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.181 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.182 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.183 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.184 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.185 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.186 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.187 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.188 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.189 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.190 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.191 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.192 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.193 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.194 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.195 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.196 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.197 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.198 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.199 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.200 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.201 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.202 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.203 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.204 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.205 = type { %struct.PyASCIIObject, [19 x i8] }
%struct.anon.206 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.207 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.208 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.209 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.210 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.211 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.212 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.213 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.214 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.215 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.216 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.217 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.218 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.219 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.220 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.221 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.222 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.223 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.224 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.225 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.226 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.227 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.228 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.229 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.230 = type { %struct.PyASCIIObject, [22 x i8] }
%struct.anon.231 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.232 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.233 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.234 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.235 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.236 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.237 = type { %struct.PyASCIIObject, [36 x i8] }
%struct.anon.238 = type { %struct.PyASCIIObject, [25 x i8] }
%struct.anon.239 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.240 = type { %struct.PyASCIIObject, [31 x i8] }
%struct.anon.241 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.242 = type { %struct.PyASCIIObject, [19 x i8] }
%struct.anon.243 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.244 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.245 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.246 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.247 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.248 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.249 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.250 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.251 = type { %struct.PyASCIIObject, [25 x i8] }
%struct.anon.252 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.253 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.254 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.255 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.256 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.257 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.258 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.259 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.260 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.261 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.262 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.263 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.264 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.265 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.266 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.267 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.268 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.269 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.270 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.271 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.272 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.273 = type { %struct.PyASCIIObject, [19 x i8] }
%struct.anon.274 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.275 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.276 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.277 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.278 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.279 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.280 = type { %struct.PyASCIIObject, [24 x i8] }
%struct.anon.281 = type { %struct.PyASCIIObject, [28 x i8] }
%struct.anon.282 = type { %struct.PyASCIIObject, [24 x i8] }
%struct.anon.283 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.284 = type { %struct.PyASCIIObject, [26 x i8] }
%struct.anon.285 = type { %struct.PyASCIIObject, [26 x i8] }
%struct.anon.286 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.287 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.288 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.289 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.290 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.291 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.292 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.293 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.294 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.295 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.296 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.297 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.298 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.299 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.300 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.301 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.302 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.303 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.304 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.305 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.306 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.307 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.308 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.309 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.310 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.311 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.312 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.313 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.314 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.315 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.316 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.317 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.318 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.319 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.320 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.321 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.322 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.323 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.324 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.325 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.326 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.327 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.328 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.329 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.330 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.331 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.332 = type { %struct.PyASCIIObject, [23 x i8] }
%struct.anon.333 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.334 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.335 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.336 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.337 = type { %struct.PyASCIIObject, [23 x i8] }
%struct.anon.338 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.339 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.340 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.341 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.342 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.343 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.344 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.345 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.346 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.347 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.348 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.349 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.350 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.351 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.352 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.353 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.354 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.355 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.356 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.357 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.358 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.359 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.360 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.361 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.362 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.363 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.364 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.365 = type { %struct.PyASCIIObject, [19 x i8] }
%struct.anon.366 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.367 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.368 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.369 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.370 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.371 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.372 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.373 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.374 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.375 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.376 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.377 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.378 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.379 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.380 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.381 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.382 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.383 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.384 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.385 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.386 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.387 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.388 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.389 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.390 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.391 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.392 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.393 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.394 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.395 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.396 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.397 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.398 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.399 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.400 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.401 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.402 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.403 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.404 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.405 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.406 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.407 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.408 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.409 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.410 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.411 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.412 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.413 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.414 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.415 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.416 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.417 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.418 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.419 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.420 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.421 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.422 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.423 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.424 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.425 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.426 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.427 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.428 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.429 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.430 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.431 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.432 = type { %struct.PyASCIIObject, [19 x i8] }
%struct.anon.433 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.434 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.435 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.436 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.437 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.438 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.439 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.440 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.441 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.442 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.443 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.444 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.445 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.446 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.447 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.448 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.449 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.450 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.451 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.452 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.453 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.454 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.455 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.456 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.457 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.458 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.459 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.460 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.461 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.462 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.463 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.464 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.465 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.466 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.467 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.468 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.469 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.470 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.471 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.472 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.473 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.474 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.475 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.476 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.477 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.478 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.479 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.480 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.481 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.482 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.483 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.484 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.485 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.486 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.487 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.488 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.489 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.490 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.491 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.492 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.493 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.494 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.495 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.496 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.497 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.498 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.499 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.500 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.501 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.502 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.503 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.504 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.505 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.506 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.507 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.508 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.509 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.510 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.511 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.512 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.513 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.514 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.515 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.516 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.517 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.518 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.519 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.520 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.521 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.522 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.523 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.524 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.525 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.526 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.527 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.528 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.529 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.530 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.531 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.532 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.533 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.534 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.535 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.536 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.537 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.538 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.539 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.540 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.541 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.542 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.543 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.544 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.545 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.546 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.547 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.548 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.549 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.550 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.551 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.552 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.553 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.554 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.555 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.556 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.557 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.558 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.559 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.560 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.561 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.562 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.563 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.564 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.565 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.566 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.567 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.568 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.569 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.570 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.571 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.572 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.573 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.574 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.575 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.576 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.577 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.578 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.579 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.580 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.581 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.582 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.583 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.584 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.585 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.586 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.587 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.588 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.589 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.590 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.591 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.592 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.593 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.594 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.595 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.596 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.597 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.598 = type { %struct.PyASCIIObject, [18 x i8] }
%struct.anon.599 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.600 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.601 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.602 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.603 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.604 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.605 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.606 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.607 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.608 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.609 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.610 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.611 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.612 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.613 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.614 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.615 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.616 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.617 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.618 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.619 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.620 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.621 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.622 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.623 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.624 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.625 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.626 = type { %struct.PyASCIIObject, [13 x i8] }
%struct.anon.627 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.628 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.629 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.630 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.631 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.632 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.633 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.634 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.635 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.636 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.637 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.638 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.639 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.640 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.641 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.642 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.643 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.644 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.645 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.646 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.647 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.648 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.649 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.650 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.651 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.652 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.653 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.654 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.655 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.656 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.657 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.658 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.659 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.660 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.661 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.662 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.663 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.664 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.665 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.666 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.667 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.668 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.669 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.670 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.671 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.672 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.673 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.674 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.675 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.676 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.677 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.678 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.679 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.680 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.681 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.682 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.683 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.684 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.685 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.686 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.687 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.688 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.689 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.690 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.691 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.692 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.693 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.694 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.695 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.696 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.697 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.698 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.699 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.700 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.701 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.702 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.703 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.704 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.705 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.706 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.707 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.708 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.709 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.710 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.711 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.712 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.713 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.714 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.715 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.716 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.717 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.718 = type { %struct.PyASCIIObject, [19 x i8] }
%struct.anon.719 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.720 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.721 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.722 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.723 = type { %struct.PyASCIIObject, [17 x i8] }
%struct.anon.724 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.725 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.726 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.727 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.728 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.729 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.730 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.731 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.732 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.733 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.734 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.735 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.736 = type { %struct.PyASCIIObject, [11 x i8] }
%struct.anon.737 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.738 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.739 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.740 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.741 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.742 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.743 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.744 = type { %struct.PyASCIIObject, [28 x i8] }
%struct.anon.745 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.746 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.747 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.748 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.749 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.750 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.751 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.752 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.753 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.754 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.755 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.756 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.757 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.758 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.759 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.760 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.761 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.762 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.763 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.764 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.765 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.766 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.767 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.768 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.769 = type { %struct.PyASCIIObject, [10 x i8] }
%struct.anon.770 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.771 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.772 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.773 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.774 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.775 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.776 = type { %struct.PyASCIIObject, [3 x i8] }
%struct.anon.777 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.778 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.779 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.780 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.781 = type { %struct.PyASCIIObject, [15 x i8] }
%struct.anon.782 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.783 = type { %struct.PyASCIIObject, [16 x i8] }
%struct.anon.784 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.785 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.786 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.787 = type { %struct.PyASCIIObject, [7 x i8] }
%struct.anon.788 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.789 = type { %struct.PyASCIIObject, [20 x i8] }
%struct.anon.790 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.791 = type { %struct.PyASCIIObject, [12 x i8] }
%struct.anon.792 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.793 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.794 = type { %struct.PyASCIIObject, [8 x i8] }
%struct.anon.795 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.796 = type { %struct.PyASCIIObject, [4 x i8] }
%struct.anon.797 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.798 = type { %struct.PyASCIIObject, [9 x i8] }
%struct.anon.799 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.800 = type { %struct.PyASCIIObject, [14 x i8] }
%struct.anon.801 = type { %struct.PyASCIIObject, [5 x i8] }
%struct.anon.802 = type { %struct.PyASCIIObject, [6 x i8] }
%struct.anon.803 = type { %struct.PyASCIIObject, [2 x i8] }
%struct.anon.804 = type { %struct.PyCompactUnicodeObject, [2 x i8] }
%struct.PyCompactUnicodeObject = type { %struct.PyASCIIObject, i64, ptr }
%struct.PyTupleObject = type { %struct.PyVarObject, [1 x ptr] }
%struct.PyGC_Head = type { i64, i64 }
%struct.PyHamtNode_Bitmap = type { %struct.PyVarObject, i32, [1 x ptr] }
%struct._PyContextTokenMissing = type { %struct._object }
%struct._is = type { %struct._ceval_state, ptr, i64, i64, i32, i64, i32, i32, i32, i64, %struct.pythreads, ptr, ptr, i64, %struct._gc_runtime_state, ptr, ptr, %struct._import_state, %struct._gil_runtime_state, %struct.codecs_state, %struct.PyConfig, i64, ptr, ptr, ptr, ptr, [8 x ptr], i8, i64, [255 x ptr], %struct._PyXI_state_t, ptr, ptr, ptr, %struct._warnings_runtime_state, %struct.atexit_state, %struct._stoptheworld_state, %struct._qsbr_shared, ptr, ptr, [8 x ptr], [8 x ptr], [8 x ptr], i8, i8, %struct._py_object_state, %struct._Py_unicode_state, %struct._Py_long_state, %struct._dtoa_state, %struct._py_func_state, %struct._py_code_state, %struct._Py_dict_state, %struct._Py_exc_state, %struct._Py_mem_interp_free_queue, %struct.ast_state, %struct.types_state, %struct.callable_cache, ptr, ptr, i64, %struct._rare_events, ptr, %struct._Py_GlobalMonitors, i8, i8, i64, i64, [8 x [19 x ptr]], [8 x ptr], [8 x i64], %struct._Py_interp_cached_objects, %struct._Py_interp_static_objects, i64, %struct._PyThreadStateImpl }
%struct._ceval_state = type { i64, i32, ptr, i32, %struct._pending_calls }
%struct.pythreads = type { i64, ptr, ptr, ptr, i64, i64 }
%struct._gc_runtime_state = type { ptr, i32, i32, i32, %struct.gc_generation, [2 x %struct.gc_generation], %struct.gc_generation, [3 x %struct.gc_generation_stats], i32, ptr, ptr, i64, i64, i32, i32 }
%struct.gc_generation = type { %struct.PyGC_Head, i32, i32 }
%struct.gc_generation_stats = type { i64, i64, i64 }
%struct._import_state = type { ptr, ptr, ptr, i32, i32, i32, ptr, %struct._PyRecursiveMutex, %struct.anon.1 }
%struct._PyRecursiveMutex = type { %struct.PyMutex, i64, i64 }
%struct.anon.1 = type { i32, i64, i32 }
%struct._gil_runtime_state = type { i64, ptr, i32, i64, %struct._opaque_pthread_cond_t, %struct._opaque_pthread_mutex_t, %struct._opaque_pthread_cond_t, %struct._opaque_pthread_mutex_t }
%struct._opaque_pthread_cond_t = type { i64, [40 x i8] }
%struct._opaque_pthread_mutex_t = type { i64, [56 x i8] }
%struct.codecs_state = type { ptr, ptr, ptr, i32 }
%struct.PyConfig = type { i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, ptr, i32, ptr, ptr, ptr, i32, %struct.PyWideStringList, %struct.PyWideStringList, %struct.PyWideStringList, %struct.PyWideStringList, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, i32, %struct.PyWideStringList, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, i32, i32, i32 }
%struct._PyXI_state_t = type { %struct._xid_lookup_state, %struct.xi_exceptions }
%struct.xi_exceptions = type { ptr, ptr, ptr }
%struct._warnings_runtime_state = type { ptr, ptr, ptr, %struct.PyMutex, i64 }
%struct.atexit_state = type { ptr, ptr }
%struct._qsbr_shared = type { i64, i64, ptr, i64, %struct.PyMutex, ptr }
%struct._py_object_state = type { %struct._Py_freelists, i32 }
%struct._Py_freelists = type { %struct._Py_freelist, %struct._Py_freelist, [20 x %struct._Py_freelist], %struct._Py_freelist, %struct._Py_freelist, %struct._Py_freelist, %struct._Py_freelist, %struct._Py_freelist, %struct._Py_freelist, %struct._Py_freelist, %struct._Py_freelist, %struct._Py_freelist, %struct._Py_freelist }
%struct._Py_freelist = type { ptr, i64 }
%struct._Py_unicode_state = type { %struct._Py_unicode_fs_codec, ptr, %struct._Py_unicode_ids }
%struct._Py_unicode_fs_codec = type { ptr, i32, ptr, i32 }
%struct._Py_unicode_ids = type { i64, ptr }
%struct._Py_long_state = type { i32 }
%struct._dtoa_state = type { [8 x ptr], [8 x ptr], [288 x double], ptr }
%struct._py_func_state = type { i32, [4096 x %struct._func_version_cache_item] }
%struct._func_version_cache_item = type { ptr, ptr }
%struct._py_code_state = type { %struct.PyMutex, ptr }
%struct._Py_dict_state = type { i32, [8 x ptr] }
%struct._Py_exc_state = type { ptr, ptr, i32, ptr }
%struct._Py_mem_interp_free_queue = type { i32, %struct.PyMutex, %struct.llist_node }
%struct.ast_state = type { %struct._PyOnceFlag, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._PyOnceFlag = type { i8 }
%struct.types_state = type { i32, %struct.type_cache, %struct.anon.2, %struct.anon.3, %struct.PyMutex, [4096 x ptr] }
%struct.type_cache = type { [4096 x %struct.type_cache_entry] }
%struct.type_cache_entry = type { i32, ptr, ptr }
%struct.anon.2 = type { i64, [200 x %struct.managed_static_type_state] }
%struct.managed_static_type_state = type { ptr, i32, i32, i32, ptr, ptr, ptr }
%struct.anon.3 = type { i64, i64, [10 x %struct.managed_static_type_state] }
%struct.callable_cache = type { ptr, ptr, ptr, ptr }
%struct._rare_events = type { i8, i8, i8, i8, i8 }
%struct._Py_GlobalMonitors = type { [16 x i8] }
%struct._Py_interp_cached_objects = type { ptr, ptr, ptr, [10 x ptr], ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._Py_interp_static_objects = type { %struct.anon.4 }
%struct.anon.4 = type { i32, %struct.PyGC_Head, %struct.PyHamtObject, %struct.PyBaseExceptionObject }
%struct.PyHamtObject = type { %struct._object, ptr, ptr, i64 }
%struct.PyBaseExceptionObject = type { %struct._object, ptr, ptr, ptr, ptr, ptr, ptr, i8 }
%struct._PyThreadStateImpl = type { %struct._ts, ptr, ptr, %struct.llist_node }
%struct._ts = type { ptr, ptr, ptr, i64, %struct.anon.0, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, i64, i64, ptr, i64, i32, ptr, ptr, ptr, i64, i64, ptr, ptr, ptr, %struct._err_stackitem, ptr, i64, ptr, ptr }
%struct.anon.0 = type { i32 }
%struct._err_stackitem = type { ptr, ptr }
%struct.anon.805 = type { %struct.PyGC_Head, %struct.PyVarObject, [5 x ptr] }
%struct._PyArg_Parser = type { ptr, ptr, ptr, ptr, %struct._PyOnceFlag, i32, i32, i32, i32, ptr, ptr }
%struct.anon.806 = type { %struct.PyGC_Head, %struct.PyVarObject, [7 x ptr] }
%struct.PyCompilerFlags = type { i32, i32 }
%struct.anon.807 = type { %struct.PyGC_Head, %struct.PyVarObject, [2 x ptr] }
%struct.anon.808 = type { %struct.PyGC_Head, %struct.PyVarObject, [3 x ptr] }
%struct.anon.810 = type { %struct.PyGC_Head, %struct.PyVarObject, [3 x ptr] }
%struct.anon.811 = type { %struct.PyGC_Head, %struct.PyVarObject, [4 x ptr] }
%struct.anon.812 = type { %struct.PyGC_Head, %struct.PyVarObject, [2 x ptr] }
%struct.anon.813 = type { %struct.PyGC_Head, %struct.PyVarObject, [1 x ptr] }
%struct.filterobject = type { %struct._object, ptr, ptr }
%struct.mapobject = type { %struct._object, ptr, ptr, i32 }
%struct.zipobject = type { %struct._object, i64, ptr, ptr, i32 }
%struct.anon = type { i32, i32 }
%struct.PyCellObject = type { %struct._object, ptr }
%struct.PyListObject = type { %struct.PyVarObject, ptr, i64 }
%struct.PyCodeObject = type { %struct.PyVarObject, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr, i64, i32, ptr, [1 x i8] }
%struct.PyAsyncMethods = type { ptr, ptr, ptr, ptr }
%struct.PyByteArrayObject = type { %struct.PyVarObject, i64, ptr, ptr, i64 }
%struct.PyUnicodeObject = type { %struct.PyCompactUnicodeObject, %union.anon.809 }
%union.anon.809 = type { ptr }
%struct.CompensatedSum = type { double, double }
%struct.Py_complex = type { double, double }
%struct.PyFloatObject = type { %struct._object, double }

@PyType_Type = external global %struct._typeobject, align 8
@.str = private unnamed_addr constant [7 x i8] c"filter\00", align 1
@filter_doc = internal constant [173 x i8] c"filter(function, iterable, /)\0A--\0A\0AReturn an iterator yielding those items of iterable for which function(item)\0Ais true. If function is None, return the items that are true.\00", align 1
@filter_methods = internal global [2 x %struct.PyMethodDef] [%struct.PyMethodDef { ptr @.str.32, ptr @filter_reduce, i32 4, ptr @reduce_doc }, %struct.PyMethodDef zeroinitializer], align 8
@PyFilter_Type = global %struct._typeobject { %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyType_Type }, i64 0 }, ptr @.str, i64 32, i64 0, ptr @filter_dealloc, i64 0, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @PyObject_GenericGetAttr, ptr null, ptr null, i64 17408, ptr @filter_doc, ptr @filter_traverse, ptr null, ptr null, i64 0, ptr @PyObject_SelfIter, ptr @filter_next, ptr @filter_methods, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, i64 0, ptr null, ptr @PyType_GenericAlloc, ptr @filter_new, ptr @PyObject_GC_Del, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, i32 0, ptr null, ptr @filter_vectorcall, i8 0, i16 0 }, align 8
@.str.1 = private unnamed_addr constant [4 x i8] c"map\00", align 1
@map_doc = internal constant [289 x i8] c"map(function, iterable, /, *iterables, strict=False)\0A--\0A\0AMake an iterator that computes the function using arguments from\0Aeach of the iterables.  Stops when the shortest iterable is exhausted.\0A\0AIf strict is true and one of the arguments is exhausted before the others,\0Araise a ValueError.\00", align 1
@map_methods = internal global [3 x %struct.PyMethodDef] [%struct.PyMethodDef { ptr @.str.32, ptr @map_reduce, i32 4, ptr @reduce_doc }, %struct.PyMethodDef { ptr @.str.52, ptr @map_setstate, i32 8, ptr @setstate_doc }, %struct.PyMethodDef zeroinitializer], align 8
@PyMap_Type = global %struct._typeobject { %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyType_Type }, i64 0 }, ptr @.str.1, i64 40, i64 0, ptr @map_dealloc, i64 0, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @PyObject_GenericGetAttr, ptr null, ptr null, i64 17408, ptr @map_doc, ptr @map_traverse, ptr null, ptr null, i64 0, ptr @PyObject_SelfIter, ptr @map_next, ptr @map_methods, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, i64 0, ptr null, ptr @PyType_GenericAlloc, ptr @map_new, ptr @PyObject_GC_Del, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, i32 0, ptr null, ptr @map_vectorcall, i8 0, i16 0 }, align 8
@.str.2 = private unnamed_addr constant [4 x i8] c"zip\00", align 1
@zip_doc = internal constant [476 x i8] c"zip(*iterables, strict=False)\0A--\0A\0AThe zip object yields n-length tuples, where n is the number of iterables\0Apassed as positional arguments to zip().  The i-th element in every tuple\0Acomes from the i-th iterable argument to zip().  This continues until the\0Ashortest argument is exhausted.\0A\0AIf strict is true and one of the arguments is exhausted before the others,\0Araise a ValueError.\0A\0A   >>> list(zip('abcdefg', range(3), range(4)))\0A   [('a', 0, 0), ('b', 1, 1), ('c', 2, 2)]\00", align 1
@zip_methods = internal global [3 x %struct.PyMethodDef] [%struct.PyMethodDef { ptr @.str.32, ptr @zip_reduce, i32 4, ptr @reduce_doc }, %struct.PyMethodDef { ptr @.str.52, ptr @zip_setstate, i32 8, ptr @setstate_doc }, %struct.PyMethodDef zeroinitializer], align 8
@PyZip_Type = global %struct._typeobject { %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyType_Type }, i64 0 }, ptr @.str.2, i64 48, i64 0, ptr @zip_dealloc, i64 0, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @PyObject_GenericGetAttr, ptr null, ptr null, i64 17408, ptr @zip_doc, ptr @zip_traverse, ptr null, ptr null, i64 0, ptr @PyObject_SelfIter, ptr @zip_next, ptr @zip_methods, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, i64 0, ptr null, ptr @PyType_GenericAlloc, ptr @zip_new, ptr @PyObject_GC_Del, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, i32 0, ptr null, ptr null, i8 0, i16 0 }, align 8
@builtinsmodule = internal global %struct.PyModuleDef { %struct.PyModuleDef_Base { %struct._object { %union.anon { i64 552977039360 }, ptr null }, ptr null, i64 0, ptr null }, ptr @.str.74, ptr @builtin_doc, i64 -1, ptr @builtin_methods, ptr null, ptr null, ptr null, ptr null }, align 8
@.str.3 = private unnamed_addr constant [5 x i8] c"None\00", align 1
@_Py_NoneStruct = external global %struct._object, align 8
@.str.4 = private unnamed_addr constant [9 x i8] c"Ellipsis\00", align 1
@_Py_EllipsisObject = external global %struct._object, align 8
@.str.5 = private unnamed_addr constant [15 x i8] c"NotImplemented\00", align 1
@_Py_NotImplementedStruct = external global %struct._object, align 8
@.str.6 = private unnamed_addr constant [6 x i8] c"False\00", align 1
@_Py_FalseStruct = external global %struct._longobject, align 8
@.str.7 = private unnamed_addr constant [5 x i8] c"True\00", align 1
@_Py_TrueStruct = external global %struct._longobject, align 8
@.str.8 = private unnamed_addr constant [5 x i8] c"bool\00", align 1
@PyBool_Type = external global %struct._typeobject, align 8
@.str.9 = private unnamed_addr constant [11 x i8] c"memoryview\00", align 1
@PyMemoryView_Type = external global %struct._typeobject, align 8
@.str.10 = private unnamed_addr constant [10 x i8] c"bytearray\00", align 1
@PyByteArray_Type = external global %struct._typeobject, align 8
@.str.11 = private unnamed_addr constant [6 x i8] c"bytes\00", align 1
@PyBytes_Type = external global %struct._typeobject, align 8
@.str.12 = private unnamed_addr constant [12 x i8] c"classmethod\00", align 1
@PyClassMethod_Type = external global %struct._typeobject, align 8
@.str.13 = private unnamed_addr constant [8 x i8] c"complex\00", align 1
@PyComplex_Type = external global %struct._typeobject, align 8
@.str.14 = private unnamed_addr constant [5 x i8] c"dict\00", align 1
@PyDict_Type = external global %struct._typeobject, align 8
@.str.15 = private unnamed_addr constant [10 x i8] c"enumerate\00", align 1
@PyEnum_Type = external global %struct._typeobject, align 8
@.str.16 = private unnamed_addr constant [6 x i8] c"float\00", align 1
@PyFloat_Type = external global %struct._typeobject, align 8
@.str.17 = private unnamed_addr constant [10 x i8] c"frozenset\00", align 1
@PyFrozenSet_Type = external global %struct._typeobject, align 8
@.str.18 = private unnamed_addr constant [9 x i8] c"property\00", align 1
@PyProperty_Type = external global %struct._typeobject, align 8
@.str.19 = private unnamed_addr constant [4 x i8] c"int\00", align 1
@PyLong_Type = external global %struct._typeobject, align 8
@.str.20 = private unnamed_addr constant [5 x i8] c"list\00", align 1
@PyList_Type = external global %struct._typeobject, align 8
@.str.21 = private unnamed_addr constant [7 x i8] c"object\00", align 1
@PyBaseObject_Type = external global %struct._typeobject, align 8
@.str.22 = private unnamed_addr constant [6 x i8] c"range\00", align 1
@PyRange_Type = external global %struct._typeobject, align 8
@.str.23 = private unnamed_addr constant [9 x i8] c"reversed\00", align 1
@PyReversed_Type = external global %struct._typeobject, align 8
@.str.24 = private unnamed_addr constant [4 x i8] c"set\00", align 1
@PySet_Type = external global %struct._typeobject, align 8
@.str.25 = private unnamed_addr constant [6 x i8] c"slice\00", align 1
@PySlice_Type = external global %struct._typeobject, align 8
@.str.26 = private unnamed_addr constant [13 x i8] c"staticmethod\00", align 1
@PyStaticMethod_Type = external global %struct._typeobject, align 8
@.str.27 = private unnamed_addr constant [4 x i8] c"str\00", align 1
@PyUnicode_Type = external global %struct._typeobject, align 8
@.str.28 = private unnamed_addr constant [6 x i8] c"super\00", align 1
@PySuper_Type = external global %struct._typeobject, align 8
@.str.29 = private unnamed_addr constant [6 x i8] c"tuple\00", align 1
@PyTuple_Type = external global %struct._typeobject, align 8
@.str.30 = private unnamed_addr constant [5 x i8] c"type\00", align 1
@.str.31 = private unnamed_addr constant [10 x i8] c"__debug__\00", align 1
@.str.32 = private unnamed_addr constant [11 x i8] c"__reduce__\00", align 1
@reduce_doc = internal constant [39 x i8] c"Return state information for pickling.\00", align 1
@.str.33 = private unnamed_addr constant [6 x i8] c"O(OO)\00", align 1
@__func__.filter_vectorcall = private unnamed_addr constant [18 x i8] c"filter_vectorcall\00", align 1
@.str.34 = private unnamed_addr constant [14 x i8] c"bltinmodule.c\00", align 1
@.str.35 = private unnamed_addr constant [19 x i8] c"PyType_Check(type)\00", align 1
@__func__.map_next = private unnamed_addr constant [9 x i8] c"map_next\00", align 1
@.str.36 = private unnamed_addr constant [25 x i8] c"PyTuple_Check(lz->iters)\00", align 1
@PyExc_StopIteration = external global ptr, align 8
@.str.37 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.38 = private unnamed_addr constant [5 x i8] c"s 1-\00", align 1
@PyExc_ValueError = external global ptr, align 8
@.str.39 = private unnamed_addr constant [47 x i8] c"map() argument %d is shorter than argument%s%d\00", align 1
@.str.40 = private unnamed_addr constant [46 x i8] c"map() argument %d is longer than argument%s%d\00", align 1
@_Py_tss_tstate = external thread_local global ptr, align 8
@__func__.PyTuple_GET_SIZE = private unnamed_addr constant [17 x i8] c"PyTuple_GET_SIZE\00", align 1
@.str.41 = private unnamed_addr constant [14 x i8] c"tupleobject.h\00", align 1
@.str.42 = private unnamed_addr constant [18 x i8] c"PyTuple_Check(op)\00", align 1
@__func__.Py_SIZE = private unnamed_addr constant [8 x i8] c"Py_SIZE\00", align 1
@.str.43 = private unnamed_addr constant [9 x i8] c"object.h\00", align 1
@.str.44 = private unnamed_addr constant [28 x i8] c"Py_TYPE(ob) != &PyLong_Type\00", align 1
@.str.45 = private unnamed_addr constant [28 x i8] c"Py_TYPE(ob) != &PyBool_Type\00", align 1
@__func__._PyObject_VectorcallTstate = private unnamed_addr constant [27 x i8] c"_PyObject_VectorcallTstate\00", align 1
@.str.46 = private unnamed_addr constant [14 x i8] c"pycore_call.h\00", align 1
@.str.47 = private unnamed_addr constant [42 x i8] c"kwnames == NULL || PyTuple_Check(kwnames)\00", align 1
@.str.48 = private unnamed_addr constant [48 x i8] c"args != NULL || PyVectorcall_NARGS(nargsf) == 0\00", align 1
@__func__._PyVectorcall_FunctionInline = private unnamed_addr constant [29 x i8] c"_PyVectorcall_FunctionInline\00", align 1
@.str.49 = private unnamed_addr constant [17 x i8] c"callable != NULL\00", align 1
@.str.50 = private unnamed_addr constant [27 x i8] c"PyCallable_Check(callable)\00", align 1
@.str.51 = private unnamed_addr constant [11 x i8] c"offset > 0\00", align 1
@.str.52 = private unnamed_addr constant [13 x i8] c"__setstate__\00", align 1
@setstate_doc = internal constant [38 x i8] c"Set state information for unpickling.\00", align 1
@__func__.map_reduce = private unnamed_addr constant [11 x i8] c"map_reduce\00", align 1
@.str.53 = private unnamed_addr constant [4 x i8] c"ONO\00", align 1
@.str.54 = private unnamed_addr constant [3 x i8] c"ON\00", align 1
@__func__.PyTuple_SET_ITEM = private unnamed_addr constant [17 x i8] c"PyTuple_SET_ITEM\00", align 1
@.str.55 = private unnamed_addr constant [11 x i8] c"0 <= index\00", align 1
@.str.56 = private unnamed_addr constant [23 x i8] c"index < Py_SIZE(tuple)\00", align 1
@map_new.kwlist = internal global [2 x ptr] [ptr @.str.57, ptr null], align 8
@.str.57 = private unnamed_addr constant [7 x i8] c"strict\00", align 1
@.str.58 = private unnamed_addr constant [8 x i8] c"|$p:map\00", align 1
@PyExc_TypeError = external global ptr, align 8
@.str.59 = private unnamed_addr constant [40 x i8] c"map() must have at least two arguments.\00", align 1
@__func__.map_new = private unnamed_addr constant [8 x i8] c"map_new\00", align 1
@.str.60 = private unnamed_addr constant [20 x i8] c"PyTuple_Check(args)\00", align 1
@__func__.map_vectorcall = private unnamed_addr constant [15 x i8] c"map_vectorcall\00", align 1
@__func__.zip_next = private unnamed_addr constant [9 x i8] c"zip_next\00", align 1
@.str.61 = private unnamed_addr constant [27 x i8] c"PyTuple_Check(lz->ittuple)\00", align 1
@.str.62 = private unnamed_addr constant [22 x i8] c"PyTuple_Check(result)\00", align 1
@.str.63 = private unnamed_addr constant [21 x i8] c"Python/bltinmodule.c\00", align 1
@.str.64 = private unnamed_addr constant [47 x i8] c"zip() argument %d is shorter than argument%s%d\00", align 1
@.str.65 = private unnamed_addr constant [46 x i8] c"zip() argument %d is longer than argument%s%d\00", align 1
@.str.66 = private unnamed_addr constant [44 x i8] c"!_PyObject_GC_IS_TRACKED(((PyObject*)(op)))\00", align 1
@.str.67 = private unnamed_addr constant [48 x i8] c"object already tracked by the garbage collector\00", align 1
@__func__._PyObject_GC_TRACK = private unnamed_addr constant [19 x i8] c"_PyObject_GC_TRACK\00", align 1
@.str.68 = private unnamed_addr constant [37 x i8] c"(gc->_gc_prev & ((uintptr_t)2)) == 0\00", align 1
@.str.69 = private unnamed_addr constant [51 x i8] c"object is in generation which is garbage collected\00", align 1
@__func__._PyGCHead_SET_NEXT = private unnamed_addr constant [19 x i8] c"_PyGCHead_SET_NEXT\00", align 1
@.str.70 = private unnamed_addr constant [12 x i8] c"pycore_gc.h\00", align 1
@.str.71 = private unnamed_addr constant [32 x i8] c"(unext & ~_PyGC_PREV_MASK) == 0\00", align 1
@__func__._PyGCHead_SET_PREV = private unnamed_addr constant [19 x i8] c"_PyGCHead_SET_PREV\00", align 1
@.str.72 = private unnamed_addr constant [32 x i8] c"(uprev & ~_PyGC_PREV_MASK) == 0\00", align 1
@zip_new.kwlist = internal global [2 x ptr] [ptr @.str.57, ptr null], align 8
@.str.73 = private unnamed_addr constant [8 x i8] c"|$p:zip\00", align 1
@__func__.zip_new = private unnamed_addr constant [8 x i8] c"zip_new\00", align 1
@.str.74 = private unnamed_addr constant [9 x i8] c"builtins\00", align 1
@builtin_doc = internal constant [427 x i8] c"Built-in functions, types, exceptions, and other objects.\0A\0AThis module provides direct access to all 'built-in'\0Aidentifiers of Python; for example, builtins.len is\0Athe full name for the built-in function len().\0A\0AThis module is not normally accessed explicitly by most\0Aapplications, but can be useful in modules that provide\0Aobjects with the same name as a built-in value, but in\0Awhich the built-in of that name is also needed.\00", align 1
@builtin_methods = internal global [45 x %struct.PyMethodDef] [%struct.PyMethodDef { ptr @.str.75, ptr @builtin___build_class__, i32 130, ptr @build_class_doc }, %struct.PyMethodDef { ptr @.str.76, ptr @builtin___import__, i32 130, ptr @builtin___import____doc__ }, %struct.PyMethodDef { ptr @.str.77, ptr @builtin_abs, i32 8, ptr @builtin_abs__doc__ }, %struct.PyMethodDef { ptr @.str.78, ptr @builtin_all, i32 8, ptr @builtin_all__doc__ }, %struct.PyMethodDef { ptr @.str.79, ptr @builtin_any, i32 8, ptr @builtin_any__doc__ }, %struct.PyMethodDef { ptr @.str.80, ptr @builtin_ascii, i32 8, ptr @builtin_ascii__doc__ }, %struct.PyMethodDef { ptr @.str.81, ptr @builtin_bin, i32 8, ptr @builtin_bin__doc__ }, %struct.PyMethodDef { ptr @.str.82, ptr @builtin_breakpoint, i32 130, ptr @breakpoint_doc }, %struct.PyMethodDef { ptr @.str.83, ptr @builtin_callable, i32 8, ptr @builtin_callable__doc__ }, %struct.PyMethodDef { ptr @.str.84, ptr @builtin_chr, i32 8, ptr @builtin_chr__doc__ }, %struct.PyMethodDef { ptr @.str.85, ptr @builtin_compile, i32 130, ptr @builtin_compile__doc__ }, %struct.PyMethodDef { ptr @.str.86, ptr @builtin_delattr, i32 128, ptr @builtin_delattr__doc__ }, %struct.PyMethodDef { ptr @.str.87, ptr @builtin_dir, i32 1, ptr @dir_doc }, %struct.PyMethodDef { ptr @.str.88, ptr @builtin_divmod, i32 128, ptr @builtin_divmod__doc__ }, %struct.PyMethodDef { ptr @.str.89, ptr @builtin_eval, i32 130, ptr @builtin_eval__doc__ }, %struct.PyMethodDef { ptr @.str.90, ptr @builtin_exec, i32 130, ptr @builtin_exec__doc__ }, %struct.PyMethodDef { ptr @.str.91, ptr @builtin_format, i32 128, ptr @builtin_format__doc__ }, %struct.PyMethodDef { ptr @.str.92, ptr @builtin_getattr, i32 128, ptr @getattr_doc }, %struct.PyMethodDef { ptr @.str.93, ptr @builtin_globals, i32 4, ptr @builtin_globals__doc__ }, %struct.PyMethodDef { ptr @.str.94, ptr @builtin_hasattr, i32 128, ptr @builtin_hasattr__doc__ }, %struct.PyMethodDef { ptr @.str.95, ptr @builtin_hash, i32 8, ptr @builtin_hash__doc__ }, %struct.PyMethodDef { ptr @.str.96, ptr @builtin_hex, i32 8, ptr @builtin_hex__doc__ }, %struct.PyMethodDef { ptr @.str.97, ptr @builtin_id, i32 8, ptr @builtin_id__doc__ }, %struct.PyMethodDef { ptr @.str.98, ptr @builtin_input, i32 128, ptr @builtin_input__doc__ }, %struct.PyMethodDef { ptr @.str.99, ptr @builtin_isinstance, i32 128, ptr @builtin_isinstance__doc__ }, %struct.PyMethodDef { ptr @.str.100, ptr @builtin_issubclass, i32 128, ptr @builtin_issubclass__doc__ }, %struct.PyMethodDef { ptr @.str.101, ptr @builtin_iter, i32 128, ptr @iter_doc }, %struct.PyMethodDef { ptr @.str.102, ptr @builtin_aiter, i32 8, ptr @builtin_aiter__doc__ }, %struct.PyMethodDef { ptr @.str.103, ptr @builtin_len, i32 8, ptr @builtin_len__doc__ }, %struct.PyMethodDef { ptr @.str.104, ptr @builtin_locals, i32 4, ptr @builtin_locals__doc__ }, %struct.PyMethodDef { ptr @.str.105, ptr @builtin_max, i32 130, ptr @max_doc }, %struct.PyMethodDef { ptr @.str.106, ptr @builtin_min, i32 130, ptr @min_doc }, %struct.PyMethodDef { ptr @.str.107, ptr @builtin_next, i32 128, ptr @next_doc }, %struct.PyMethodDef { ptr @.str.108, ptr @builtin_anext, i32 128, ptr @builtin_anext__doc__ }, %struct.PyMethodDef { ptr @.str.109, ptr @builtin_oct, i32 8, ptr @builtin_oct__doc__ }, %struct.PyMethodDef { ptr @.str.110, ptr @builtin_ord, i32 8, ptr @builtin_ord__doc__ }, %struct.PyMethodDef { ptr @.str.111, ptr @builtin_pow, i32 130, ptr @builtin_pow__doc__ }, %struct.PyMethodDef { ptr @.str.112, ptr @builtin_print, i32 130, ptr @builtin_print__doc__ }, %struct.PyMethodDef { ptr @.str.113, ptr @builtin_repr, i32 8, ptr @builtin_repr__doc__ }, %struct.PyMethodDef { ptr @.str.114, ptr @builtin_round, i32 130, ptr @builtin_round__doc__ }, %struct.PyMethodDef { ptr @.str.115, ptr @builtin_setattr, i32 128, ptr @builtin_setattr__doc__ }, %struct.PyMethodDef { ptr @.str.116, ptr @builtin_sorted, i32 130, ptr @builtin_sorted__doc__ }, %struct.PyMethodDef { ptr @.str.117, ptr @builtin_sum, i32 130, ptr @builtin_sum__doc__ }, %struct.PyMethodDef { ptr @.str.118, ptr @builtin_vars, i32 1, ptr @vars_doc }, %struct.PyMethodDef zeroinitializer], align 8
@.str.75 = private unnamed_addr constant [16 x i8] c"__build_class__\00", align 1
@build_class_doc = internal constant [124 x i8] c"__build_class__(func, name, /, *bases, [metaclass], **kwds) -> class\0A\0AInternal helper function used by the class statement.\00", align 1
@.str.76 = private unnamed_addr constant [11 x i8] c"__import__\00", align 1
@builtin___import____doc__ = internal constant [892 x i8] c"__import__($module, /, name, globals=None, locals=None, fromlist=(),\0A           level=0)\0A--\0A\0AImport a module.\0A\0ABecause this function is meant for use by the Python\0Ainterpreter and not for general use, it is better to use\0Aimportlib.import_module() to programmatically import a module.\0A\0AThe globals argument is only used to determine the context;\0Athey are not modified.  The locals argument is unused.  The fromlist\0Ashould be a list of names to emulate ``from name import ...``, or an\0Aempty list to emulate ``import name``.\0AWhen importing a module from a package, note that __import__('A.B', ...)\0Areturns package A when fromlist is empty, but its submodule B when\0Afromlist is not empty.  The level argument is used to determine whether to\0Aperform absolute or relative imports: 0 is absolute, while a positive number\0Ais the number of parent directories to search relative to the current module.\00", align 1
@.str.77 = private unnamed_addr constant [4 x i8] c"abs\00", align 1
@builtin_abs__doc__ = internal constant [66 x i8] c"abs($module, x, /)\0A--\0A\0AReturn the absolute value of the argument.\00", align 1
@.str.78 = private unnamed_addr constant [4 x i8] c"all\00", align 1
@builtin_all__doc__ = internal constant [135 x i8] c"all($module, iterable, /)\0A--\0A\0AReturn True if bool(x) is True for all values x in the iterable.\0A\0AIf the iterable is empty, return True.\00", align 1
@.str.79 = private unnamed_addr constant [4 x i8] c"any\00", align 1
@builtin_any__doc__ = internal constant [129 x i8] c"any($module, iterable, /)\0A--\0A\0AReturn True if bool(x) is True for any x in the iterable.\0A\0AIf the iterable is empty, return False.\00", align 1
@.str.80 = private unnamed_addr constant [6 x i8] c"ascii\00", align 1
@builtin_ascii__doc__ = internal constant [329 x i8] c"ascii($module, obj, /)\0A--\0A\0AReturn an ASCII-only representation of an object.\0A\0AAs repr(), return a string containing a printable representation of an\0Aobject, but escape the non-ASCII characters in the string returned by\0Arepr() using \\\\x, \\\\u or \\\\U escapes. This generates a string similar\0Ato that returned by repr() in Python 2.\00", align 1
@.str.81 = private unnamed_addr constant [4 x i8] c"bin\00", align 1
@builtin_bin__doc__ = internal constant [127 x i8] c"bin($module, number, /)\0A--\0A\0AReturn the binary representation of an integer.\0A\0A   >>> bin(2796202)\0A   '0b1010101010101010101010'\00", align 1
@.str.82 = private unnamed_addr constant [11 x i8] c"breakpoint\00", align 1
@breakpoint_doc = internal constant [196 x i8] c"breakpoint($module, /, *args, **kws)\0A--\0A\0ACall sys.breakpointhook(*args, **kws).  sys.breakpointhook() must accept\0Awhatever arguments are passed.\0A\0ABy default, this drops you into the pdb debugger.\00", align 1
@.str.83 = private unnamed_addr constant [9 x i8] c"callable\00", align 1
@builtin_callable__doc__ = internal constant [186 x i8] c"callable($module, obj, /)\0A--\0A\0AReturn whether the object is callable (i.e., some kind of function).\0A\0ANote that classes are callable, as are instances of classes with a\0A__call__() method.\00", align 1
@.str.84 = private unnamed_addr constant [4 x i8] c"chr\00", align 1
@builtin_chr__doc__ = internal constant [100 x i8] c"chr($module, i, /)\0A--\0A\0AReturn a Unicode string of one character with ordinal i; 0 <= i <= 0x10ffff.\00", align 1
@.str.85 = private unnamed_addr constant [8 x i8] c"compile\00", align 1
@builtin_compile__doc__ = internal constant [826 x i8] c"compile($module, /, source, filename, mode, flags=0,\0A        dont_inherit=False, optimize=-1, *, _feature_version=-1)\0A--\0A\0ACompile source into a code object that can be executed by exec() or eval().\0A\0AThe source code may represent a Python module, statement or expression.\0AThe filename will be used for run-time error messages.\0AThe mode must be 'exec' to compile a module, 'single' to compile a\0Asingle (interactive) statement, or 'eval' to compile an expression.\0AThe flags argument, if present, controls which future statements influence\0Athe compilation of the code.\0AThe dont_inherit argument, if true, stops the compilation inheriting\0Athe effects of any future statements in effect in the code calling\0Acompile; if absent or false these statements do influence the compilation,\0Ain addition to any features explicitly specified.\00", align 1
@.str.86 = private unnamed_addr constant [8 x i8] c"delattr\00", align 1
@builtin_delattr__doc__ = internal constant [132 x i8] c"delattr($module, obj, name, /)\0A--\0A\0ADeletes the named attribute from the given object.\0A\0Adelattr(x, 'y') is equivalent to ``del x.y``\00", align 1
@.str.87 = private unnamed_addr constant [4 x i8] c"dir\00", align 1
@dir_doc = internal constant [624 x i8] c"dir([object]) -> list of strings\0A\0AIf called without an argument, return the names in the current scope.\0AElse, return an alphabetized list of names comprising (some of) the attributes\0Aof the given object, and of attributes reachable from it.\0AIf the object supplies a method named __dir__, it will be used; otherwise\0Athe default dir() logic is used and returns:\0A  for a module object: the module's attributes.\0A  for a class object:  its attributes, and recursively the attributes\0A    of its bases.\0A  for any other object: its attributes, its class's attributes, and\0A    recursively the attributes of its class's base classes.\00", align 1
@.str.88 = private unnamed_addr constant [7 x i8] c"divmod\00", align 1
@builtin_divmod__doc__ = internal constant [89 x i8] c"divmod($module, x, y, /)\0A--\0A\0AReturn the tuple (x//y, x%y).  Invariant: div*y + mod == x.\00", align 1
@.str.89 = private unnamed_addr constant [5 x i8] c"eval\00", align 1
@builtin_eval__doc__ = internal constant [383 x i8] c"eval($module, source, /, globals=None, locals=None)\0A--\0A\0AEvaluate the given source in the context of globals and locals.\0A\0AThe source may be a string representing a Python expression\0Aor a code object as returned by compile().\0AThe globals must be a dictionary and locals can be any mapping,\0Adefaulting to the current globals and locals.\0AIf only globals is given, locals defaults to it.\00", align 1
@.str.90 = private unnamed_addr constant [5 x i8] c"exec\00", align 1
@builtin_exec__doc__ = internal constant [538 x i8] c"exec($module, source, /, globals=None, locals=None, *, closure=None)\0A--\0A\0AExecute the given source in the context of globals and locals.\0A\0AThe source may be a string representing one or more Python statements\0Aor a code object as returned by compile().\0AThe globals must be a dictionary and locals can be any mapping,\0Adefaulting to the current globals and locals.\0AIf only globals is given, locals defaults to it.\0AThe closure must be a tuple of cellvars, and can only be used\0Awhen source is a code object requiring exactly that many cellvars.\00", align 1
@.str.91 = private unnamed_addr constant [7 x i8] c"format\00", align 1
@builtin_format__doc__ = internal constant [362 x i8] c"format($module, value, format_spec='', /)\0A--\0A\0AReturn type(value).__format__(value, format_spec)\0A\0AMany built-in types implement format_spec according to the\0AFormat Specification Mini-language. See help('FORMATTING').\0A\0AIf type(value) does not supply a method named __format__\0Aand format_spec is empty, then str(value) is returned.\0ASee also help('SPECIALMETHODS').\00", align 1
@.str.92 = private unnamed_addr constant [8 x i8] c"getattr\00", align 1
@getattr_doc = internal constant [251 x i8] c"getattr(object, name[, default]) -> value\0A\0AGet a named attribute from an object; getattr(x, 'y') is equivalent to x.y.\0AWhen a default argument is given, it is returned when the attribute doesn't\0Aexist; without it, an exception is raised in that case.\00", align 1
@.str.93 = private unnamed_addr constant [8 x i8] c"globals\00", align 1
@builtin_globals__doc__ = internal constant [200 x i8] c"globals($module, /)\0A--\0A\0AReturn the dictionary containing the current scope's global variables.\0A\0ANOTE: Updates to this dictionary *will* affect name lookups in the current\0Aglobal scope and vice-versa.\00", align 1
@.str.94 = private unnamed_addr constant [8 x i8] c"hasattr\00", align 1
@builtin_hasattr__doc__ = internal constant [172 x i8] c"hasattr($module, obj, name, /)\0A--\0A\0AReturn whether the object has an attribute with the given name.\0A\0AThis is done by calling getattr(obj, name) and catching AttributeError.\00", align 1
@.str.95 = private unnamed_addr constant [5 x i8] c"hash\00", align 1
@builtin_hash__doc__ = internal constant [179 x i8] c"hash($module, obj, /)\0A--\0A\0AReturn the hash value for the given object.\0A\0ATwo objects that compare equal must also have the same hash value, but the\0Areverse is not necessarily true.\00", align 1
@.str.96 = private unnamed_addr constant [4 x i8] c"hex\00", align 1
@builtin_hex__doc__ = internal constant [117 x i8] c"hex($module, number, /)\0A--\0A\0AReturn the hexadecimal representation of an integer.\0A\0A   >>> hex(12648430)\0A   '0xc0ffee'\00", align 1
@.str.97 = private unnamed_addr constant [3 x i8] c"id\00", align 1
@builtin_id__doc__ = internal constant [174 x i8] c"id($module, obj, /)\0A--\0A\0AReturn the identity of an object.\0A\0AThis is guaranteed to be unique among simultaneously existing objects.\0A(CPython uses the object's memory address.)\00", align 1
@.str.98 = private unnamed_addr constant [6 x i8] c"input\00", align 1
@builtin_input__doc__ = internal constant [338 x i8] c"input($module, prompt='', /)\0A--\0A\0ARead a string from standard input.  The trailing newline is stripped.\0A\0AThe prompt string, if given, is printed to standard output without a\0Atrailing newline before reading input.\0A\0AIf the user hits EOF (*nix: Ctrl-D, Windows: Ctrl-Z+Return), raise EOFError.\0AOn *nix systems, readline is used if available.\00", align 1
@.str.99 = private unnamed_addr constant [11 x i8] c"isinstance\00", align 1
@builtin_isinstance__doc__ = internal constant [293 x i8] c"isinstance($module, obj, class_or_tuple, /)\0A--\0A\0AReturn whether an object is an instance of a class or of a subclass thereof.\0A\0AA tuple, as in ``isinstance(x, (A, B, ...))``, may be given as the target to\0Acheck against. This is equivalent to ``isinstance(x, A) or isinstance(x, B)\0Aor ...`` etc.\00", align 1
@.str.100 = private unnamed_addr constant [11 x i8] c"issubclass\00", align 1
@builtin_issubclass__doc__ = internal constant [285 x i8] c"issubclass($module, cls, class_or_tuple, /)\0A--\0A\0AReturn whether 'cls' is derived from another class or is the same class.\0A\0AA tuple, as in ``issubclass(x, (A, B, ...))``, may be given as the target to\0Acheck against. This is equivalent to ``issubclass(x, A) or issubclass(x, B)\0Aor ...``.\00", align 1
@.str.101 = private unnamed_addr constant [5 x i8] c"iter\00", align 1
@iter_doc = internal constant [252 x i8] c"iter(iterable) -> iterator\0Aiter(callable, sentinel) -> iterator\0A\0AGet an iterator from an object.  In the first form, the argument must\0Asupply its own iterator, or be a sequence.\0AIn the second form, the callable is called until it returns the sentinel.\00", align 1
@.str.102 = private unnamed_addr constant [6 x i8] c"aiter\00", align 1
@builtin_aiter__doc__ = internal constant [91 x i8] c"aiter($module, async_iterable, /)\0A--\0A\0AReturn an AsyncIterator for an AsyncIterable object.\00", align 1
@.str.103 = private unnamed_addr constant [4 x i8] c"len\00", align 1
@builtin_len__doc__ = internal constant [68 x i8] c"len($module, obj, /)\0A--\0A\0AReturn the number of items in a container.\00", align 1
@.str.104 = private unnamed_addr constant [7 x i8] c"locals\00", align 1
@builtin_locals__doc__ = internal constant [288 x i8] c"locals($module, /)\0A--\0A\0AReturn a dictionary containing the current scope's local variables.\0A\0ANOTE: Whether or not updates to this dictionary will affect name lookups in\0Athe local scope and vice-versa is *implementation dependent* and not\0Acovered by any backwards compatibility guarantees.\00", align 1
@.str.105 = private unnamed_addr constant [4 x i8] c"max\00", align 1
@max_doc = internal constant [324 x i8] c"max(iterable, *[, default=obj, key=func]) -> value\0Amax(arg1, arg2, *args, *[, key=func]) -> value\0A\0AWith a single iterable argument, return its biggest item. The\0Adefault keyword-only argument specifies an object to return if\0Athe provided iterable is empty.\0AWith two or more positional arguments, return the largest argument.\00", align 1
@.str.106 = private unnamed_addr constant [4 x i8] c"min\00", align 1
@min_doc = internal constant [326 x i8] c"min(iterable, *[, default=obj, key=func]) -> value\0Amin(arg1, arg2, *args, *[, key=func]) -> value\0A\0AWith a single iterable argument, return its smallest item. The\0Adefault keyword-only argument specifies an object to return if\0Athe provided iterable is empty.\0AWith two or more positional arguments, return the smallest argument.\00", align 1
@.str.107 = private unnamed_addr constant [5 x i8] c"next\00", align 1
@next_doc = internal constant [167 x i8] c"next(iterator[, default])\0A\0AReturn the next item from the iterator. If default is given and the iterator\0Ais exhausted, it is returned instead of raising StopIteration.\00", align 1
@.str.108 = private unnamed_addr constant [6 x i8] c"anext\00", align 1
@builtin_anext__doc__ = internal constant [218 x i8] c"anext($module, aiterator, default=<unrepresentable>, /)\0A--\0A\0AReturn the next item from the async iterator.\0A\0AIf default is given and the async iterator is exhausted,\0Ait is returned instead of raising StopAsyncIteration.\00", align 1
@.str.109 = private unnamed_addr constant [4 x i8] c"oct\00", align 1
@builtin_oct__doc__ = internal constant [110 x i8] c"oct($module, number, /)\0A--\0A\0AReturn the octal representation of an integer.\0A\0A   >>> oct(342391)\0A   '0o1234567'\00", align 1
@.str.110 = private unnamed_addr constant [4 x i8] c"ord\00", align 1
@builtin_ord__doc__ = internal constant [81 x i8] c"ord($module, c, /)\0A--\0A\0AReturn the Unicode code point for a one-character string.\00", align 1
@.str.111 = private unnamed_addr constant [4 x i8] c"pow\00", align 1
@builtin_pow__doc__ = internal constant [232 x i8] c"pow($module, /, base, exp, mod=None)\0A--\0A\0AEquivalent to base**exp with 2 arguments or base**exp % mod with 3 arguments\0A\0ASome types, such as ints, are able to use a more efficient algorithm when\0Ainvoked using the three argument form.\00", align 1
@.str.112 = private unnamed_addr constant [6 x i8] c"print\00", align 1
@builtin_print__doc__ = internal constant [385 x i8] c"print($module, /, *args, sep=' ', end='\\n', file=None, flush=False)\0A--\0A\0APrints the values to a stream, or to sys.stdout by default.\0A\0A  sep\0A    string inserted between values, default a space.\0A  end\0A    string appended after the last value, default a newline.\0A  file\0A    a file-like object (stream); defaults to the current sys.stdout.\0A  flush\0A    whether to forcibly flush the stream.\00", align 1
@.str.113 = private unnamed_addr constant [5 x i8] c"repr\00", align 1
@builtin_repr__doc__ = internal constant [157 x i8] c"repr($module, obj, /)\0A--\0A\0AReturn the canonical string representation of the object.\0A\0AFor many object types, including most builtins, eval(repr(obj)) == obj.\00", align 1
@.str.114 = private unnamed_addr constant [6 x i8] c"round\00", align 1
@builtin_round__doc__ = internal constant [249 x i8] c"round($module, /, number, ndigits=None)\0A--\0A\0ARound a number to a given precision in decimal digits.\0A\0AThe return value is an integer if ndigits is omitted or None.  Otherwise\0Athe return value has the same type as the number.  ndigits may be negative.\00", align 1
@.str.115 = private unnamed_addr constant [8 x i8] c"setattr\00", align 1
@builtin_setattr__doc__ = internal constant [160 x i8] c"setattr($module, obj, name, value, /)\0A--\0A\0ASets the named attribute on the given object to the specified value.\0A\0Asetattr(x, 'y', v) is equivalent to ``x.y = v``\00", align 1
@.str.116 = private unnamed_addr constant [7 x i8] c"sorted\00", align 1
@builtin_sorted__doc__ = internal constant [281 x i8] c"sorted($module, iterable, /, *, key=None, reverse=False)\0A--\0A\0AReturn a new list containing all items from the iterable in ascending order.\0A\0AA custom key function can be supplied to customize the sort order, and the\0Areverse flag can be set to request the result in descending order.\00", align 1
@.str.117 = private unnamed_addr constant [4 x i8] c"sum\00", align 1
@builtin_sum__doc__ = internal constant [268 x i8] c"sum($module, iterable, /, start=0)\0A--\0A\0AReturn the sum of a 'start' value (default: 0) plus an iterable of numbers\0A\0AWhen the iterable is empty, return the start value.\0AThis function is intended specifically for use with numeric values and may\0Areject non-numeric types.\00", align 1
@.str.118 = private unnamed_addr constant [5 x i8] c"vars\00", align 1
@vars_doc = internal constant [122 x i8] c"vars([object]) -> dictionary\0A\0AWithout arguments, equivalent to locals().\0AWith an argument, equivalent to object.__dict__.\00", align 1
@.str.119 = private unnamed_addr constant [38 x i8] c"__build_class__: not enough arguments\00", align 1
@PyFunction_Type = external global %struct._typeobject, align 8
@.str.120 = private unnamed_addr constant [41 x i8] c"__build_class__: func must be a function\00", align 1
@.str.121 = private unnamed_addr constant [38 x i8] c"__build_class__: name is not a string\00", align 1
@_PyRuntime = external global %struct.pyruntimestate, align 8
@__func__.builtin___build_class__ = private unnamed_addr constant [24 x i8] c"builtin___build_class__\00", align 1
@.str.122 = private unnamed_addr constant [21 x i8] c"PyTuple_Check(bases)\00", align 1
@.str.123 = private unnamed_addr constant [55 x i8] c"%.200s.__prepare__() must return a mapping, not %.200s\00", align 1
@.str.124 = private unnamed_addr constant [12 x i8] c"<metaclass>\00", align 1
@.str.125 = private unnamed_addr constant [15 x i8] c"__orig_bases__\00", align 1
@PyCell_Type = external global %struct._typeobject, align 8
@.str.126 = private unnamed_addr constant [91 x i8] c"__class__ not set defining %.200R as %.200R. Was __classcell__ propagated to type.__new__?\00", align 1
@PyExc_RuntimeError = external global ptr, align 8
@.str.127 = private unnamed_addr constant [50 x i8] c"__class__ set to %.200R defining %.200R as %.200R\00", align 1
@__func__.update_bases = private unnamed_addr constant [13 x i8] c"update_bases\00", align 1
@.str.128 = private unnamed_addr constant [36 x i8] c"__mro_entries__ must return a tuple\00", align 1
@__func__.PyList_SET_ITEM = private unnamed_addr constant [16 x i8] c"PyList_SET_ITEM\00", align 1
@.str.129 = private unnamed_addr constant [13 x i8] c"listobject.h\00", align 1
@.str.130 = private unnamed_addr constant [17 x i8] c"PyList_Check(op)\00", align 1
@.str.131 = private unnamed_addr constant [24 x i8] c"index < list->allocated\00", align 1
@__func__.PyList_GET_SIZE = private unnamed_addr constant [16 x i8] c"PyList_GET_SIZE\00", align 1
@builtin___import__._kwtuple = internal global %struct.anon.805 { %struct.PyGC_Head zeroinitializer, %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyTuple_Type }, i64 5 }, [5 x ptr] [ptr getelementptr (i8, ptr @_PyRuntime, i64 63424), ptr getelementptr (i8, ptr @_PyRuntime, i64 57208), ptr getelementptr (i8, ptr @_PyRuntime, i64 61544), ptr getelementptr (i8, ptr @_PyRuntime, i64 56376), ptr getelementptr (i8, ptr @_PyRuntime, i64 61096)] }, align 8
@builtin___import__._keywords = internal constant [6 x ptr] [ptr @.str.132, ptr @.str.93, ptr @.str.104, ptr @.str.133, ptr @.str.134, ptr null], align 8
@.str.132 = private unnamed_addr constant [5 x i8] c"name\00", align 1
@.str.133 = private unnamed_addr constant [9 x i8] c"fromlist\00", align 1
@.str.134 = private unnamed_addr constant [6 x i8] c"level\00", align 1
@builtin___import__._parser = internal global %struct._PyArg_Parser { ptr null, ptr @builtin___import__._keywords, ptr @.str.76, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr getelementptr (i8, ptr @builtin___import__._kwtuple, i64 16), ptr null }, align 8
@.str.135 = private unnamed_addr constant [15 x i8] c"breakpointhook\00", align 1
@.str.136 = private unnamed_addr constant [24 x i8] c"lost sys.breakpointhook\00", align 1
@.str.137 = private unnamed_addr constant [20 x i8] c"builtins.breakpoint\00", align 1
@.str.138 = private unnamed_addr constant [2 x i8] c"O\00", align 1
@builtin_compile._kwtuple = internal global %struct.anon.806 { %struct.PyGC_Head zeroinitializer, %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyTuple_Type }, i64 7 }, [7 x ptr] [ptr getelementptr (i8, ptr @_PyRuntime, i64 69760), ptr getelementptr (i8, ptr @_PyRuntime, i64 55544), ptr getelementptr (i8, ptr @_PyRuntime, i64 62752), ptr getelementptr (i8, ptr @_PyRuntime, i64 56016), ptr getelementptr (i8, ptr @_PyRuntime, i64 53560), ptr getelementptr (i8, ptr @_PyRuntime, i64 65128), ptr getelementptr (i8, ptr @_PyRuntime, i64 45272)] }, align 8
@builtin_compile._keywords = internal constant [8 x ptr] [ptr @.str.139, ptr @.str.140, ptr @.str.141, ptr @.str.142, ptr @.str.143, ptr @.str.144, ptr @.str.145, ptr null], align 8
@.str.139 = private unnamed_addr constant [7 x i8] c"source\00", align 1
@.str.140 = private unnamed_addr constant [9 x i8] c"filename\00", align 1
@.str.141 = private unnamed_addr constant [5 x i8] c"mode\00", align 1
@.str.142 = private unnamed_addr constant [6 x i8] c"flags\00", align 1
@.str.143 = private unnamed_addr constant [13 x i8] c"dont_inherit\00", align 1
@.str.144 = private unnamed_addr constant [9 x i8] c"optimize\00", align 1
@.str.145 = private unnamed_addr constant [17 x i8] c"_feature_version\00", align 1
@builtin_compile._parser = internal global %struct._PyArg_Parser { ptr null, ptr @builtin_compile._keywords, ptr @.str.85, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr getelementptr (i8, ptr @builtin_compile._kwtuple, i64 16), ptr null }, align 8
@.str.146 = private unnamed_addr constant [16 x i8] c"argument 'mode'\00", align 1
@.str.147 = private unnamed_addr constant [24 x i8] c"embedded null character\00", align 1
@__const.builtin_compile_impl.start = private unnamed_addr constant [4 x i32] [i32 257, i32 258, i32 256, i32 345], align 4
@__const.builtin_compile_impl.cf = private unnamed_addr constant %struct.PyCompilerFlags { i32 0, i32 14 }, align 4
@.str.148 = private unnamed_addr constant [30 x i8] c"compile(): unrecognised flags\00", align 1
@.str.149 = private unnamed_addr constant [34 x i8] c"compile(): invalid optimize value\00", align 1
@.str.150 = private unnamed_addr constant [7 x i8] c"single\00", align 1
@.str.151 = private unnamed_addr constant [10 x i8] c"func_type\00", align 1
@.str.152 = private unnamed_addr constant [55 x i8] c"compile() mode 'func_type' requires flag PyCF_ONLY_AST\00", align 1
@.str.153 = private unnamed_addr constant [63 x i8] c"compile() mode must be 'exec', 'eval', 'single' or 'func_type'\00", align 1
@.str.154 = private unnamed_addr constant [50 x i8] c"compile() mode must be 'exec', 'eval' or 'single'\00", align 1
@.str.155 = private unnamed_addr constant [21 x i8] c"string, bytes or AST\00", align 1
@builtin_eval._kwtuple = internal global %struct.anon.807 { %struct.PyGC_Head zeroinitializer, %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyTuple_Type }, i64 2 }, [2 x ptr] [ptr getelementptr (i8, ptr @_PyRuntime, i64 57208), ptr getelementptr (i8, ptr @_PyRuntime, i64 61544)] }, align 8
@builtin_eval._keywords = internal constant [4 x ptr] [ptr @.str.156, ptr @.str.93, ptr @.str.104, ptr null], align 8
@.str.156 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@builtin_eval._parser = internal global %struct._PyArg_Parser { ptr null, ptr @builtin_eval._keywords, ptr @.str.89, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr getelementptr (i8, ptr @builtin_eval._kwtuple, i64 16), ptr null }, align 8
@.str.157 = private unnamed_addr constant [25 x i8] c"locals must be a mapping\00", align 1
@.str.158 = private unnamed_addr constant [57 x i8] c"globals must be a real dict; try eval(expr, {}, mapping)\00", align 1
@.str.159 = private unnamed_addr constant [23 x i8] c"globals must be a dict\00", align 1
@.str.160 = private unnamed_addr constant [66 x i8] c"eval must be given globals and locals when called without a frame\00", align 1
@PyCode_Type = external global %struct._typeobject, align 8
@.str.161 = private unnamed_addr constant [60 x i8] c"code object passed to eval() may not contain free variables\00", align 1
@__const.builtin_eval_impl.cf = private unnamed_addr constant %struct.PyCompilerFlags { i32 0, i32 14 }, align 4
@.str.162 = private unnamed_addr constant [22 x i8] c"string, bytes or code\00", align 1
@__func__.PyCode_GetNumFree = private unnamed_addr constant [18 x i8] c"PyCode_GetNumFree\00", align 1
@.str.163 = private unnamed_addr constant [7 x i8] c"code.h\00", align 1
@.str.164 = private unnamed_addr constant [17 x i8] c"PyCode_Check(op)\00", align 1
@builtin_exec._kwtuple = internal global %struct.anon.808 { %struct.PyGC_Head zeroinitializer, %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyTuple_Type }, i64 3 }, [3 x ptr] [ptr getelementptr (i8, ptr @_PyRuntime, i64 57208), ptr getelementptr (i8, ptr @_PyRuntime, i64 61544), ptr getelementptr (i8, ptr @_PyRuntime, i64 50392)] }, align 8
@builtin_exec._keywords = internal constant [5 x ptr] [ptr @.str.156, ptr @.str.93, ptr @.str.104, ptr @.str.165, ptr null], align 8
@.str.165 = private unnamed_addr constant [8 x i8] c"closure\00", align 1
@builtin_exec._parser = internal global %struct._PyArg_Parser { ptr null, ptr @builtin_exec._keywords, ptr @.str.90, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr getelementptr (i8, ptr @builtin_exec._kwtuple, i64 16), ptr null }, align 8
@PyExc_SystemError = external global ptr, align 8
@.str.166 = private unnamed_addr constant [34 x i8] c"globals and locals cannot be NULL\00", align 1
@.str.167 = private unnamed_addr constant [42 x i8] c"exec() globals must be a dict, not %.100s\00", align 1
@.str.168 = private unnamed_addr constant [45 x i8] c"locals must be a mapping or None, not %.100s\00", align 1
@.str.169 = private unnamed_addr constant [43 x i8] c"cannot use a closure with this code object\00", align 1
@__func__.builtin_exec_impl = private unnamed_addr constant [18 x i8] c"builtin_exec_impl\00", align 1
@.str.170 = private unnamed_addr constant [23 x i8] c"PyTuple_Check(closure)\00", align 1
@.str.171 = private unnamed_addr constant [53 x i8] c"code object requires a closure of exactly length %zd\00", align 1
@.str.172 = private unnamed_addr constant [54 x i8] c"closure can only be used when source is a code object\00", align 1
@__const.builtin_exec_impl.cf = private unnamed_addr constant %struct.PyCompilerFlags { i32 0, i32 14 }, align 4
@.str.173 = private unnamed_addr constant [11 x i8] c"argument 2\00", align 1
@.str.174 = private unnamed_addr constant [12 x i8] c"builtins.id\00", align 1
@.str.175 = private unnamed_addr constant [24 x i8] c"input(): lost sys.stdin\00", align 1
@.str.176 = private unnamed_addr constant [25 x i8] c"input(): lost sys.stdout\00", align 1
@.str.177 = private unnamed_addr constant [25 x i8] c"input(): lost sys.stderr\00", align 1
@.str.178 = private unnamed_addr constant [15 x i8] c"builtins.input\00", align 1
@__stdinp = external global ptr, align 8
@__stdoutp = external global ptr, align 8
@__func__.builtin_input_impl = private unnamed_addr constant [19 x i8] c"builtin_input_impl\00", align 1
@.str.179 = private unnamed_addr constant [18 x i8] c"PyBytes_Check(po)\00", align 1
@.str.180 = private unnamed_addr constant [52 x i8] c"input: prompt string cannot contain null characters\00", align 1
@PyExc_KeyboardInterrupt = external global ptr, align 8
@PyExc_EOFError = external global ptr, align 8
@PyExc_OverflowError = external global ptr, align 8
@.str.181 = private unnamed_addr constant [22 x i8] c"input: input too long\00", align 1
@.str.182 = private unnamed_addr constant [22 x i8] c"builtins.input/result\00", align 1
@__func__.PyBytes_AS_STRING = private unnamed_addr constant [18 x i8] c"PyBytes_AS_STRING\00", align 1
@.str.183 = private unnamed_addr constant [14 x i8] c"bytesobject.h\00", align 1
@.str.184 = private unnamed_addr constant [18 x i8] c"PyBytes_Check(op)\00", align 1
@__func__.PyBytes_GET_SIZE = private unnamed_addr constant [17 x i8] c"PyBytes_GET_SIZE\00", align 1
@.str.185 = private unnamed_addr constant [31 x i8] c"iter(v, w): v must be callable\00", align 1
@__func__.builtin_len = private unnamed_addr constant [12 x i8] c"builtin_len\00", align 1
@.str.186 = private unnamed_addr constant [17 x i8] c"PyErr_Occurred()\00", align 1
@min_max.keywords = internal constant [3 x ptr] [ptr @.str.187, ptr @.str.188, ptr null], align 8
@.str.187 = private unnamed_addr constant [4 x i8] c"key\00", align 1
@.str.188 = private unnamed_addr constant [8 x i8] c"default\00", align 1
@min_max._parser_min = internal global %struct._PyArg_Parser { ptr @.str.189, ptr @min_max.keywords, ptr null, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr null, ptr null }, align 8
@.str.189 = private unnamed_addr constant [9 x i8] c"|$OO:min\00", align 1
@min_max._parser_max = internal global %struct._PyArg_Parser { ptr @.str.190, ptr @min_max.keywords, ptr null, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr null, ptr null }, align 8
@.str.190 = private unnamed_addr constant [9 x i8] c"|$OO:max\00", align 1
@.str.191 = private unnamed_addr constant [39 x i8] c"%s expected at least 1 argument, got 0\00", align 1
@.str.192 = private unnamed_addr constant [69 x i8] c"Cannot specify a default for %s() with multiple positional arguments\00", align 1
@__func__.min_max = private unnamed_addr constant [8 x i8] c"min_max\00", align 1
@.str.193 = private unnamed_addr constant [16 x i8] c"maxitem == NULL\00", align 1
@.str.194 = private unnamed_addr constant [32 x i8] c"%s() iterable argument is empty\00", align 1
@.str.195 = private unnamed_addr constant [35 x i8] c"'%.200s' object is not an iterator\00", align 1
@.str.196 = private unnamed_addr constant [41 x i8] c"'%.200s' object is not an async iterator\00", align 1
@.str.197 = private unnamed_addr constant [52 x i8] c"ord() expected string of length 1, but %.200s found\00", align 1
@.str.198 = private unnamed_addr constant [59 x i8] c"ord() expected a character, but string of length %zd found\00", align 1
@__func__.PyUnicode_GET_LENGTH = private unnamed_addr constant [21 x i8] c"PyUnicode_GET_LENGTH\00", align 1
@.str.199 = private unnamed_addr constant [16 x i8] c"unicodeobject.h\00", align 1
@.str.200 = private unnamed_addr constant [20 x i8] c"PyUnicode_Check(op)\00", align 1
@__func__.PyUnicode_READ_CHAR = private unnamed_addr constant [20 x i8] c"PyUnicode_READ_CHAR\00", align 1
@.str.201 = private unnamed_addr constant [11 x i8] c"index >= 0\00", align 1
@.str.202 = private unnamed_addr constant [39 x i8] c"index <= PyUnicode_GET_LENGTH(unicode)\00", align 1
@.str.203 = private unnamed_addr constant [25 x i8] c"PyUnicode_Check(unicode)\00", align 1
@.str.204 = private unnamed_addr constant [29 x i8] c"kind == PyUnicode_4BYTE_KIND\00", align 1
@__func__.PyUnicode_IS_COMPACT = private unnamed_addr constant [21 x i8] c"PyUnicode_IS_COMPACT\00", align 1
@__func__._PyUnicode_COMPACT_DATA = private unnamed_addr constant [24 x i8] c"_PyUnicode_COMPACT_DATA\00", align 1
@__func__.PyUnicode_IS_ASCII = private unnamed_addr constant [19 x i8] c"PyUnicode_IS_ASCII\00", align 1
@__func__._PyUnicode_NONCOMPACT_DATA = private unnamed_addr constant [27 x i8] c"_PyUnicode_NONCOMPACT_DATA\00", align 1
@.str.205 = private unnamed_addr constant [26 x i8] c"!PyUnicode_IS_COMPACT(op)\00", align 1
@.str.206 = private unnamed_addr constant [13 x i8] c"data != NULL\00", align 1
@__func__.PyByteArray_GET_SIZE = private unnamed_addr constant [21 x i8] c"PyByteArray_GET_SIZE\00", align 1
@.str.207 = private unnamed_addr constant [18 x i8] c"bytearrayobject.h\00", align 1
@.str.208 = private unnamed_addr constant [22 x i8] c"PyByteArray_Check(op)\00", align 1
@__func__.PyByteArray_AS_STRING = private unnamed_addr constant [22 x i8] c"PyByteArray_AS_STRING\00", align 1
@_PyByteArray_empty_string = external global [0 x i8], align 1
@builtin_pow._kwtuple = internal global %struct.anon.810 { %struct.PyGC_Head zeroinitializer, %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyTuple_Type }, i64 3 }, [3 x ptr] [ptr getelementptr (i8, ptr @_PyRuntime, i64 48392), ptr getelementptr (i8, ptr @_PyRuntime, i64 54800), ptr getelementptr (i8, ptr @_PyRuntime, i64 62704)] }, align 8
@builtin_pow._keywords = internal constant [4 x ptr] [ptr @.str.209, ptr @.str.210, ptr @.str.211, ptr null], align 8
@.str.209 = private unnamed_addr constant [5 x i8] c"base\00", align 1
@.str.210 = private unnamed_addr constant [4 x i8] c"exp\00", align 1
@.str.211 = private unnamed_addr constant [4 x i8] c"mod\00", align 1
@builtin_pow._parser = internal global %struct._PyArg_Parser { ptr null, ptr @builtin_pow._keywords, ptr @.str.111, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr getelementptr (i8, ptr @builtin_pow._kwtuple, i64 16), ptr null }, align 8
@builtin_print._kwtuple = internal global %struct.anon.811 { %struct.PyGC_Head zeroinitializer, %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyTuple_Type }, i64 4 }, [4 x ptr] [ptr getelementptr (i8, ptr @_PyRuntime, i64 68712), ptr getelementptr (i8, ptr @_PyRuntime, i64 53992), ptr getelementptr (i8, ptr @_PyRuntime, i64 55440), ptr getelementptr (i8, ptr @_PyRuntime, i64 56064)] }, align 8
@builtin_print._keywords = internal constant [5 x ptr] [ptr @.str.212, ptr @.str.213, ptr @.str.214, ptr @.str.215, ptr null], align 8
@.str.212 = private unnamed_addr constant [4 x i8] c"sep\00", align 1
@.str.213 = private unnamed_addr constant [4 x i8] c"end\00", align 1
@.str.214 = private unnamed_addr constant [5 x i8] c"file\00", align 1
@.str.215 = private unnamed_addr constant [6 x i8] c"flush\00", align 1
@builtin_print._parser = internal global %struct._PyArg_Parser { ptr null, ptr @builtin_print._keywords, ptr @.str.112, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr getelementptr (i8, ptr @builtin_print._kwtuple, i64 16), ptr null }, align 8
@.str.216 = private unnamed_addr constant [16 x i8] c"lost sys.stdout\00", align 1
@.str.217 = private unnamed_addr constant [41 x i8] c"sep must be None or a string, not %.200s\00", align 1
@.str.218 = private unnamed_addr constant [41 x i8] c"end must be None or a string, not %.200s\00", align 1
@.str.219 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@builtin_round._kwtuple = internal global %struct.anon.812 { %struct.PyGC_Head zeroinitializer, %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyTuple_Type }, i64 2 }, [2 x ptr] [ptr getelementptr (i8, ptr @_PyRuntime, i64 64408), ptr getelementptr (i8, ptr @_PyRuntime, i64 63696)] }, align 8
@builtin_round._keywords = internal constant [3 x ptr] [ptr @.str.220, ptr @.str.221, ptr null], align 8
@.str.220 = private unnamed_addr constant [7 x i8] c"number\00", align 1
@.str.221 = private unnamed_addr constant [8 x i8] c"ndigits\00", align 1
@builtin_round._parser = internal global %struct._PyArg_Parser { ptr null, ptr @builtin_round._keywords, ptr @.str.114, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr getelementptr (i8, ptr @builtin_round._kwtuple, i64 16), ptr null }, align 8
@.str.222 = private unnamed_addr constant [44 x i8] c"type %.100s doesn't define __round__ method\00", align 1
@__func__.builtin_sorted = private unnamed_addr constant [15 x i8] c"builtin_sorted\00", align 1
@.str.223 = private unnamed_addr constant [11 x i8] c"nargs >= 1\00", align 1
@builtin_sum._kwtuple = internal global %struct.anon.813 { %struct.PyGC_Head zeroinitializer, %struct.PyVarObject { %struct._object { %union.anon { i64 552977039360 }, ptr @PyTuple_Type }, i64 1 }, [1 x ptr] [ptr getelementptr (i8, ptr @_PyRuntime, i64 70080)] }, align 8
@builtin_sum._keywords = internal constant [3 x ptr] [ptr @.str.156, ptr @.str.224, ptr null], align 8
@.str.224 = private unnamed_addr constant [6 x i8] c"start\00", align 1
@builtin_sum._parser = internal global %struct._PyArg_Parser { ptr null, ptr @builtin_sum._keywords, ptr @.str.117, ptr null, %struct._PyOnceFlag zeroinitializer, i32 0, i32 0, i32 0, i32 0, ptr getelementptr (i8, ptr @builtin_sum._kwtuple, i64 16), ptr null }, align 8
@.str.225 = private unnamed_addr constant [51 x i8] c"sum() can't sum strings [use ''.join(seq) instead]\00", align 1
@.str.226 = private unnamed_addr constant [50 x i8] c"sum() can't sum bytes [use b''.join(seq) instead]\00", align 1
@.str.227 = private unnamed_addr constant [54 x i8] c"sum() can't sum bytearray [use b''.join(seq) instead]\00", align 1
@__func__._PyLong_IsCompact = private unnamed_addr constant [18 x i8] c"_PyLong_IsCompact\00", align 1
@.str.228 = private unnamed_addr constant [14 x i8] c"longintrepr.h\00", align 1
@.str.229 = private unnamed_addr constant [65 x i8] c"PyType_HasFeature(op->ob_base.ob_type, Py_TPFLAGS_LONG_SUBCLASS)\00", align 1
@__func__._PyLong_CompactValue = private unnamed_addr constant [21 x i8] c"_PyLong_CompactValue\00", align 1
@.str.230 = private unnamed_addr constant [30 x i8] c"PyUnstable_Long_IsCompact(op)\00", align 1
@__func__.PyFloat_AS_DOUBLE = private unnamed_addr constant [18 x i8] c"PyFloat_AS_DOUBLE\00", align 1
@.str.231 = private unnamed_addr constant [14 x i8] c"floatobject.h\00", align 1
@.str.232 = private unnamed_addr constant [18 x i8] c"PyFloat_Check(op)\00", align 1
@__func__._Py_DECREF_SPECIALIZED = private unnamed_addr constant [23 x i8] c"_Py_DECREF_SPECIALIZED\00", align 1
@.str.233 = private unnamed_addr constant [16 x i8] c"pycore_object.h\00", align 1
@.str.234 = private unnamed_addr constant [18 x i8] c"op->ob_refcnt > 0\00", align 1
@.str.235 = private unnamed_addr constant [45 x i8] c"vars() argument must have __dict__ attribute\00", align 1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @filter_dealloc(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  call void @PyObject_GC_UnTrack(ptr noundef %4)
  br label %5

5:                                                ; preds = %1
  %6 = call ptr @PyThreadState_Get()
  store ptr %6, ptr %3, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = getelementptr inbounds %struct._ts, ptr %7, i32 0, i32 9
  %9 = load i32, ptr %8, align 4
  %10 = icmp sle i32 %9, 50
  br i1 %10, label %11, label %20

11:                                               ; preds = %5
  %12 = load ptr, ptr %2, align 8
  %13 = call ptr @_Py_TYPE(ptr noundef %12)
  %14 = getelementptr inbounds %struct._typeobject, ptr %13, i32 0, i32 4
  %15 = load ptr, ptr %14, align 8
  %16 = icmp eq ptr %15, @filter_dealloc
  br i1 %16, label %17, label %20

17:                                               ; preds = %11
  %18 = load ptr, ptr %3, align 8
  %19 = load ptr, ptr %2, align 8
  call void @_PyTrash_thread_deposit_object(ptr noundef %18, ptr noundef %19)
  br label %52

20:                                               ; preds = %11, %5
  %21 = load ptr, ptr %3, align 8
  %22 = getelementptr inbounds %struct._ts, ptr %21, i32 0, i32 9
  %23 = load i32, ptr %22, align 4
  %24 = add nsw i32 %23, -1
  store i32 %24, ptr %22, align 4
  %25 = load ptr, ptr %2, align 8
  %26 = getelementptr inbounds %struct.filterobject, ptr %25, i32 0, i32 1
  %27 = load ptr, ptr %26, align 8
  call void @Py_XDECREF(ptr noundef %27)
  %28 = load ptr, ptr %2, align 8
  %29 = getelementptr inbounds %struct.filterobject, ptr %28, i32 0, i32 2
  %30 = load ptr, ptr %29, align 8
  call void @Py_XDECREF(ptr noundef %30)
  %31 = load ptr, ptr %2, align 8
  %32 = call ptr @_Py_TYPE(ptr noundef %31)
  %33 = getelementptr inbounds %struct._typeobject, ptr %32, i32 0, i32 38
  %34 = load ptr, ptr %33, align 8
  %35 = load ptr, ptr %2, align 8
  call void %34(ptr noundef %35)
  %36 = load ptr, ptr %3, align 8
  %37 = getelementptr inbounds %struct._ts, ptr %36, i32 0, i32 9
  %38 = load i32, ptr %37, align 4
  %39 = add nsw i32 %38, 1
  store i32 %39, ptr %37, align 4
  %40 = load ptr, ptr %3, align 8
  %41 = getelementptr inbounds %struct._ts, ptr %40, i32 0, i32 25
  %42 = load ptr, ptr %41, align 8
  %43 = icmp ne ptr %42, null
  br i1 %43, label %44, label %51

44:                                               ; preds = %20
  %45 = load ptr, ptr %3, align 8
  %46 = getelementptr inbounds %struct._ts, ptr %45, i32 0, i32 9
  %47 = load i32, ptr %46, align 4
  %48 = icmp sgt i32 %47, 100
  br i1 %48, label %49, label %51

49:                                               ; preds = %44
  %50 = load ptr, ptr %3, align 8
  call void @_PyTrash_thread_destroy_chain(ptr noundef %50)
  br label %51

51:                                               ; preds = %49, %44, %20
  br label %52

52:                                               ; preds = %51, %17
  ret void
}

declare ptr @PyObject_GenericGetAttr(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @filter_traverse(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  br label %10

10:                                               ; preds = %3
  %11 = load ptr, ptr %5, align 8
  %12 = getelementptr inbounds %struct.filterobject, ptr %11, i32 0, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %27

15:                                               ; preds = %10
  %16 = load ptr, ptr %6, align 8
  %17 = load ptr, ptr %5, align 8
  %18 = getelementptr inbounds %struct.filterobject, ptr %17, i32 0, i32 2
  %19 = load ptr, ptr %18, align 8
  %20 = load ptr, ptr %7, align 8
  %21 = call i32 %16(ptr noundef %19, ptr noundef %20)
  store i32 %21, ptr %8, align 4
  %22 = load i32, ptr %8, align 4
  %23 = icmp ne i32 %22, 0
  br i1 %23, label %24, label %26

24:                                               ; preds = %15
  %25 = load i32, ptr %8, align 4
  store i32 %25, ptr %4, align 4
  br label %48

26:                                               ; preds = %15
  br label %27

27:                                               ; preds = %26, %10
  br label %28

28:                                               ; preds = %27
  br label %29

29:                                               ; preds = %28
  %30 = load ptr, ptr %5, align 8
  %31 = getelementptr inbounds %struct.filterobject, ptr %30, i32 0, i32 1
  %32 = load ptr, ptr %31, align 8
  %33 = icmp ne ptr %32, null
  br i1 %33, label %34, label %46

34:                                               ; preds = %29
  %35 = load ptr, ptr %6, align 8
  %36 = load ptr, ptr %5, align 8
  %37 = getelementptr inbounds %struct.filterobject, ptr %36, i32 0, i32 1
  %38 = load ptr, ptr %37, align 8
  %39 = load ptr, ptr %7, align 8
  %40 = call i32 %35(ptr noundef %38, ptr noundef %39)
  store i32 %40, ptr %9, align 4
  %41 = load i32, ptr %9, align 4
  %42 = icmp ne i32 %41, 0
  br i1 %42, label %43, label %45

43:                                               ; preds = %34
  %44 = load i32, ptr %9, align 4
  store i32 %44, ptr %4, align 4
  br label %48

45:                                               ; preds = %34
  br label %46

46:                                               ; preds = %45, %29
  br label %47

47:                                               ; preds = %46
  store i32 0, ptr %4, align 4
  br label %48

48:                                               ; preds = %47, %43, %24
  %49 = load i32, ptr %4, align 4
  ret i32 %49
}

declare ptr @PyObject_SelfIter(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @filter_next(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i64, align 8
  %13 = alloca ptr, align 8
  %14 = alloca i32, align 4
  %15 = alloca ptr, align 8
  store ptr %0, ptr %9, align 8
  %16 = load ptr, ptr %9, align 8
  %17 = getelementptr inbounds %struct.filterobject, ptr %16, i32 0, i32 2
  %18 = load ptr, ptr %17, align 8
  store ptr %18, ptr %11, align 8
  %19 = load ptr, ptr %9, align 8
  %20 = getelementptr inbounds %struct.filterobject, ptr %19, i32 0, i32 1
  %21 = load ptr, ptr %20, align 8
  %22 = icmp eq ptr %21, @_Py_NoneStruct
  br i1 %22, label %28, label %23

23:                                               ; preds = %1
  %24 = load ptr, ptr %9, align 8
  %25 = getelementptr inbounds %struct.filterobject, ptr %24, i32 0, i32 1
  %26 = load ptr, ptr %25, align 8
  %27 = icmp eq ptr %26, @PyBool_Type
  br label %28

28:                                               ; preds = %23, %1
  %29 = phi i1 [ true, %1 ], [ %27, %23 ]
  %30 = zext i1 %29 to i32
  store i32 %30, ptr %14, align 4
  %31 = load ptr, ptr %11, align 8
  %32 = call ptr @_Py_TYPE(ptr noundef %31)
  %33 = getelementptr inbounds %struct._typeobject, ptr %32, i32 0, i32 26
  %34 = load ptr, ptr %33, align 8
  store ptr %34, ptr %13, align 8
  br label %35

35:                                               ; preds = %119, %28
  %36 = load ptr, ptr %13, align 8
  %37 = load ptr, ptr %11, align 8
  %38 = call ptr %36(ptr noundef %37)
  store ptr %38, ptr %10, align 8
  %39 = load ptr, ptr %10, align 8
  %40 = icmp eq ptr %39, null
  br i1 %40, label %41, label %42

41:                                               ; preds = %35
  store ptr null, ptr %8, align 8
  br label %120

42:                                               ; preds = %35
  %43 = load i32, ptr %14, align 4
  %44 = icmp ne i32 %43, 0
  br i1 %44, label %45, label %49

45:                                               ; preds = %42
  %46 = load ptr, ptr %10, align 8
  %47 = call i32 @PyObject_IsTrue(ptr noundef %46)
  %48 = sext i32 %47 to i64
  store i64 %48, ptr %12, align 8
  br label %94

49:                                               ; preds = %42
  %50 = load ptr, ptr %9, align 8
  %51 = getelementptr inbounds %struct.filterobject, ptr %50, i32 0, i32 1
  %52 = load ptr, ptr %51, align 8
  %53 = load ptr, ptr %10, align 8
  %54 = call ptr @PyObject_CallOneArg(ptr noundef %52, ptr noundef %53)
  store ptr %54, ptr %15, align 8
  %55 = load ptr, ptr %15, align 8
  %56 = icmp eq ptr %55, null
  br i1 %56, label %57, label %74

57:                                               ; preds = %49
  %58 = load ptr, ptr %10, align 8
  store ptr %58, ptr %5, align 8
  %59 = load ptr, ptr %5, align 8
  store ptr %59, ptr %4, align 8
  %60 = load ptr, ptr %4, align 8
  %61 = load i32, ptr %60, align 8
  %62 = icmp slt i32 %61, 0
  %63 = zext i1 %62 to i32
  %64 = icmp ne i32 %63, 0
  br i1 %64, label %65, label %66

65:                                               ; preds = %57
  br label %73

66:                                               ; preds = %57
  %67 = load ptr, ptr %5, align 8
  %68 = load i32, ptr %67, align 8
  %69 = add i32 %68, -1
  store i32 %69, ptr %67, align 8
  %70 = icmp eq i32 %69, 0
  br i1 %70, label %71, label %73

71:                                               ; preds = %66
  %72 = load ptr, ptr %5, align 8
  call void @_Py_Dealloc(ptr noundef %72) #7
  br label %73

73:                                               ; preds = %65, %66, %71
  store ptr null, ptr %8, align 8
  br label %120

74:                                               ; preds = %49
  %75 = load ptr, ptr %15, align 8
  %76 = call i32 @PyObject_IsTrue(ptr noundef %75)
  %77 = sext i32 %76 to i64
  store i64 %77, ptr %12, align 8
  %78 = load ptr, ptr %15, align 8
  store ptr %78, ptr %6, align 8
  %79 = load ptr, ptr %6, align 8
  store ptr %79, ptr %3, align 8
  %80 = load ptr, ptr %3, align 8
  %81 = load i32, ptr %80, align 8
  %82 = icmp slt i32 %81, 0
  %83 = zext i1 %82 to i32
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %86

85:                                               ; preds = %74
  br label %93

86:                                               ; preds = %74
  %87 = load ptr, ptr %6, align 8
  %88 = load i32, ptr %87, align 8
  %89 = add i32 %88, -1
  store i32 %89, ptr %87, align 8
  %90 = icmp eq i32 %89, 0
  br i1 %90, label %91, label %93

91:                                               ; preds = %86
  %92 = load ptr, ptr %6, align 8
  call void @_Py_Dealloc(ptr noundef %92) #7
  br label %93

93:                                               ; preds = %85, %86, %91
  br label %94

94:                                               ; preds = %93, %45
  %95 = load i64, ptr %12, align 8
  %96 = icmp sgt i64 %95, 0
  br i1 %96, label %97, label %99

97:                                               ; preds = %94
  %98 = load ptr, ptr %10, align 8
  store ptr %98, ptr %8, align 8
  br label %120

99:                                               ; preds = %94
  %100 = load ptr, ptr %10, align 8
  store ptr %100, ptr %7, align 8
  %101 = load ptr, ptr %7, align 8
  store ptr %101, ptr %2, align 8
  %102 = load ptr, ptr %2, align 8
  %103 = load i32, ptr %102, align 8
  %104 = icmp slt i32 %103, 0
  %105 = zext i1 %104 to i32
  %106 = icmp ne i32 %105, 0
  br i1 %106, label %107, label %108

107:                                              ; preds = %99
  br label %115

108:                                              ; preds = %99
  %109 = load ptr, ptr %7, align 8
  %110 = load i32, ptr %109, align 8
  %111 = add i32 %110, -1
  store i32 %111, ptr %109, align 8
  %112 = icmp eq i32 %111, 0
  br i1 %112, label %113, label %115

113:                                              ; preds = %108
  %114 = load ptr, ptr %7, align 8
  call void @_Py_Dealloc(ptr noundef %114) #7
  br label %115

115:                                              ; preds = %107, %108, %113
  %116 = load i64, ptr %12, align 8
  %117 = icmp slt i64 %116, 0
  br i1 %117, label %118, label %119

118:                                              ; preds = %115
  store ptr null, ptr %8, align 8
  br label %120

119:                                              ; preds = %115
  br label %35

120:                                              ; preds = %118, %97, %73, %41
  %121 = load ptr, ptr %8, align 8
  ret ptr %121
}

declare ptr @PyType_GenericAlloc(ptr noundef, i64 noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @filter_new(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  %14 = load ptr, ptr %7, align 8
  %15 = icmp eq ptr %14, @PyFilter_Type
  br i1 %15, label %22, label %16

16:                                               ; preds = %3
  %17 = load ptr, ptr %7, align 8
  %18 = getelementptr inbounds %struct._typeobject, ptr %17, i32 0, i32 35
  %19 = load ptr, ptr %18, align 8
  %20 = load ptr, ptr getelementptr inbounds (%struct._typeobject, ptr @PyFilter_Type, i32 0, i32 35), align 8
  %21 = icmp eq ptr %19, %20
  br i1 %21, label %22, label %30

22:                                               ; preds = %16, %3
  %23 = load ptr, ptr %9, align 8
  %24 = icmp eq ptr %23, null
  br i1 %24, label %30, label %25

25:                                               ; preds = %22
  %26 = load ptr, ptr %9, align 8
  %27 = call i32 @_PyArg_NoKeywords(ptr noundef @.str, ptr noundef %26)
  %28 = icmp ne i32 %27, 0
  br i1 %28, label %30, label %29

29:                                               ; preds = %25
  store ptr null, ptr %6, align 8
  br label %75

30:                                               ; preds = %25, %22, %16
  %31 = load ptr, ptr %8, align 8
  %32 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr noundef %31, ptr noundef @.str, i64 noundef 2, i64 noundef 2, ptr noundef %10, ptr noundef %11)
  %33 = icmp ne i32 %32, 0
  br i1 %33, label %35, label %34

34:                                               ; preds = %30
  store ptr null, ptr %6, align 8
  br label %75

35:                                               ; preds = %30
  %36 = load ptr, ptr %11, align 8
  %37 = call ptr @PyObject_GetIter(ptr noundef %36)
  store ptr %37, ptr %12, align 8
  %38 = load ptr, ptr %12, align 8
  %39 = icmp eq ptr %38, null
  br i1 %39, label %40, label %41

40:                                               ; preds = %35
  store ptr null, ptr %6, align 8
  br label %75

41:                                               ; preds = %35
  %42 = load ptr, ptr %7, align 8
  %43 = getelementptr inbounds %struct._typeobject, ptr %42, i32 0, i32 36
  %44 = load ptr, ptr %43, align 8
  %45 = load ptr, ptr %7, align 8
  %46 = call ptr %44(ptr noundef %45, i64 noundef 0)
  store ptr %46, ptr %13, align 8
  %47 = load ptr, ptr %13, align 8
  %48 = icmp eq ptr %47, null
  br i1 %48, label %49, label %66

49:                                               ; preds = %41
  %50 = load ptr, ptr %12, align 8
  store ptr %50, ptr %5, align 8
  %51 = load ptr, ptr %5, align 8
  store ptr %51, ptr %4, align 8
  %52 = load ptr, ptr %4, align 8
  %53 = load i32, ptr %52, align 8
  %54 = icmp slt i32 %53, 0
  %55 = zext i1 %54 to i32
  %56 = icmp ne i32 %55, 0
  br i1 %56, label %57, label %58

57:                                               ; preds = %49
  br label %65

58:                                               ; preds = %49
  %59 = load ptr, ptr %5, align 8
  %60 = load i32, ptr %59, align 8
  %61 = add i32 %60, -1
  store i32 %61, ptr %59, align 8
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %63, label %65

63:                                               ; preds = %58
  %64 = load ptr, ptr %5, align 8
  call void @_Py_Dealloc(ptr noundef %64) #7
  br label %65

65:                                               ; preds = %57, %58, %63
  store ptr null, ptr %6, align 8
  br label %75

66:                                               ; preds = %41
  %67 = load ptr, ptr %10, align 8
  %68 = call ptr @_Py_NewRef(ptr noundef %67)
  %69 = load ptr, ptr %13, align 8
  %70 = getelementptr inbounds %struct.filterobject, ptr %69, i32 0, i32 1
  store ptr %68, ptr %70, align 8
  %71 = load ptr, ptr %12, align 8
  %72 = load ptr, ptr %13, align 8
  %73 = getelementptr inbounds %struct.filterobject, ptr %72, i32 0, i32 2
  store ptr %71, ptr %73, align 8
  %74 = load ptr, ptr %13, align 8
  store ptr %74, ptr %6, align 8
  br label %75

75:                                               ; preds = %66, %65, %40, %34, %29
  %76 = load ptr, ptr %6, align 8
  ret ptr %76
}

declare void @PyObject_GC_Del(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @filter_vectorcall(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i64, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i64, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  store ptr %0, ptr %8, align 8
  store ptr %1, ptr %9, align 8
  store i64 %2, ptr %10, align 8
  store ptr %3, ptr %11, align 8
  %16 = load ptr, ptr %8, align 8
  %17 = call i32 @PyType_Check(ptr noundef %16)
  %18 = icmp ne i32 %17, 0
  %19 = xor i1 %18, true
  %20 = zext i1 %19 to i32
  %21 = sext i32 %20 to i64
  %22 = icmp ne i64 %21, 0
  br i1 %22, label %23, label %25

23:                                               ; preds = %4
  call void @__assert_rtn(ptr noundef @__func__.filter_vectorcall, ptr noundef @.str.34, i32 noundef 533, ptr noundef @.str.35) #8
  unreachable

24:                                               ; No predecessors!
  br label %26

25:                                               ; preds = %4
  br label %26

26:                                               ; preds = %25, %24
  %27 = load ptr, ptr %8, align 8
  store ptr %27, ptr %12, align 8
  %28 = load ptr, ptr %12, align 8
  %29 = icmp eq ptr %28, @PyFilter_Type
  br i1 %29, label %30, label %38

30:                                               ; preds = %26
  %31 = load ptr, ptr %11, align 8
  %32 = icmp eq ptr %31, null
  br i1 %32, label %38, label %33

33:                                               ; preds = %30
  %34 = load ptr, ptr %11, align 8
  %35 = call i32 @_PyArg_NoKwnames(ptr noundef @.str, ptr noundef %34)
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %38, label %37

37:                                               ; preds = %33
  store ptr null, ptr %7, align 8
  br label %95

38:                                               ; preds = %33, %30, %26
  %39 = load i64, ptr %10, align 8
  %40 = call i64 @_PyVectorcall_NARGS(i64 noundef %39)
  store i64 %40, ptr %13, align 8
  %41 = load i64, ptr %13, align 8
  %42 = icmp sle i64 2, %41
  br i1 %42, label %43, label %46

43:                                               ; preds = %38
  %44 = load i64, ptr %13, align 8
  %45 = icmp sle i64 %44, 2
  br i1 %45, label %51, label %46

46:                                               ; preds = %43, %38
  %47 = load i64, ptr %13, align 8
  %48 = call i32 @_PyArg_CheckPositional(ptr noundef @.str, i64 noundef %47, i64 noundef 2, i64 noundef 2)
  %49 = icmp ne i32 %48, 0
  br i1 %49, label %51, label %50

50:                                               ; preds = %46
  store ptr null, ptr %7, align 8
  br label %95

51:                                               ; preds = %46, %43
  %52 = load ptr, ptr %9, align 8
  %53 = getelementptr inbounds ptr, ptr %52, i64 1
  %54 = load ptr, ptr %53, align 8
  %55 = call ptr @PyObject_GetIter(ptr noundef %54)
  store ptr %55, ptr %14, align 8
  %56 = load ptr, ptr %14, align 8
  %57 = icmp eq ptr %56, null
  br i1 %57, label %58, label %59

58:                                               ; preds = %51
  store ptr null, ptr %7, align 8
  br label %95

59:                                               ; preds = %51
  %60 = load ptr, ptr %12, align 8
  %61 = getelementptr inbounds %struct._typeobject, ptr %60, i32 0, i32 36
  %62 = load ptr, ptr %61, align 8
  %63 = load ptr, ptr %12, align 8
  %64 = call ptr %62(ptr noundef %63, i64 noundef 0)
  store ptr %64, ptr %15, align 8
  %65 = load ptr, ptr %15, align 8
  %66 = icmp eq ptr %65, null
  br i1 %66, label %67, label %84

67:                                               ; preds = %59
  %68 = load ptr, ptr %14, align 8
  store ptr %68, ptr %6, align 8
  %69 = load ptr, ptr %6, align 8
  store ptr %69, ptr %5, align 8
  %70 = load ptr, ptr %5, align 8
  %71 = load i32, ptr %70, align 8
  %72 = icmp slt i32 %71, 0
  %73 = zext i1 %72 to i32
  %74 = icmp ne i32 %73, 0
  br i1 %74, label %75, label %76

75:                                               ; preds = %67
  br label %83

76:                                               ; preds = %67
  %77 = load ptr, ptr %6, align 8
  %78 = load i32, ptr %77, align 8
  %79 = add i32 %78, -1
  store i32 %79, ptr %77, align 8
  %80 = icmp eq i32 %79, 0
  br i1 %80, label %81, label %83

81:                                               ; preds = %76
  %82 = load ptr, ptr %6, align 8
  call void @_Py_Dealloc(ptr noundef %82) #7
  br label %83

83:                                               ; preds = %75, %76, %81
  store ptr null, ptr %7, align 8
  br label %95

84:                                               ; preds = %59
  %85 = load ptr, ptr %9, align 8
  %86 = getelementptr inbounds ptr, ptr %85, i64 0
  %87 = load ptr, ptr %86, align 8
  %88 = call ptr @_Py_NewRef(ptr noundef %87)
  %89 = load ptr, ptr %15, align 8
  %90 = getelementptr inbounds %struct.filterobject, ptr %89, i32 0, i32 1
  store ptr %88, ptr %90, align 8
  %91 = load ptr, ptr %14, align 8
  %92 = load ptr, ptr %15, align 8
  %93 = getelementptr inbounds %struct.filterobject, ptr %92, i32 0, i32 2
  store ptr %91, ptr %93, align 8
  %94 = load ptr, ptr %15, align 8
  store ptr %94, ptr %7, align 8
  br label %95

95:                                               ; preds = %84, %83, %58, %50, %37
  %96 = load ptr, ptr %7, align 8
  ret ptr %96
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @map_dealloc(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @PyObject_GC_UnTrack(ptr noundef %3)
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds %struct.mapobject, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  call void @Py_XDECREF(ptr noundef %6)
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr inbounds %struct.mapobject, ptr %7, i32 0, i32 2
  %9 = load ptr, ptr %8, align 8
  call void @Py_XDECREF(ptr noundef %9)
  %10 = load ptr, ptr %2, align 8
  %11 = call ptr @_Py_TYPE(ptr noundef %10)
  %12 = getelementptr inbounds %struct._typeobject, ptr %11, i32 0, i32 38
  %13 = load ptr, ptr %12, align 8
  %14 = load ptr, ptr %2, align 8
  call void %13(ptr noundef %14)
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @map_traverse(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  br label %10

10:                                               ; preds = %3
  %11 = load ptr, ptr %5, align 8
  %12 = getelementptr inbounds %struct.mapobject, ptr %11, i32 0, i32 1
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %27

15:                                               ; preds = %10
  %16 = load ptr, ptr %6, align 8
  %17 = load ptr, ptr %5, align 8
  %18 = getelementptr inbounds %struct.mapobject, ptr %17, i32 0, i32 1
  %19 = load ptr, ptr %18, align 8
  %20 = load ptr, ptr %7, align 8
  %21 = call i32 %16(ptr noundef %19, ptr noundef %20)
  store i32 %21, ptr %8, align 4
  %22 = load i32, ptr %8, align 4
  %23 = icmp ne i32 %22, 0
  br i1 %23, label %24, label %26

24:                                               ; preds = %15
  %25 = load i32, ptr %8, align 4
  store i32 %25, ptr %4, align 4
  br label %48

26:                                               ; preds = %15
  br label %27

27:                                               ; preds = %26, %10
  br label %28

28:                                               ; preds = %27
  br label %29

29:                                               ; preds = %28
  %30 = load ptr, ptr %5, align 8
  %31 = getelementptr inbounds %struct.mapobject, ptr %30, i32 0, i32 2
  %32 = load ptr, ptr %31, align 8
  %33 = icmp ne ptr %32, null
  br i1 %33, label %34, label %46

34:                                               ; preds = %29
  %35 = load ptr, ptr %6, align 8
  %36 = load ptr, ptr %5, align 8
  %37 = getelementptr inbounds %struct.mapobject, ptr %36, i32 0, i32 2
  %38 = load ptr, ptr %37, align 8
  %39 = load ptr, ptr %7, align 8
  %40 = call i32 %35(ptr noundef %38, ptr noundef %39)
  store i32 %40, ptr %9, align 4
  %41 = load i32, ptr %9, align 4
  %42 = icmp ne i32 %41, 0
  br i1 %42, label %43, label %45

43:                                               ; preds = %34
  %44 = load i32, ptr %9, align 4
  store i32 %44, ptr %4, align 4
  br label %48

45:                                               ; preds = %34
  br label %46

46:                                               ; preds = %45, %29
  br label %47

47:                                               ; preds = %46
  store i32 0, ptr %4, align 4
  br label %48

48:                                               ; preds = %47, %43, %24
  %49 = load i32, ptr %4, align 4
  ret i32 %49
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @map_next(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  %9 = alloca [5 x ptr], align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i64, align 8
  %14 = alloca i64, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  store ptr null, ptr %11, align 8
  %21 = call ptr @_PyThreadState_GET()
  store ptr %21, ptr %12, align 8
  %22 = load ptr, ptr %7, align 8
  %23 = getelementptr inbounds %struct.mapobject, ptr %22, i32 0, i32 1
  %24 = load ptr, ptr %23, align 8
  %25 = call i64 @PyTuple_GET_SIZE(ptr noundef %24)
  store i64 %25, ptr %13, align 8
  %26 = load i64, ptr %13, align 8
  %27 = icmp sle i64 %26, 5
  br i1 %27, label %28, label %30

28:                                               ; preds = %1
  %29 = getelementptr inbounds [5 x ptr], ptr %9, i64 0, i64 0
  store ptr %29, ptr %10, align 8
  br label %40

30:                                               ; preds = %1
  %31 = load i64, ptr %13, align 8
  %32 = mul i64 %31, 8
  %33 = call ptr @PyMem_Malloc(i64 noundef %32)
  store ptr %33, ptr %10, align 8
  %34 = load ptr, ptr %10, align 8
  %35 = icmp eq ptr %34, null
  br i1 %35, label %36, label %39

36:                                               ; preds = %30
  %37 = load ptr, ptr %12, align 8
  %38 = call ptr @_PyErr_NoMemory(ptr noundef %37)
  store ptr null, ptr %6, align 8
  br label %235

39:                                               ; preds = %30
  br label %40

40:                                               ; preds = %39, %28
  store i64 0, ptr %14, align 8
  store i64 0, ptr %8, align 8
  br label %41

41:                                               ; preds = %89, %40
  %42 = load i64, ptr %8, align 8
  %43 = load i64, ptr %13, align 8
  %44 = icmp slt i64 %42, %43
  br i1 %44, label %45, label %92

45:                                               ; preds = %41
  %46 = load ptr, ptr %7, align 8
  %47 = getelementptr inbounds %struct.mapobject, ptr %46, i32 0, i32 1
  %48 = load ptr, ptr %47, align 8
  %49 = call ptr @_Py_TYPE(ptr noundef %48)
  %50 = call i32 @PyType_HasFeature(ptr noundef %49, i64 noundef 67108864)
  %51 = icmp ne i32 %50, 0
  %52 = xor i1 %51, true
  %53 = zext i1 %52 to i32
  %54 = sext i32 %53 to i64
  %55 = icmp ne i64 %54, 0
  br i1 %55, label %56, label %58

56:                                               ; preds = %45
  call void @__assert_rtn(ptr noundef @__func__.map_next, ptr noundef @.str.34, i32 noundef 1464, ptr noundef @.str.36) #8
  unreachable

57:                                               ; No predecessors!
  br label %59

58:                                               ; preds = %45
  br label %59

59:                                               ; preds = %58, %57
  %60 = load ptr, ptr %7, align 8
  %61 = getelementptr inbounds %struct.mapobject, ptr %60, i32 0, i32 1
  %62 = load ptr, ptr %61, align 8
  %63 = getelementptr inbounds %struct.PyTupleObject, ptr %62, i32 0, i32 1
  %64 = load i64, ptr %8, align 8
  %65 = getelementptr inbounds [1 x ptr], ptr %63, i64 0, i64 %64
  %66 = load ptr, ptr %65, align 8
  store ptr %66, ptr %15, align 8
  %67 = load ptr, ptr %15, align 8
  %68 = call ptr @_Py_TYPE(ptr noundef %67)
  %69 = getelementptr inbounds %struct._typeobject, ptr %68, i32 0, i32 26
  %70 = load ptr, ptr %69, align 8
  %71 = load ptr, ptr %15, align 8
  %72 = call ptr %70(ptr noundef %71)
  store ptr %72, ptr %16, align 8
  %73 = load ptr, ptr %16, align 8
  %74 = icmp eq ptr %73, null
  br i1 %74, label %75, label %82

75:                                               ; preds = %59
  %76 = load ptr, ptr %7, align 8
  %77 = getelementptr inbounds %struct.mapobject, ptr %76, i32 0, i32 3
  %78 = load i32, ptr %77, align 8
  %79 = icmp ne i32 %78, 0
  br i1 %79, label %80, label %81

80:                                               ; preds = %75
  br label %136

81:                                               ; preds = %75
  br label %100

82:                                               ; preds = %59
  %83 = load ptr, ptr %16, align 8
  %84 = load ptr, ptr %10, align 8
  %85 = load i64, ptr %8, align 8
  %86 = getelementptr inbounds ptr, ptr %84, i64 %85
  store ptr %83, ptr %86, align 8
  %87 = load i64, ptr %14, align 8
  %88 = add nsw i64 %87, 1
  store i64 %88, ptr %14, align 8
  br label %89

89:                                               ; preds = %82
  %90 = load i64, ptr %8, align 8
  %91 = add nsw i64 %90, 1
  store i64 %91, ptr %8, align 8
  br label %41, !llvm.loop !6

92:                                               ; preds = %41
  %93 = load ptr, ptr %12, align 8
  %94 = load ptr, ptr %7, align 8
  %95 = getelementptr inbounds %struct.mapobject, ptr %94, i32 0, i32 2
  %96 = load ptr, ptr %95, align 8
  %97 = load ptr, ptr %10, align 8
  %98 = load i64, ptr %14, align 8
  %99 = call ptr @_PyObject_VectorcallTstate(ptr noundef %93, ptr noundef %96, ptr noundef %97, i64 noundef %98, ptr noundef null)
  store ptr %99, ptr %11, align 8
  br label %100

100:                                              ; preds = %234, %92, %81
  store i64 0, ptr %8, align 8
  br label %101

101:                                              ; preds = %125, %100
  %102 = load i64, ptr %8, align 8
  %103 = load i64, ptr %14, align 8
  %104 = icmp slt i64 %102, %103
  br i1 %104, label %105, label %128

105:                                              ; preds = %101
  %106 = load ptr, ptr %10, align 8
  %107 = load i64, ptr %8, align 8
  %108 = getelementptr inbounds ptr, ptr %106, i64 %107
  %109 = load ptr, ptr %108, align 8
  store ptr %109, ptr %4, align 8
  %110 = load ptr, ptr %4, align 8
  store ptr %110, ptr %3, align 8
  %111 = load ptr, ptr %3, align 8
  %112 = load i32, ptr %111, align 8
  %113 = icmp slt i32 %112, 0
  %114 = zext i1 %113 to i32
  %115 = icmp ne i32 %114, 0
  br i1 %115, label %116, label %117

116:                                              ; preds = %105
  br label %124

117:                                              ; preds = %105
  %118 = load ptr, ptr %4, align 8
  %119 = load i32, ptr %118, align 8
  %120 = add i32 %119, -1
  store i32 %120, ptr %118, align 8
  %121 = icmp eq i32 %120, 0
  br i1 %121, label %122, label %124

122:                                              ; preds = %117
  %123 = load ptr, ptr %4, align 8
  call void @_Py_Dealloc(ptr noundef %123) #7
  br label %124

124:                                              ; preds = %116, %117, %122
  br label %125

125:                                              ; preds = %124
  %126 = load i64, ptr %8, align 8
  %127 = add nsw i64 %126, 1
  store i64 %127, ptr %8, align 8
  br label %101, !llvm.loop !8

128:                                              ; preds = %101
  %129 = load ptr, ptr %10, align 8
  %130 = getelementptr inbounds [5 x ptr], ptr %9, i64 0, i64 0
  %131 = icmp ne ptr %129, %130
  br i1 %131, label %132, label %134

132:                                              ; preds = %128
  %133 = load ptr, ptr %10, align 8
  call void @PyMem_Free(ptr noundef %133)
  br label %134

134:                                              ; preds = %132, %128
  %135 = load ptr, ptr %11, align 8
  store ptr %135, ptr %6, align 8
  br label %235

136:                                              ; preds = %80
  %137 = call ptr @PyErr_Occurred()
  %138 = icmp ne ptr %137, null
  br i1 %138, label %139, label %145

139:                                              ; preds = %136
  %140 = load ptr, ptr @PyExc_StopIteration, align 8
  %141 = call i32 @PyErr_ExceptionMatches(ptr noundef %140)
  %142 = icmp ne i32 %141, 0
  br i1 %142, label %144, label %143

143:                                              ; preds = %139
  store ptr null, ptr %6, align 8
  br label %235

144:                                              ; preds = %139
  call void @PyErr_Clear()
  br label %145

145:                                              ; preds = %144, %136
  %146 = load i64, ptr %8, align 8
  %147 = icmp ne i64 %146, 0
  br i1 %147, label %148, label %159

148:                                              ; preds = %145
  %149 = load i64, ptr %8, align 8
  %150 = icmp eq i64 %149, 1
  %151 = zext i1 %150 to i64
  %152 = select i1 %150, ptr @.str.37, ptr @.str.38
  store ptr %152, ptr %17, align 8
  %153 = load ptr, ptr @PyExc_ValueError, align 8
  %154 = load i64, ptr %8, align 8
  %155 = add nsw i64 %154, 1
  %156 = load ptr, ptr %17, align 8
  %157 = load i64, ptr %8, align 8
  %158 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %153, ptr noundef @.str.39, i64 noundef %155, ptr noundef %156, i64 noundef %157)
  store ptr %158, ptr %6, align 8
  br label %235

159:                                              ; preds = %145
  store i64 1, ptr %8, align 8
  br label %160

160:                                              ; preds = %231, %159
  %161 = load i64, ptr %8, align 8
  %162 = load i64, ptr %13, align 8
  %163 = icmp slt i64 %161, %162
  br i1 %163, label %164, label %234

164:                                              ; preds = %160
  %165 = load ptr, ptr %7, align 8
  %166 = getelementptr inbounds %struct.mapobject, ptr %165, i32 0, i32 1
  %167 = load ptr, ptr %166, align 8
  %168 = call ptr @_Py_TYPE(ptr noundef %167)
  %169 = call i32 @PyType_HasFeature(ptr noundef %168, i64 noundef 67108864)
  %170 = icmp ne i32 %169, 0
  %171 = xor i1 %170, true
  %172 = zext i1 %171 to i32
  %173 = sext i32 %172 to i64
  %174 = icmp ne i64 %173, 0
  br i1 %174, label %175, label %177

175:                                              ; preds = %164
  call void @__assert_rtn(ptr noundef @__func__.map_next, ptr noundef @.str.34, i32 noundef 1503, ptr noundef @.str.36) #8
  unreachable

176:                                              ; No predecessors!
  br label %178

177:                                              ; preds = %164
  br label %178

178:                                              ; preds = %177, %176
  %179 = load ptr, ptr %7, align 8
  %180 = getelementptr inbounds %struct.mapobject, ptr %179, i32 0, i32 1
  %181 = load ptr, ptr %180, align 8
  %182 = getelementptr inbounds %struct.PyTupleObject, ptr %181, i32 0, i32 1
  %183 = load i64, ptr %8, align 8
  %184 = getelementptr inbounds [1 x ptr], ptr %182, i64 0, i64 %183
  %185 = load ptr, ptr %184, align 8
  store ptr %185, ptr %18, align 8
  %186 = load ptr, ptr %18, align 8
  %187 = call ptr @_Py_TYPE(ptr noundef %186)
  %188 = getelementptr inbounds %struct._typeobject, ptr %187, i32 0, i32 26
  %189 = load ptr, ptr %188, align 8
  %190 = load ptr, ptr %18, align 8
  %191 = call ptr %189(ptr noundef %190)
  store ptr %191, ptr %19, align 8
  %192 = load ptr, ptr %19, align 8
  %193 = icmp ne ptr %192, null
  br i1 %193, label %194, label %221

194:                                              ; preds = %178
  %195 = load ptr, ptr %19, align 8
  store ptr %195, ptr %5, align 8
  %196 = load ptr, ptr %5, align 8
  store ptr %196, ptr %2, align 8
  %197 = load ptr, ptr %2, align 8
  %198 = load i32, ptr %197, align 8
  %199 = icmp slt i32 %198, 0
  %200 = zext i1 %199 to i32
  %201 = icmp ne i32 %200, 0
  br i1 %201, label %202, label %203

202:                                              ; preds = %194
  br label %210

203:                                              ; preds = %194
  %204 = load ptr, ptr %5, align 8
  %205 = load i32, ptr %204, align 8
  %206 = add i32 %205, -1
  store i32 %206, ptr %204, align 8
  %207 = icmp eq i32 %206, 0
  br i1 %207, label %208, label %210

208:                                              ; preds = %203
  %209 = load ptr, ptr %5, align 8
  call void @_Py_Dealloc(ptr noundef %209) #7
  br label %210

210:                                              ; preds = %202, %203, %208
  %211 = load i64, ptr %8, align 8
  %212 = icmp eq i64 %211, 1
  %213 = zext i1 %212 to i64
  %214 = select i1 %212, ptr @.str.37, ptr @.str.38
  store ptr %214, ptr %20, align 8
  %215 = load ptr, ptr @PyExc_ValueError, align 8
  %216 = load i64, ptr %8, align 8
  %217 = add nsw i64 %216, 1
  %218 = load ptr, ptr %20, align 8
  %219 = load i64, ptr %8, align 8
  %220 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %215, ptr noundef @.str.40, i64 noundef %217, ptr noundef %218, i64 noundef %219)
  store ptr %220, ptr %6, align 8
  br label %235

221:                                              ; preds = %178
  %222 = call ptr @PyErr_Occurred()
  %223 = icmp ne ptr %222, null
  br i1 %223, label %224, label %230

224:                                              ; preds = %221
  %225 = load ptr, ptr @PyExc_StopIteration, align 8
  %226 = call i32 @PyErr_ExceptionMatches(ptr noundef %225)
  %227 = icmp ne i32 %226, 0
  br i1 %227, label %229, label %228

228:                                              ; preds = %224
  store ptr null, ptr %6, align 8
  br label %235

229:                                              ; preds = %224
  call void @PyErr_Clear()
  br label %230

230:                                              ; preds = %229, %221
  br label %231

231:                                              ; preds = %230
  %232 = load i64, ptr %8, align 8
  %233 = add nsw i64 %232, 1
  store i64 %233, ptr %8, align 8
  br label %160, !llvm.loop !9

234:                                              ; preds = %160
  br label %100

235:                                              ; preds = %228, %210, %148, %143, %134, %36
  %236 = load ptr, ptr %6, align 8
  ret ptr %236
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @map_new(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca i64, align 8
  %19 = alloca i64, align 8
  %20 = alloca i32, align 4
  %21 = alloca ptr, align 8
  %22 = alloca i32, align 4
  store ptr %0, ptr %11, align 8
  store ptr %1, ptr %12, align 8
  store ptr %2, ptr %13, align 8
  store i32 0, ptr %20, align 4
  %23 = load ptr, ptr %13, align 8
  %24 = icmp ne ptr %23, null
  br i1 %24, label %25, label %54

25:                                               ; preds = %3
  %26 = call ptr @PyTuple_New(i64 noundef 0)
  store ptr %26, ptr %21, align 8
  %27 = load ptr, ptr %21, align 8
  %28 = icmp eq ptr %27, null
  br i1 %28, label %29, label %30

29:                                               ; preds = %25
  store ptr null, ptr %10, align 8
  br label %172

30:                                               ; preds = %25
  %31 = load ptr, ptr %21, align 8
  %32 = load ptr, ptr %13, align 8
  %33 = call i32 (ptr, ptr, ptr, ptr, ...) @PyArg_ParseTupleAndKeywords(ptr noundef %31, ptr noundef %32, ptr noundef @.str.58, ptr noundef @map_new.kwlist, ptr noundef %20)
  store i32 %33, ptr %22, align 4
  %34 = load ptr, ptr %21, align 8
  store ptr %34, ptr %7, align 8
  %35 = load ptr, ptr %7, align 8
  store ptr %35, ptr %6, align 8
  %36 = load ptr, ptr %6, align 8
  %37 = load i32, ptr %36, align 8
  %38 = icmp slt i32 %37, 0
  %39 = zext i1 %38 to i32
  %40 = icmp ne i32 %39, 0
  br i1 %40, label %41, label %42

41:                                               ; preds = %30
  br label %49

42:                                               ; preds = %30
  %43 = load ptr, ptr %7, align 8
  %44 = load i32, ptr %43, align 8
  %45 = add i32 %44, -1
  store i32 %45, ptr %43, align 8
  %46 = icmp eq i32 %45, 0
  br i1 %46, label %47, label %49

47:                                               ; preds = %42
  %48 = load ptr, ptr %7, align 8
  call void @_Py_Dealloc(ptr noundef %48) #7
  br label %49

49:                                               ; preds = %41, %42, %47
  %50 = load i32, ptr %22, align 4
  %51 = icmp ne i32 %50, 0
  br i1 %51, label %53, label %52

52:                                               ; preds = %49
  store ptr null, ptr %10, align 8
  br label %172

53:                                               ; preds = %49
  br label %54

54:                                               ; preds = %53, %3
  %55 = load ptr, ptr %12, align 8
  %56 = call i64 @PyTuple_Size(ptr noundef %55)
  store i64 %56, ptr %18, align 8
  %57 = load i64, ptr %18, align 8
  %58 = icmp slt i64 %57, 2
  br i1 %58, label %59, label %61

59:                                               ; preds = %54
  %60 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %60, ptr noundef @.str.59)
  store ptr null, ptr %10, align 8
  br label %172

61:                                               ; preds = %54
  %62 = load i64, ptr %18, align 8
  %63 = sub nsw i64 %62, 1
  %64 = call ptr @PyTuple_New(i64 noundef %63)
  store ptr %64, ptr %15, align 8
  %65 = load ptr, ptr %15, align 8
  %66 = icmp eq ptr %65, null
  br i1 %66, label %67, label %68

67:                                               ; preds = %61
  store ptr null, ptr %10, align 8
  br label %172

68:                                               ; preds = %61
  store i64 1, ptr %19, align 8
  br label %69

69:                                               ; preds = %116, %68
  %70 = load i64, ptr %19, align 8
  %71 = load i64, ptr %18, align 8
  %72 = icmp slt i64 %70, %71
  br i1 %72, label %73, label %119

73:                                               ; preds = %69
  %74 = load ptr, ptr %12, align 8
  %75 = call ptr @_Py_TYPE(ptr noundef %74)
  %76 = call i32 @PyType_HasFeature(ptr noundef %75, i64 noundef 67108864)
  %77 = icmp ne i32 %76, 0
  %78 = xor i1 %77, true
  %79 = zext i1 %78 to i32
  %80 = sext i32 %79 to i64
  %81 = icmp ne i64 %80, 0
  br i1 %81, label %82, label %84

82:                                               ; preds = %73
  call void @__assert_rtn(ptr noundef @__func__.map_new, ptr noundef @.str.34, i32 noundef 1357, ptr noundef @.str.60) #8
  unreachable

83:                                               ; No predecessors!
  br label %85

84:                                               ; preds = %73
  br label %85

85:                                               ; preds = %84, %83
  %86 = load ptr, ptr %12, align 8
  %87 = getelementptr inbounds %struct.PyTupleObject, ptr %86, i32 0, i32 1
  %88 = load i64, ptr %19, align 8
  %89 = getelementptr inbounds [1 x ptr], ptr %87, i64 0, i64 %88
  %90 = load ptr, ptr %89, align 8
  %91 = call ptr @PyObject_GetIter(ptr noundef %90)
  store ptr %91, ptr %14, align 8
  %92 = load ptr, ptr %14, align 8
  %93 = icmp eq ptr %92, null
  br i1 %93, label %94, label %111

94:                                               ; preds = %85
  %95 = load ptr, ptr %15, align 8
  store ptr %95, ptr %8, align 8
  %96 = load ptr, ptr %8, align 8
  store ptr %96, ptr %5, align 8
  %97 = load ptr, ptr %5, align 8
  %98 = load i32, ptr %97, align 8
  %99 = icmp slt i32 %98, 0
  %100 = zext i1 %99 to i32
  %101 = icmp ne i32 %100, 0
  br i1 %101, label %102, label %103

102:                                              ; preds = %94
  br label %110

103:                                              ; preds = %94
  %104 = load ptr, ptr %8, align 8
  %105 = load i32, ptr %104, align 8
  %106 = add i32 %105, -1
  store i32 %106, ptr %104, align 8
  %107 = icmp eq i32 %106, 0
  br i1 %107, label %108, label %110

108:                                              ; preds = %103
  %109 = load ptr, ptr %8, align 8
  call void @_Py_Dealloc(ptr noundef %109) #7
  br label %110

110:                                              ; preds = %102, %103, %108
  store ptr null, ptr %10, align 8
  br label %172

111:                                              ; preds = %85
  %112 = load ptr, ptr %15, align 8
  %113 = load i64, ptr %19, align 8
  %114 = sub nsw i64 %113, 1
  %115 = load ptr, ptr %14, align 8
  call void @PyTuple_SET_ITEM(ptr noundef %112, i64 noundef %114, ptr noundef %115)
  br label %116

116:                                              ; preds = %111
  %117 = load i64, ptr %19, align 8
  %118 = add nsw i64 %117, 1
  store i64 %118, ptr %19, align 8
  br label %69, !llvm.loop !10

119:                                              ; preds = %69
  %120 = load ptr, ptr %11, align 8
  %121 = getelementptr inbounds %struct._typeobject, ptr %120, i32 0, i32 36
  %122 = load ptr, ptr %121, align 8
  %123 = load ptr, ptr %11, align 8
  %124 = call ptr %122(ptr noundef %123, i64 noundef 0)
  store ptr %124, ptr %17, align 8
  %125 = load ptr, ptr %17, align 8
  %126 = icmp eq ptr %125, null
  br i1 %126, label %127, label %144

127:                                              ; preds = %119
  %128 = load ptr, ptr %15, align 8
  store ptr %128, ptr %9, align 8
  %129 = load ptr, ptr %9, align 8
  store ptr %129, ptr %4, align 8
  %130 = load ptr, ptr %4, align 8
  %131 = load i32, ptr %130, align 8
  %132 = icmp slt i32 %131, 0
  %133 = zext i1 %132 to i32
  %134 = icmp ne i32 %133, 0
  br i1 %134, label %135, label %136

135:                                              ; preds = %127
  br label %143

136:                                              ; preds = %127
  %137 = load ptr, ptr %9, align 8
  %138 = load i32, ptr %137, align 8
  %139 = add i32 %138, -1
  store i32 %139, ptr %137, align 8
  %140 = icmp eq i32 %139, 0
  br i1 %140, label %141, label %143

141:                                              ; preds = %136
  %142 = load ptr, ptr %9, align 8
  call void @_Py_Dealloc(ptr noundef %142) #7
  br label %143

143:                                              ; preds = %135, %136, %141
  store ptr null, ptr %10, align 8
  br label %172

144:                                              ; preds = %119
  %145 = load ptr, ptr %15, align 8
  %146 = load ptr, ptr %17, align 8
  %147 = getelementptr inbounds %struct.mapobject, ptr %146, i32 0, i32 1
  store ptr %145, ptr %147, align 8
  %148 = load ptr, ptr %12, align 8
  %149 = call ptr @_Py_TYPE(ptr noundef %148)
  %150 = call i32 @PyType_HasFeature(ptr noundef %149, i64 noundef 67108864)
  %151 = icmp ne i32 %150, 0
  %152 = xor i1 %151, true
  %153 = zext i1 %152 to i32
  %154 = sext i32 %153 to i64
  %155 = icmp ne i64 %154, 0
  br i1 %155, label %156, label %158

156:                                              ; preds = %144
  call void @__assert_rtn(ptr noundef @__func__.map_new, ptr noundef @.str.34, i32 noundef 1372, ptr noundef @.str.60) #8
  unreachable

157:                                              ; No predecessors!
  br label %159

158:                                              ; preds = %144
  br label %159

159:                                              ; preds = %158, %157
  %160 = load ptr, ptr %12, align 8
  %161 = getelementptr inbounds %struct.PyTupleObject, ptr %160, i32 0, i32 1
  %162 = getelementptr inbounds [1 x ptr], ptr %161, i64 0, i64 0
  %163 = load ptr, ptr %162, align 8
  store ptr %163, ptr %16, align 8
  %164 = load ptr, ptr %16, align 8
  %165 = call ptr @_Py_NewRef(ptr noundef %164)
  %166 = load ptr, ptr %17, align 8
  %167 = getelementptr inbounds %struct.mapobject, ptr %166, i32 0, i32 2
  store ptr %165, ptr %167, align 8
  %168 = load i32, ptr %20, align 4
  %169 = load ptr, ptr %17, align 8
  %170 = getelementptr inbounds %struct.mapobject, ptr %169, i32 0, i32 3
  store i32 %168, ptr %170, align 8
  %171 = load ptr, ptr %17, align 8
  store ptr %171, ptr %10, align 8
  br label %172

172:                                              ; preds = %159, %143, %110, %67, %59, %52, %29
  %173 = load ptr, ptr %10, align 8
  ret ptr %173
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @map_vectorcall(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i64, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca i64, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca i32, align 4
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  store ptr %0, ptr %10, align 8
  store ptr %1, ptr %11, align 8
  store i64 %2, ptr %12, align 8
  store ptr %3, ptr %13, align 8
  %21 = load ptr, ptr %10, align 8
  %22 = call i32 @PyType_Check(ptr noundef %21)
  %23 = icmp ne i32 %22, 0
  %24 = xor i1 %23, true
  %25 = zext i1 %24 to i32
  %26 = sext i32 %25 to i64
  %27 = icmp ne i64 %26, 0
  br i1 %27, label %28, label %30

28:                                               ; preds = %4
  call void @__assert_rtn(ptr noundef @__func__.map_vectorcall, ptr noundef @.str.34, i32 noundef 1383, ptr noundef @.str.35) #8
  unreachable

29:                                               ; No predecessors!
  br label %31

30:                                               ; preds = %4
  br label %31

31:                                               ; preds = %30, %29
  %32 = load ptr, ptr %10, align 8
  store ptr %32, ptr %14, align 8
  %33 = load i64, ptr %12, align 8
  %34 = call i64 @_PyVectorcall_NARGS(i64 noundef %33)
  store i64 %34, ptr %15, align 8
  %35 = load ptr, ptr %13, align 8
  %36 = icmp ne ptr %35, null
  br i1 %36, label %37, label %49

37:                                               ; preds = %31
  %38 = load ptr, ptr %13, align 8
  %39 = call i64 @PyTuple_GET_SIZE(ptr noundef %38)
  %40 = icmp ne i64 %39, 0
  br i1 %40, label %41, label %49

41:                                               ; preds = %37
  %42 = call ptr @_PyThreadState_GET()
  store ptr %42, ptr %16, align 8
  %43 = load ptr, ptr %16, align 8
  %44 = load ptr, ptr %10, align 8
  %45 = load ptr, ptr %11, align 8
  %46 = load i64, ptr %15, align 8
  %47 = load ptr, ptr %13, align 8
  %48 = call ptr @_PyObject_MakeTpCall(ptr noundef %43, ptr noundef %44, ptr noundef %45, i64 noundef %46, ptr noundef %47)
  store ptr %48, ptr %9, align 8
  br label %140

49:                                               ; preds = %37, %31
  %50 = load i64, ptr %15, align 8
  %51 = icmp slt i64 %50, 2
  br i1 %51, label %52, label %54

52:                                               ; preds = %49
  %53 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %53, ptr noundef @.str.59)
  store ptr null, ptr %9, align 8
  br label %140

54:                                               ; preds = %49
  %55 = load i64, ptr %15, align 8
  %56 = sub nsw i64 %55, 1
  %57 = call ptr @PyTuple_New(i64 noundef %56)
  store ptr %57, ptr %17, align 8
  %58 = load ptr, ptr %17, align 8
  %59 = icmp eq ptr %58, null
  br i1 %59, label %60, label %61

60:                                               ; preds = %54
  store ptr null, ptr %9, align 8
  br label %140

61:                                               ; preds = %54
  store i32 1, ptr %18, align 4
  br label %62

62:                                               ; preds = %99, %61
  %63 = load i32, ptr %18, align 4
  %64 = sext i32 %63 to i64
  %65 = load i64, ptr %15, align 8
  %66 = icmp slt i64 %64, %65
  br i1 %66, label %67, label %102

67:                                               ; preds = %62
  %68 = load ptr, ptr %11, align 8
  %69 = load i32, ptr %18, align 4
  %70 = sext i32 %69 to i64
  %71 = getelementptr inbounds ptr, ptr %68, i64 %70
  %72 = load ptr, ptr %71, align 8
  %73 = call ptr @PyObject_GetIter(ptr noundef %72)
  store ptr %73, ptr %19, align 8
  %74 = load ptr, ptr %19, align 8
  %75 = icmp eq ptr %74, null
  br i1 %75, label %76, label %93

76:                                               ; preds = %67
  %77 = load ptr, ptr %17, align 8
  store ptr %77, ptr %7, align 8
  %78 = load ptr, ptr %7, align 8
  store ptr %78, ptr %6, align 8
  %79 = load ptr, ptr %6, align 8
  %80 = load i32, ptr %79, align 8
  %81 = icmp slt i32 %80, 0
  %82 = zext i1 %81 to i32
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %84, label %85

84:                                               ; preds = %76
  br label %92

85:                                               ; preds = %76
  %86 = load ptr, ptr %7, align 8
  %87 = load i32, ptr %86, align 8
  %88 = add i32 %87, -1
  store i32 %88, ptr %86, align 8
  %89 = icmp eq i32 %88, 0
  br i1 %89, label %90, label %92

90:                                               ; preds = %85
  %91 = load ptr, ptr %7, align 8
  call void @_Py_Dealloc(ptr noundef %91) #7
  br label %92

92:                                               ; preds = %84, %85, %90
  store ptr null, ptr %9, align 8
  br label %140

93:                                               ; preds = %67
  %94 = load ptr, ptr %17, align 8
  %95 = load i32, ptr %18, align 4
  %96 = sub nsw i32 %95, 1
  %97 = sext i32 %96 to i64
  %98 = load ptr, ptr %19, align 8
  call void @PyTuple_SET_ITEM(ptr noundef %94, i64 noundef %97, ptr noundef %98)
  br label %99

99:                                               ; preds = %93
  %100 = load i32, ptr %18, align 4
  %101 = add nsw i32 %100, 1
  store i32 %101, ptr %18, align 4
  br label %62, !llvm.loop !11

102:                                              ; preds = %62
  %103 = load ptr, ptr %14, align 8
  %104 = getelementptr inbounds %struct._typeobject, ptr %103, i32 0, i32 36
  %105 = load ptr, ptr %104, align 8
  %106 = load ptr, ptr %14, align 8
  %107 = call ptr %105(ptr noundef %106, i64 noundef 0)
  store ptr %107, ptr %20, align 8
  %108 = load ptr, ptr %20, align 8
  %109 = icmp eq ptr %108, null
  br i1 %109, label %110, label %127

110:                                              ; preds = %102
  %111 = load ptr, ptr %17, align 8
  store ptr %111, ptr %8, align 8
  %112 = load ptr, ptr %8, align 8
  store ptr %112, ptr %5, align 8
  %113 = load ptr, ptr %5, align 8
  %114 = load i32, ptr %113, align 8
  %115 = icmp slt i32 %114, 0
  %116 = zext i1 %115 to i32
  %117 = icmp ne i32 %116, 0
  br i1 %117, label %118, label %119

118:                                              ; preds = %110
  br label %126

119:                                              ; preds = %110
  %120 = load ptr, ptr %8, align 8
  %121 = load i32, ptr %120, align 8
  %122 = add i32 %121, -1
  store i32 %122, ptr %120, align 8
  %123 = icmp eq i32 %122, 0
  br i1 %123, label %124, label %126

124:                                              ; preds = %119
  %125 = load ptr, ptr %8, align 8
  call void @_Py_Dealloc(ptr noundef %125) #7
  br label %126

126:                                              ; preds = %118, %119, %124
  store ptr null, ptr %9, align 8
  br label %140

127:                                              ; preds = %102
  %128 = load ptr, ptr %17, align 8
  %129 = load ptr, ptr %20, align 8
  %130 = getelementptr inbounds %struct.mapobject, ptr %129, i32 0, i32 1
  store ptr %128, ptr %130, align 8
  %131 = load ptr, ptr %11, align 8
  %132 = getelementptr inbounds ptr, ptr %131, i64 0
  %133 = load ptr, ptr %132, align 8
  %134 = call ptr @_Py_NewRef(ptr noundef %133)
  %135 = load ptr, ptr %20, align 8
  %136 = getelementptr inbounds %struct.mapobject, ptr %135, i32 0, i32 2
  store ptr %134, ptr %136, align 8
  %137 = load ptr, ptr %20, align 8
  %138 = getelementptr inbounds %struct.mapobject, ptr %137, i32 0, i32 3
  store i32 0, ptr %138, align 8
  %139 = load ptr, ptr %20, align 8
  store ptr %139, ptr %9, align 8
  br label %140

140:                                              ; preds = %127, %126, %92, %60, %52, %41
  %141 = load ptr, ptr %9, align 8
  ret ptr %141
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @zip_dealloc(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @PyObject_GC_UnTrack(ptr noundef %3)
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds %struct.zipobject, ptr %4, i32 0, i32 2
  %6 = load ptr, ptr %5, align 8
  call void @Py_XDECREF(ptr noundef %6)
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr inbounds %struct.zipobject, ptr %7, i32 0, i32 3
  %9 = load ptr, ptr %8, align 8
  call void @Py_XDECREF(ptr noundef %9)
  %10 = load ptr, ptr %2, align 8
  %11 = call ptr @_Py_TYPE(ptr noundef %10)
  %12 = getelementptr inbounds %struct._typeobject, ptr %11, i32 0, i32 38
  %13 = load ptr, ptr %12, align 8
  %14 = load ptr, ptr %2, align 8
  call void %13(ptr noundef %14)
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @zip_traverse(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  br label %10

10:                                               ; preds = %3
  %11 = load ptr, ptr %5, align 8
  %12 = getelementptr inbounds %struct.zipobject, ptr %11, i32 0, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %27

15:                                               ; preds = %10
  %16 = load ptr, ptr %6, align 8
  %17 = load ptr, ptr %5, align 8
  %18 = getelementptr inbounds %struct.zipobject, ptr %17, i32 0, i32 2
  %19 = load ptr, ptr %18, align 8
  %20 = load ptr, ptr %7, align 8
  %21 = call i32 %16(ptr noundef %19, ptr noundef %20)
  store i32 %21, ptr %8, align 4
  %22 = load i32, ptr %8, align 4
  %23 = icmp ne i32 %22, 0
  br i1 %23, label %24, label %26

24:                                               ; preds = %15
  %25 = load i32, ptr %8, align 4
  store i32 %25, ptr %4, align 4
  br label %48

26:                                               ; preds = %15
  br label %27

27:                                               ; preds = %26, %10
  br label %28

28:                                               ; preds = %27
  br label %29

29:                                               ; preds = %28
  %30 = load ptr, ptr %5, align 8
  %31 = getelementptr inbounds %struct.zipobject, ptr %30, i32 0, i32 3
  %32 = load ptr, ptr %31, align 8
  %33 = icmp ne ptr %32, null
  br i1 %33, label %34, label %46

34:                                               ; preds = %29
  %35 = load ptr, ptr %6, align 8
  %36 = load ptr, ptr %5, align 8
  %37 = getelementptr inbounds %struct.zipobject, ptr %36, i32 0, i32 3
  %38 = load ptr, ptr %37, align 8
  %39 = load ptr, ptr %7, align 8
  %40 = call i32 %35(ptr noundef %38, ptr noundef %39)
  store i32 %40, ptr %9, align 4
  %41 = load i32, ptr %9, align 4
  %42 = icmp ne i32 %41, 0
  br i1 %42, label %43, label %45

43:                                               ; preds = %34
  %44 = load i32, ptr %9, align 4
  store i32 %44, ptr %4, align 4
  br label %48

45:                                               ; preds = %34
  br label %46

46:                                               ; preds = %45, %29
  br label %47

47:                                               ; preds = %46
  store i32 0, ptr %4, align 4
  br label %48

48:                                               ; preds = %47, %43, %24
  %49 = load i32, ptr %4, align 4
  ret i32 %49
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @zip_next(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca i64, align 8
  %15 = alloca i64, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  store ptr %0, ptr %13, align 8
  %22 = load ptr, ptr %13, align 8
  %23 = getelementptr inbounds %struct.zipobject, ptr %22, i32 0, i32 1
  %24 = load i64, ptr %23, align 8
  store i64 %24, ptr %15, align 8
  %25 = load ptr, ptr %13, align 8
  %26 = getelementptr inbounds %struct.zipobject, ptr %25, i32 0, i32 3
  %27 = load ptr, ptr %26, align 8
  store ptr %27, ptr %16, align 8
  %28 = load i64, ptr %15, align 8
  %29 = icmp eq i64 %28, 0
  br i1 %29, label %30, label %31

30:                                               ; preds = %1
  store ptr null, ptr %12, align 8
  br label %324

31:                                               ; preds = %1
  %32 = load ptr, ptr %16, align 8
  %33 = call i32 @_PyObject_IsUniquelyReferenced(ptr noundef %32)
  %34 = icmp ne i32 %33, 0
  br i1 %34, label %35, label %151

35:                                               ; preds = %31
  %36 = load ptr, ptr %16, align 8
  store ptr %36, ptr %6, align 8
  %37 = load ptr, ptr %6, align 8
  %38 = load i32, ptr %37, align 8
  store i32 %38, ptr %7, align 4
  %39 = load i32, ptr %7, align 4
  %40 = icmp slt i32 %39, 0
  br i1 %40, label %41, label %42

41:                                               ; preds = %35
  br label %46

42:                                               ; preds = %35
  %43 = load i32, ptr %7, align 4
  %44 = add i32 %43, 1
  %45 = load ptr, ptr %6, align 8
  store i32 %44, ptr %45, align 8
  br label %46

46:                                               ; preds = %41, %42
  store i64 0, ptr %14, align 8
  br label %47

47:                                               ; preds = %141, %46
  %48 = load i64, ptr %14, align 8
  %49 = load i64, ptr %15, align 8
  %50 = icmp slt i64 %48, %49
  br i1 %50, label %51, label %144

51:                                               ; preds = %47
  %52 = load ptr, ptr %13, align 8
  %53 = getelementptr inbounds %struct.zipobject, ptr %52, i32 0, i32 2
  %54 = load ptr, ptr %53, align 8
  %55 = call ptr @_Py_TYPE(ptr noundef %54)
  %56 = call i32 @PyType_HasFeature(ptr noundef %55, i64 noundef 67108864)
  %57 = icmp ne i32 %56, 0
  %58 = xor i1 %57, true
  %59 = zext i1 %58 to i32
  %60 = sext i32 %59 to i64
  %61 = icmp ne i64 %60, 0
  br i1 %61, label %62, label %64

62:                                               ; preds = %51
  call void @__assert_rtn(ptr noundef @__func__.zip_next, ptr noundef @.str.34, i32 noundef 3068, ptr noundef @.str.61) #8
  unreachable

63:                                               ; No predecessors!
  br label %65

64:                                               ; preds = %51
  br label %65

65:                                               ; preds = %64, %63
  %66 = load ptr, ptr %13, align 8
  %67 = getelementptr inbounds %struct.zipobject, ptr %66, i32 0, i32 2
  %68 = load ptr, ptr %67, align 8
  %69 = getelementptr inbounds %struct.PyTupleObject, ptr %68, i32 0, i32 1
  %70 = load i64, ptr %14, align 8
  %71 = getelementptr inbounds [1 x ptr], ptr %69, i64 0, i64 %70
  %72 = load ptr, ptr %71, align 8
  store ptr %72, ptr %17, align 8
  %73 = load ptr, ptr %17, align 8
  %74 = call ptr @_Py_TYPE(ptr noundef %73)
  %75 = getelementptr inbounds %struct._typeobject, ptr %74, i32 0, i32 26
  %76 = load ptr, ptr %75, align 8
  %77 = load ptr, ptr %17, align 8
  %78 = call ptr %76(ptr noundef %77)
  store ptr %78, ptr %18, align 8
  %79 = load ptr, ptr %18, align 8
  %80 = icmp eq ptr %79, null
  br i1 %80, label %81, label %104

81:                                               ; preds = %65
  %82 = load ptr, ptr %16, align 8
  store ptr %82, ptr %8, align 8
  %83 = load ptr, ptr %8, align 8
  store ptr %83, ptr %5, align 8
  %84 = load ptr, ptr %5, align 8
  %85 = load i32, ptr %84, align 8
  %86 = icmp slt i32 %85, 0
  %87 = zext i1 %86 to i32
  %88 = icmp ne i32 %87, 0
  br i1 %88, label %89, label %90

89:                                               ; preds = %81
  br label %97

90:                                               ; preds = %81
  %91 = load ptr, ptr %8, align 8
  %92 = load i32, ptr %91, align 8
  %93 = add i32 %92, -1
  store i32 %93, ptr %91, align 8
  %94 = icmp eq i32 %93, 0
  br i1 %94, label %95, label %97

95:                                               ; preds = %90
  %96 = load ptr, ptr %8, align 8
  call void @_Py_Dealloc(ptr noundef %96) #7
  br label %97

97:                                               ; preds = %89, %90, %95
  %98 = load ptr, ptr %13, align 8
  %99 = getelementptr inbounds %struct.zipobject, ptr %98, i32 0, i32 4
  %100 = load i32, ptr %99, align 8
  %101 = icmp ne i32 %100, 0
  br i1 %101, label %102, label %103

102:                                              ; preds = %97
  br label %225

103:                                              ; preds = %97
  store ptr null, ptr %12, align 8
  br label %324

104:                                              ; preds = %65
  %105 = load ptr, ptr %16, align 8
  %106 = call ptr @_Py_TYPE(ptr noundef %105)
  %107 = call i32 @PyType_HasFeature(ptr noundef %106, i64 noundef 67108864)
  %108 = icmp ne i32 %107, 0
  %109 = xor i1 %108, true
  %110 = zext i1 %109 to i32
  %111 = sext i32 %110 to i64
  %112 = icmp ne i64 %111, 0
  br i1 %112, label %113, label %115

113:                                              ; preds = %104
  call void @__assert_rtn(ptr noundef @__func__.zip_next, ptr noundef @.str.34, i32 noundef 3077, ptr noundef @.str.62) #8
  unreachable

114:                                              ; No predecessors!
  br label %116

115:                                              ; preds = %104
  br label %116

116:                                              ; preds = %115, %114
  %117 = load ptr, ptr %16, align 8
  %118 = getelementptr inbounds %struct.PyTupleObject, ptr %117, i32 0, i32 1
  %119 = load i64, ptr %14, align 8
  %120 = getelementptr inbounds [1 x ptr], ptr %118, i64 0, i64 %119
  %121 = load ptr, ptr %120, align 8
  store ptr %121, ptr %19, align 8
  %122 = load ptr, ptr %16, align 8
  %123 = load i64, ptr %14, align 8
  %124 = load ptr, ptr %18, align 8
  call void @PyTuple_SET_ITEM(ptr noundef %122, i64 noundef %123, ptr noundef %124)
  %125 = load ptr, ptr %19, align 8
  store ptr %125, ptr %9, align 8
  %126 = load ptr, ptr %9, align 8
  store ptr %126, ptr %4, align 8
  %127 = load ptr, ptr %4, align 8
  %128 = load i32, ptr %127, align 8
  %129 = icmp slt i32 %128, 0
  %130 = zext i1 %129 to i32
  %131 = icmp ne i32 %130, 0
  br i1 %131, label %132, label %133

132:                                              ; preds = %116
  br label %140

133:                                              ; preds = %116
  %134 = load ptr, ptr %9, align 8
  %135 = load i32, ptr %134, align 8
  %136 = add i32 %135, -1
  store i32 %136, ptr %134, align 8
  %137 = icmp eq i32 %136, 0
  br i1 %137, label %138, label %140

138:                                              ; preds = %133
  %139 = load ptr, ptr %9, align 8
  call void @_Py_Dealloc(ptr noundef %139) #7
  br label %140

140:                                              ; preds = %132, %133, %138
  br label %141

141:                                              ; preds = %140
  %142 = load i64, ptr %14, align 8
  %143 = add nsw i64 %142, 1
  store i64 %143, ptr %14, align 8
  br label %47, !llvm.loop !12

144:                                              ; preds = %47
  %145 = load ptr, ptr %16, align 8
  %146 = call i32 @_PyObject_GC_IS_TRACKED(ptr noundef %145)
  %147 = icmp ne i32 %146, 0
  br i1 %147, label %150, label %148

148:                                              ; preds = %144
  %149 = load ptr, ptr %16, align 8
  call void @_PyObject_GC_TRACK(ptr noundef @.str.63, i32 noundef 3084, ptr noundef %149)
  br label %150

150:                                              ; preds = %148, %144
  br label %223

151:                                              ; preds = %31
  %152 = load i64, ptr %15, align 8
  %153 = call ptr @PyTuple_New(i64 noundef %152)
  store ptr %153, ptr %16, align 8
  %154 = load ptr, ptr %16, align 8
  %155 = icmp eq ptr %154, null
  br i1 %155, label %156, label %157

156:                                              ; preds = %151
  store ptr null, ptr %12, align 8
  br label %324

157:                                              ; preds = %151
  store i64 0, ptr %14, align 8
  br label %158

158:                                              ; preds = %219, %157
  %159 = load i64, ptr %14, align 8
  %160 = load i64, ptr %15, align 8
  %161 = icmp slt i64 %159, %160
  br i1 %161, label %162, label %222

162:                                              ; preds = %158
  %163 = load ptr, ptr %13, align 8
  %164 = getelementptr inbounds %struct.zipobject, ptr %163, i32 0, i32 2
  %165 = load ptr, ptr %164, align 8
  %166 = call ptr @_Py_TYPE(ptr noundef %165)
  %167 = call i32 @PyType_HasFeature(ptr noundef %166, i64 noundef 67108864)
  %168 = icmp ne i32 %167, 0
  %169 = xor i1 %168, true
  %170 = zext i1 %169 to i32
  %171 = sext i32 %170 to i64
  %172 = icmp ne i64 %171, 0
  br i1 %172, label %173, label %175

173:                                              ; preds = %162
  call void @__assert_rtn(ptr noundef @__func__.zip_next, ptr noundef @.str.34, i32 noundef 3091, ptr noundef @.str.61) #8
  unreachable

174:                                              ; No predecessors!
  br label %176

175:                                              ; preds = %162
  br label %176

176:                                              ; preds = %175, %174
  %177 = load ptr, ptr %13, align 8
  %178 = getelementptr inbounds %struct.zipobject, ptr %177, i32 0, i32 2
  %179 = load ptr, ptr %178, align 8
  %180 = getelementptr inbounds %struct.PyTupleObject, ptr %179, i32 0, i32 1
  %181 = load i64, ptr %14, align 8
  %182 = getelementptr inbounds [1 x ptr], ptr %180, i64 0, i64 %181
  %183 = load ptr, ptr %182, align 8
  store ptr %183, ptr %17, align 8
  %184 = load ptr, ptr %17, align 8
  %185 = call ptr @_Py_TYPE(ptr noundef %184)
  %186 = getelementptr inbounds %struct._typeobject, ptr %185, i32 0, i32 26
  %187 = load ptr, ptr %186, align 8
  %188 = load ptr, ptr %17, align 8
  %189 = call ptr %187(ptr noundef %188)
  store ptr %189, ptr %18, align 8
  %190 = load ptr, ptr %18, align 8
  %191 = icmp eq ptr %190, null
  br i1 %191, label %192, label %215

192:                                              ; preds = %176
  %193 = load ptr, ptr %16, align 8
  store ptr %193, ptr %10, align 8
  %194 = load ptr, ptr %10, align 8
  store ptr %194, ptr %3, align 8
  %195 = load ptr, ptr %3, align 8
  %196 = load i32, ptr %195, align 8
  %197 = icmp slt i32 %196, 0
  %198 = zext i1 %197 to i32
  %199 = icmp ne i32 %198, 0
  br i1 %199, label %200, label %201

200:                                              ; preds = %192
  br label %208

201:                                              ; preds = %192
  %202 = load ptr, ptr %10, align 8
  %203 = load i32, ptr %202, align 8
  %204 = add i32 %203, -1
  store i32 %204, ptr %202, align 8
  %205 = icmp eq i32 %204, 0
  br i1 %205, label %206, label %208

206:                                              ; preds = %201
  %207 = load ptr, ptr %10, align 8
  call void @_Py_Dealloc(ptr noundef %207) #7
  br label %208

208:                                              ; preds = %200, %201, %206
  %209 = load ptr, ptr %13, align 8
  %210 = getelementptr inbounds %struct.zipobject, ptr %209, i32 0, i32 4
  %211 = load i32, ptr %210, align 8
  %212 = icmp ne i32 %211, 0
  br i1 %212, label %213, label %214

213:                                              ; preds = %208
  br label %225

214:                                              ; preds = %208
  store ptr null, ptr %12, align 8
  br label %324

215:                                              ; preds = %176
  %216 = load ptr, ptr %16, align 8
  %217 = load i64, ptr %14, align 8
  %218 = load ptr, ptr %18, align 8
  call void @PyTuple_SET_ITEM(ptr noundef %216, i64 noundef %217, ptr noundef %218)
  br label %219

219:                                              ; preds = %215
  %220 = load i64, ptr %14, align 8
  %221 = add nsw i64 %220, 1
  store i64 %221, ptr %14, align 8
  br label %158, !llvm.loop !13

222:                                              ; preds = %158
  br label %223

223:                                              ; preds = %222, %150
  %224 = load ptr, ptr %16, align 8
  store ptr %224, ptr %12, align 8
  br label %324

225:                                              ; preds = %213, %102
  %226 = call ptr @PyErr_Occurred()
  %227 = icmp ne ptr %226, null
  br i1 %227, label %228, label %234

228:                                              ; preds = %225
  %229 = load ptr, ptr @PyExc_StopIteration, align 8
  %230 = call i32 @PyErr_ExceptionMatches(ptr noundef %229)
  %231 = icmp ne i32 %230, 0
  br i1 %231, label %233, label %232

232:                                              ; preds = %228
  store ptr null, ptr %12, align 8
  br label %324

233:                                              ; preds = %228
  call void @PyErr_Clear()
  br label %234

234:                                              ; preds = %233, %225
  %235 = load i64, ptr %14, align 8
  %236 = icmp ne i64 %235, 0
  br i1 %236, label %237, label %248

237:                                              ; preds = %234
  %238 = load i64, ptr %14, align 8
  %239 = icmp eq i64 %238, 1
  %240 = zext i1 %239 to i64
  %241 = select i1 %239, ptr @.str.37, ptr @.str.38
  store ptr %241, ptr %20, align 8
  %242 = load ptr, ptr @PyExc_ValueError, align 8
  %243 = load i64, ptr %14, align 8
  %244 = add nsw i64 %243, 1
  %245 = load ptr, ptr %20, align 8
  %246 = load i64, ptr %14, align 8
  %247 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %242, ptr noundef @.str.64, i64 noundef %244, ptr noundef %245, i64 noundef %246)
  store ptr %247, ptr %12, align 8
  br label %324

248:                                              ; preds = %234
  store i64 1, ptr %14, align 8
  br label %249

249:                                              ; preds = %320, %248
  %250 = load i64, ptr %14, align 8
  %251 = load i64, ptr %15, align 8
  %252 = icmp slt i64 %250, %251
  br i1 %252, label %253, label %323

253:                                              ; preds = %249
  %254 = load ptr, ptr %13, align 8
  %255 = getelementptr inbounds %struct.zipobject, ptr %254, i32 0, i32 2
  %256 = load ptr, ptr %255, align 8
  %257 = call ptr @_Py_TYPE(ptr noundef %256)
  %258 = call i32 @PyType_HasFeature(ptr noundef %257, i64 noundef 67108864)
  %259 = icmp ne i32 %258, 0
  %260 = xor i1 %259, true
  %261 = zext i1 %260 to i32
  %262 = sext i32 %261 to i64
  %263 = icmp ne i64 %262, 0
  br i1 %263, label %264, label %266

264:                                              ; preds = %253
  call void @__assert_rtn(ptr noundef @__func__.zip_next, ptr noundef @.str.34, i32 noundef 3121, ptr noundef @.str.61) #8
  unreachable

265:                                              ; No predecessors!
  br label %267

266:                                              ; preds = %253
  br label %267

267:                                              ; preds = %266, %265
  %268 = load ptr, ptr %13, align 8
  %269 = getelementptr inbounds %struct.zipobject, ptr %268, i32 0, i32 2
  %270 = load ptr, ptr %269, align 8
  %271 = getelementptr inbounds %struct.PyTupleObject, ptr %270, i32 0, i32 1
  %272 = load i64, ptr %14, align 8
  %273 = getelementptr inbounds [1 x ptr], ptr %271, i64 0, i64 %272
  %274 = load ptr, ptr %273, align 8
  store ptr %274, ptr %17, align 8
  %275 = load ptr, ptr %17, align 8
  %276 = call ptr @_Py_TYPE(ptr noundef %275)
  %277 = getelementptr inbounds %struct._typeobject, ptr %276, i32 0, i32 26
  %278 = load ptr, ptr %277, align 8
  %279 = load ptr, ptr %17, align 8
  %280 = call ptr %278(ptr noundef %279)
  store ptr %280, ptr %18, align 8
  %281 = load ptr, ptr %18, align 8
  %282 = icmp ne ptr %281, null
  br i1 %282, label %283, label %310

283:                                              ; preds = %267
  %284 = load ptr, ptr %18, align 8
  store ptr %284, ptr %11, align 8
  %285 = load ptr, ptr %11, align 8
  store ptr %285, ptr %2, align 8
  %286 = load ptr, ptr %2, align 8
  %287 = load i32, ptr %286, align 8
  %288 = icmp slt i32 %287, 0
  %289 = zext i1 %288 to i32
  %290 = icmp ne i32 %289, 0
  br i1 %290, label %291, label %292

291:                                              ; preds = %283
  br label %299

292:                                              ; preds = %283
  %293 = load ptr, ptr %11, align 8
  %294 = load i32, ptr %293, align 8
  %295 = add i32 %294, -1
  store i32 %295, ptr %293, align 8
  %296 = icmp eq i32 %295, 0
  br i1 %296, label %297, label %299

297:                                              ; preds = %292
  %298 = load ptr, ptr %11, align 8
  call void @_Py_Dealloc(ptr noundef %298) #7
  br label %299

299:                                              ; preds = %291, %292, %297
  %300 = load i64, ptr %14, align 8
  %301 = icmp eq i64 %300, 1
  %302 = zext i1 %301 to i64
  %303 = select i1 %301, ptr @.str.37, ptr @.str.38
  store ptr %303, ptr %21, align 8
  %304 = load ptr, ptr @PyExc_ValueError, align 8
  %305 = load i64, ptr %14, align 8
  %306 = add nsw i64 %305, 1
  %307 = load ptr, ptr %21, align 8
  %308 = load i64, ptr %14, align 8
  %309 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %304, ptr noundef @.str.65, i64 noundef %306, ptr noundef %307, i64 noundef %308)
  store ptr %309, ptr %12, align 8
  br label %324

310:                                              ; preds = %267
  %311 = call ptr @PyErr_Occurred()
  %312 = icmp ne ptr %311, null
  br i1 %312, label %313, label %319

313:                                              ; preds = %310
  %314 = load ptr, ptr @PyExc_StopIteration, align 8
  %315 = call i32 @PyErr_ExceptionMatches(ptr noundef %314)
  %316 = icmp ne i32 %315, 0
  br i1 %316, label %318, label %317

317:                                              ; preds = %313
  store ptr null, ptr %12, align 8
  br label %324

318:                                              ; preds = %313
  call void @PyErr_Clear()
  br label %319

319:                                              ; preds = %318, %310
  br label %320

320:                                              ; preds = %319
  %321 = load i64, ptr %14, align 8
  %322 = add nsw i64 %321, 1
  store i64 %322, ptr %14, align 8
  br label %249, !llvm.loop !14

323:                                              ; preds = %249
  store ptr null, ptr %12, align 8
  br label %324

324:                                              ; preds = %323, %317, %299, %237, %232, %223, %214, %156, %103, %30
  %325 = load ptr, ptr %12, align 8
  ret ptr %325
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @zip_new(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca i64, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  %22 = alloca i64, align 8
  %23 = alloca i32, align 4
  %24 = alloca ptr, align 8
  %25 = alloca i32, align 4
  %26 = alloca ptr, align 8
  %27 = alloca ptr, align 8
  store ptr %0, ptr %15, align 8
  store ptr %1, ptr %16, align 8
  store ptr %2, ptr %17, align 8
  store i32 0, ptr %23, align 4
  %28 = load ptr, ptr %17, align 8
  %29 = icmp ne ptr %28, null
  br i1 %29, label %30, label %59

30:                                               ; preds = %3
  %31 = call ptr @PyTuple_New(i64 noundef 0)
  store ptr %31, ptr %24, align 8
  %32 = load ptr, ptr %24, align 8
  %33 = icmp eq ptr %32, null
  br i1 %33, label %34, label %35

34:                                               ; preds = %30
  store ptr null, ptr %14, align 8
  br label %219

35:                                               ; preds = %30
  %36 = load ptr, ptr %24, align 8
  %37 = load ptr, ptr %17, align 8
  %38 = call i32 (ptr, ptr, ptr, ptr, ...) @PyArg_ParseTupleAndKeywords(ptr noundef %36, ptr noundef %37, ptr noundef @.str.73, ptr noundef @zip_new.kwlist, ptr noundef %23)
  store i32 %38, ptr %25, align 4
  %39 = load ptr, ptr %24, align 8
  store ptr %39, ptr %9, align 8
  %40 = load ptr, ptr %9, align 8
  store ptr %40, ptr %8, align 8
  %41 = load ptr, ptr %8, align 8
  %42 = load i32, ptr %41, align 8
  %43 = icmp slt i32 %42, 0
  %44 = zext i1 %43 to i32
  %45 = icmp ne i32 %44, 0
  br i1 %45, label %46, label %47

46:                                               ; preds = %35
  br label %54

47:                                               ; preds = %35
  %48 = load ptr, ptr %9, align 8
  %49 = load i32, ptr %48, align 8
  %50 = add i32 %49, -1
  store i32 %50, ptr %48, align 8
  %51 = icmp eq i32 %50, 0
  br i1 %51, label %52, label %54

52:                                               ; preds = %47
  %53 = load ptr, ptr %9, align 8
  call void @_Py_Dealloc(ptr noundef %53) #7
  br label %54

54:                                               ; preds = %46, %47, %52
  %55 = load i32, ptr %25, align 4
  %56 = icmp ne i32 %55, 0
  br i1 %56, label %58, label %57

57:                                               ; preds = %54
  store ptr null, ptr %14, align 8
  br label %219

58:                                               ; preds = %54
  br label %59

59:                                               ; preds = %58, %3
  %60 = load ptr, ptr %16, align 8
  %61 = call ptr @_Py_TYPE(ptr noundef %60)
  %62 = call i32 @PyType_HasFeature(ptr noundef %61, i64 noundef 67108864)
  %63 = icmp ne i32 %62, 0
  %64 = xor i1 %63, true
  %65 = zext i1 %64 to i32
  %66 = sext i32 %65 to i64
  %67 = icmp ne i64 %66, 0
  br i1 %67, label %68, label %70

68:                                               ; preds = %59
  call void @__assert_rtn(ptr noundef @__func__.zip_new, ptr noundef @.str.34, i32 noundef 2993, ptr noundef @.str.60) #8
  unreachable

69:                                               ; No predecessors!
  br label %71

70:                                               ; preds = %59
  br label %71

71:                                               ; preds = %70, %69
  %72 = load ptr, ptr %16, align 8
  %73 = call i64 @PyTuple_GET_SIZE(ptr noundef %72)
  store i64 %73, ptr %22, align 8
  %74 = load i64, ptr %22, align 8
  %75 = call ptr @PyTuple_New(i64 noundef %74)
  store ptr %75, ptr %20, align 8
  %76 = load ptr, ptr %20, align 8
  %77 = icmp eq ptr %76, null
  br i1 %77, label %78, label %79

78:                                               ; preds = %71
  store ptr null, ptr %14, align 8
  br label %219

79:                                               ; preds = %71
  store i64 0, ptr %19, align 8
  br label %80

80:                                               ; preds = %127, %79
  %81 = load i64, ptr %19, align 8
  %82 = load i64, ptr %22, align 8
  %83 = icmp slt i64 %81, %82
  br i1 %83, label %84, label %130

84:                                               ; preds = %80
  %85 = load ptr, ptr %16, align 8
  %86 = call ptr @_Py_TYPE(ptr noundef %85)
  %87 = call i32 @PyType_HasFeature(ptr noundef %86, i64 noundef 67108864)
  %88 = icmp ne i32 %87, 0
  %89 = xor i1 %88, true
  %90 = zext i1 %89 to i32
  %91 = sext i32 %90 to i64
  %92 = icmp ne i64 %91, 0
  br i1 %92, label %93, label %95

93:                                               ; preds = %84
  call void @__assert_rtn(ptr noundef @__func__.zip_new, ptr noundef @.str.34, i32 noundef 3001, ptr noundef @.str.60) #8
  unreachable

94:                                               ; No predecessors!
  br label %96

95:                                               ; preds = %84
  br label %96

96:                                               ; preds = %95, %94
  %97 = load ptr, ptr %16, align 8
  %98 = getelementptr inbounds %struct.PyTupleObject, ptr %97, i32 0, i32 1
  %99 = load i64, ptr %19, align 8
  %100 = getelementptr inbounds [1 x ptr], ptr %98, i64 0, i64 %99
  %101 = load ptr, ptr %100, align 8
  store ptr %101, ptr %26, align 8
  %102 = load ptr, ptr %26, align 8
  %103 = call ptr @PyObject_GetIter(ptr noundef %102)
  store ptr %103, ptr %27, align 8
  %104 = load ptr, ptr %27, align 8
  %105 = icmp eq ptr %104, null
  br i1 %105, label %106, label %123

106:                                              ; preds = %96
  %107 = load ptr, ptr %20, align 8
  store ptr %107, ptr %10, align 8
  %108 = load ptr, ptr %10, align 8
  store ptr %108, ptr %7, align 8
  %109 = load ptr, ptr %7, align 8
  %110 = load i32, ptr %109, align 8
  %111 = icmp slt i32 %110, 0
  %112 = zext i1 %111 to i32
  %113 = icmp ne i32 %112, 0
  br i1 %113, label %114, label %115

114:                                              ; preds = %106
  br label %122

115:                                              ; preds = %106
  %116 = load ptr, ptr %10, align 8
  %117 = load i32, ptr %116, align 8
  %118 = add i32 %117, -1
  store i32 %118, ptr %116, align 8
  %119 = icmp eq i32 %118, 0
  br i1 %119, label %120, label %122

120:                                              ; preds = %115
  %121 = load ptr, ptr %10, align 8
  call void @_Py_Dealloc(ptr noundef %121) #7
  br label %122

122:                                              ; preds = %114, %115, %120
  store ptr null, ptr %14, align 8
  br label %219

123:                                              ; preds = %96
  %124 = load ptr, ptr %20, align 8
  %125 = load i64, ptr %19, align 8
  %126 = load ptr, ptr %27, align 8
  call void @PyTuple_SET_ITEM(ptr noundef %124, i64 noundef %125, ptr noundef %126)
  br label %127

127:                                              ; preds = %123
  %128 = load i64, ptr %19, align 8
  %129 = add nsw i64 %128, 1
  store i64 %129, ptr %19, align 8
  br label %80, !llvm.loop !15

130:                                              ; preds = %80
  %131 = load i64, ptr %22, align 8
  %132 = call ptr @PyTuple_New(i64 noundef %131)
  store ptr %132, ptr %21, align 8
  %133 = load ptr, ptr %21, align 8
  %134 = icmp eq ptr %133, null
  br i1 %134, label %135, label %152

135:                                              ; preds = %130
  %136 = load ptr, ptr %20, align 8
  store ptr %136, ptr %11, align 8
  %137 = load ptr, ptr %11, align 8
  store ptr %137, ptr %6, align 8
  %138 = load ptr, ptr %6, align 8
  %139 = load i32, ptr %138, align 8
  %140 = icmp slt i32 %139, 0
  %141 = zext i1 %140 to i32
  %142 = icmp ne i32 %141, 0
  br i1 %142, label %143, label %144

143:                                              ; preds = %135
  br label %151

144:                                              ; preds = %135
  %145 = load ptr, ptr %11, align 8
  %146 = load i32, ptr %145, align 8
  %147 = add i32 %146, -1
  store i32 %147, ptr %145, align 8
  %148 = icmp eq i32 %147, 0
  br i1 %148, label %149, label %151

149:                                              ; preds = %144
  %150 = load ptr, ptr %11, align 8
  call void @_Py_Dealloc(ptr noundef %150) #7
  br label %151

151:                                              ; preds = %143, %144, %149
  store ptr null, ptr %14, align 8
  br label %219

152:                                              ; preds = %130
  store i64 0, ptr %19, align 8
  br label %153

153:                                              ; preds = %161, %152
  %154 = load i64, ptr %19, align 8
  %155 = load i64, ptr %22, align 8
  %156 = icmp slt i64 %154, %155
  br i1 %156, label %157, label %164

157:                                              ; preds = %153
  %158 = load ptr, ptr %21, align 8
  %159 = load i64, ptr %19, align 8
  %160 = call ptr @_Py_NewRef(ptr noundef @_Py_NoneStruct)
  call void @PyTuple_SET_ITEM(ptr noundef %158, i64 noundef %159, ptr noundef %160)
  br label %161

161:                                              ; preds = %157
  %162 = load i64, ptr %19, align 8
  %163 = add nsw i64 %162, 1
  store i64 %163, ptr %19, align 8
  br label %153, !llvm.loop !16

164:                                              ; preds = %153
  %165 = load ptr, ptr %15, align 8
  %166 = getelementptr inbounds %struct._typeobject, ptr %165, i32 0, i32 36
  %167 = load ptr, ptr %166, align 8
  %168 = load ptr, ptr %15, align 8
  %169 = call ptr %167(ptr noundef %168, i64 noundef 0)
  store ptr %169, ptr %18, align 8
  %170 = load ptr, ptr %18, align 8
  %171 = icmp eq ptr %170, null
  br i1 %171, label %172, label %205

172:                                              ; preds = %164
  %173 = load ptr, ptr %20, align 8
  store ptr %173, ptr %12, align 8
  %174 = load ptr, ptr %12, align 8
  store ptr %174, ptr %5, align 8
  %175 = load ptr, ptr %5, align 8
  %176 = load i32, ptr %175, align 8
  %177 = icmp slt i32 %176, 0
  %178 = zext i1 %177 to i32
  %179 = icmp ne i32 %178, 0
  br i1 %179, label %180, label %181

180:                                              ; preds = %172
  br label %188

181:                                              ; preds = %172
  %182 = load ptr, ptr %12, align 8
  %183 = load i32, ptr %182, align 8
  %184 = add i32 %183, -1
  store i32 %184, ptr %182, align 8
  %185 = icmp eq i32 %184, 0
  br i1 %185, label %186, label %188

186:                                              ; preds = %181
  %187 = load ptr, ptr %12, align 8
  call void @_Py_Dealloc(ptr noundef %187) #7
  br label %188

188:                                              ; preds = %180, %181, %186
  %189 = load ptr, ptr %21, align 8
  store ptr %189, ptr %13, align 8
  %190 = load ptr, ptr %13, align 8
  store ptr %190, ptr %4, align 8
  %191 = load ptr, ptr %4, align 8
  %192 = load i32, ptr %191, align 8
  %193 = icmp slt i32 %192, 0
  %194 = zext i1 %193 to i32
  %195 = icmp ne i32 %194, 0
  br i1 %195, label %196, label %197

196:                                              ; preds = %188
  br label %204

197:                                              ; preds = %188
  %198 = load ptr, ptr %13, align 8
  %199 = load i32, ptr %198, align 8
  %200 = add i32 %199, -1
  store i32 %200, ptr %198, align 8
  %201 = icmp eq i32 %200, 0
  br i1 %201, label %202, label %204

202:                                              ; preds = %197
  %203 = load ptr, ptr %13, align 8
  call void @_Py_Dealloc(ptr noundef %203) #7
  br label %204

204:                                              ; preds = %196, %197, %202
  store ptr null, ptr %14, align 8
  br label %219

205:                                              ; preds = %164
  %206 = load ptr, ptr %20, align 8
  %207 = load ptr, ptr %18, align 8
  %208 = getelementptr inbounds %struct.zipobject, ptr %207, i32 0, i32 2
  store ptr %206, ptr %208, align 8
  %209 = load i64, ptr %22, align 8
  %210 = load ptr, ptr %18, align 8
  %211 = getelementptr inbounds %struct.zipobject, ptr %210, i32 0, i32 1
  store i64 %209, ptr %211, align 8
  %212 = load ptr, ptr %21, align 8
  %213 = load ptr, ptr %18, align 8
  %214 = getelementptr inbounds %struct.zipobject, ptr %213, i32 0, i32 3
  store ptr %212, ptr %214, align 8
  %215 = load i32, ptr %23, align 4
  %216 = load ptr, ptr %18, align 8
  %217 = getelementptr inbounds %struct.zipobject, ptr %216, i32 0, i32 4
  store i32 %215, ptr %217, align 8
  %218 = load ptr, ptr %18, align 8
  store ptr %218, ptr %14, align 8
  br label %219

219:                                              ; preds = %205, %204, %151, %122, %78, %57, %34
  %220 = load ptr, ptr %14, align 8
  ret ptr %220
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define ptr @_PyBuiltin_Init(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  %12 = load ptr, ptr %7, align 8
  %13 = call ptr @_PyInterpreterState_GetConfig(ptr noundef %12)
  store ptr %13, ptr %11, align 8
  %14 = call ptr @_PyModule_CreateInitialized(ptr noundef @builtinsmodule, i32 noundef 1013)
  store ptr %14, ptr %8, align 8
  %15 = load ptr, ptr %8, align 8
  %16 = icmp eq ptr %15, null
  br i1 %16, label %17, label %18

17:                                               ; preds = %1
  store ptr null, ptr %6, align 8
  br label %222

18:                                               ; preds = %1
  %19 = load ptr, ptr %8, align 8
  %20 = call ptr @PyModule_GetDict(ptr noundef %19)
  store ptr %20, ptr %9, align 8
  %21 = load ptr, ptr %9, align 8
  %22 = call i32 @PyDict_SetItemString(ptr noundef %21, ptr noundef @.str.3, ptr noundef @_Py_NoneStruct)
  %23 = icmp slt i32 %22, 0
  br i1 %23, label %24, label %25

24:                                               ; preds = %18
  store ptr null, ptr %6, align 8
  br label %222

25:                                               ; preds = %18
  %26 = load ptr, ptr %9, align 8
  %27 = call i32 @PyDict_SetItemString(ptr noundef %26, ptr noundef @.str.4, ptr noundef @_Py_EllipsisObject)
  %28 = icmp slt i32 %27, 0
  br i1 %28, label %29, label %30

29:                                               ; preds = %25
  store ptr null, ptr %6, align 8
  br label %222

30:                                               ; preds = %25
  %31 = load ptr, ptr %9, align 8
  %32 = call i32 @PyDict_SetItemString(ptr noundef %31, ptr noundef @.str.5, ptr noundef @_Py_NotImplementedStruct)
  %33 = icmp slt i32 %32, 0
  br i1 %33, label %34, label %35

34:                                               ; preds = %30
  store ptr null, ptr %6, align 8
  br label %222

35:                                               ; preds = %30
  %36 = load ptr, ptr %9, align 8
  %37 = call i32 @PyDict_SetItemString(ptr noundef %36, ptr noundef @.str.6, ptr noundef @_Py_FalseStruct)
  %38 = icmp slt i32 %37, 0
  br i1 %38, label %39, label %40

39:                                               ; preds = %35
  store ptr null, ptr %6, align 8
  br label %222

40:                                               ; preds = %35
  %41 = load ptr, ptr %9, align 8
  %42 = call i32 @PyDict_SetItemString(ptr noundef %41, ptr noundef @.str.7, ptr noundef @_Py_TrueStruct)
  %43 = icmp slt i32 %42, 0
  br i1 %43, label %44, label %45

44:                                               ; preds = %40
  store ptr null, ptr %6, align 8
  br label %222

45:                                               ; preds = %40
  %46 = load ptr, ptr %9, align 8
  %47 = call i32 @PyDict_SetItemString(ptr noundef %46, ptr noundef @.str.8, ptr noundef @PyBool_Type)
  %48 = icmp slt i32 %47, 0
  br i1 %48, label %49, label %50

49:                                               ; preds = %45
  store ptr null, ptr %6, align 8
  br label %222

50:                                               ; preds = %45
  %51 = load ptr, ptr %9, align 8
  %52 = call i32 @PyDict_SetItemString(ptr noundef %51, ptr noundef @.str.9, ptr noundef @PyMemoryView_Type)
  %53 = icmp slt i32 %52, 0
  br i1 %53, label %54, label %55

54:                                               ; preds = %50
  store ptr null, ptr %6, align 8
  br label %222

55:                                               ; preds = %50
  %56 = load ptr, ptr %9, align 8
  %57 = call i32 @PyDict_SetItemString(ptr noundef %56, ptr noundef @.str.10, ptr noundef @PyByteArray_Type)
  %58 = icmp slt i32 %57, 0
  br i1 %58, label %59, label %60

59:                                               ; preds = %55
  store ptr null, ptr %6, align 8
  br label %222

60:                                               ; preds = %55
  %61 = load ptr, ptr %9, align 8
  %62 = call i32 @PyDict_SetItemString(ptr noundef %61, ptr noundef @.str.11, ptr noundef @PyBytes_Type)
  %63 = icmp slt i32 %62, 0
  br i1 %63, label %64, label %65

64:                                               ; preds = %60
  store ptr null, ptr %6, align 8
  br label %222

65:                                               ; preds = %60
  %66 = load ptr, ptr %9, align 8
  %67 = call i32 @PyDict_SetItemString(ptr noundef %66, ptr noundef @.str.12, ptr noundef @PyClassMethod_Type)
  %68 = icmp slt i32 %67, 0
  br i1 %68, label %69, label %70

69:                                               ; preds = %65
  store ptr null, ptr %6, align 8
  br label %222

70:                                               ; preds = %65
  %71 = load ptr, ptr %9, align 8
  %72 = call i32 @PyDict_SetItemString(ptr noundef %71, ptr noundef @.str.13, ptr noundef @PyComplex_Type)
  %73 = icmp slt i32 %72, 0
  br i1 %73, label %74, label %75

74:                                               ; preds = %70
  store ptr null, ptr %6, align 8
  br label %222

75:                                               ; preds = %70
  %76 = load ptr, ptr %9, align 8
  %77 = call i32 @PyDict_SetItemString(ptr noundef %76, ptr noundef @.str.14, ptr noundef @PyDict_Type)
  %78 = icmp slt i32 %77, 0
  br i1 %78, label %79, label %80

79:                                               ; preds = %75
  store ptr null, ptr %6, align 8
  br label %222

80:                                               ; preds = %75
  %81 = load ptr, ptr %9, align 8
  %82 = call i32 @PyDict_SetItemString(ptr noundef %81, ptr noundef @.str.15, ptr noundef @PyEnum_Type)
  %83 = icmp slt i32 %82, 0
  br i1 %83, label %84, label %85

84:                                               ; preds = %80
  store ptr null, ptr %6, align 8
  br label %222

85:                                               ; preds = %80
  %86 = load ptr, ptr %9, align 8
  %87 = call i32 @PyDict_SetItemString(ptr noundef %86, ptr noundef @.str, ptr noundef @PyFilter_Type)
  %88 = icmp slt i32 %87, 0
  br i1 %88, label %89, label %90

89:                                               ; preds = %85
  store ptr null, ptr %6, align 8
  br label %222

90:                                               ; preds = %85
  %91 = load ptr, ptr %9, align 8
  %92 = call i32 @PyDict_SetItemString(ptr noundef %91, ptr noundef @.str.16, ptr noundef @PyFloat_Type)
  %93 = icmp slt i32 %92, 0
  br i1 %93, label %94, label %95

94:                                               ; preds = %90
  store ptr null, ptr %6, align 8
  br label %222

95:                                               ; preds = %90
  %96 = load ptr, ptr %9, align 8
  %97 = call i32 @PyDict_SetItemString(ptr noundef %96, ptr noundef @.str.17, ptr noundef @PyFrozenSet_Type)
  %98 = icmp slt i32 %97, 0
  br i1 %98, label %99, label %100

99:                                               ; preds = %95
  store ptr null, ptr %6, align 8
  br label %222

100:                                              ; preds = %95
  %101 = load ptr, ptr %9, align 8
  %102 = call i32 @PyDict_SetItemString(ptr noundef %101, ptr noundef @.str.18, ptr noundef @PyProperty_Type)
  %103 = icmp slt i32 %102, 0
  br i1 %103, label %104, label %105

104:                                              ; preds = %100
  store ptr null, ptr %6, align 8
  br label %222

105:                                              ; preds = %100
  %106 = load ptr, ptr %9, align 8
  %107 = call i32 @PyDict_SetItemString(ptr noundef %106, ptr noundef @.str.19, ptr noundef @PyLong_Type)
  %108 = icmp slt i32 %107, 0
  br i1 %108, label %109, label %110

109:                                              ; preds = %105
  store ptr null, ptr %6, align 8
  br label %222

110:                                              ; preds = %105
  %111 = load ptr, ptr %9, align 8
  %112 = call i32 @PyDict_SetItemString(ptr noundef %111, ptr noundef @.str.20, ptr noundef @PyList_Type)
  %113 = icmp slt i32 %112, 0
  br i1 %113, label %114, label %115

114:                                              ; preds = %110
  store ptr null, ptr %6, align 8
  br label %222

115:                                              ; preds = %110
  %116 = load ptr, ptr %9, align 8
  %117 = call i32 @PyDict_SetItemString(ptr noundef %116, ptr noundef @.str.1, ptr noundef @PyMap_Type)
  %118 = icmp slt i32 %117, 0
  br i1 %118, label %119, label %120

119:                                              ; preds = %115
  store ptr null, ptr %6, align 8
  br label %222

120:                                              ; preds = %115
  %121 = load ptr, ptr %9, align 8
  %122 = call i32 @PyDict_SetItemString(ptr noundef %121, ptr noundef @.str.21, ptr noundef @PyBaseObject_Type)
  %123 = icmp slt i32 %122, 0
  br i1 %123, label %124, label %125

124:                                              ; preds = %120
  store ptr null, ptr %6, align 8
  br label %222

125:                                              ; preds = %120
  %126 = load ptr, ptr %9, align 8
  %127 = call i32 @PyDict_SetItemString(ptr noundef %126, ptr noundef @.str.22, ptr noundef @PyRange_Type)
  %128 = icmp slt i32 %127, 0
  br i1 %128, label %129, label %130

129:                                              ; preds = %125
  store ptr null, ptr %6, align 8
  br label %222

130:                                              ; preds = %125
  %131 = load ptr, ptr %9, align 8
  %132 = call i32 @PyDict_SetItemString(ptr noundef %131, ptr noundef @.str.23, ptr noundef @PyReversed_Type)
  %133 = icmp slt i32 %132, 0
  br i1 %133, label %134, label %135

134:                                              ; preds = %130
  store ptr null, ptr %6, align 8
  br label %222

135:                                              ; preds = %130
  %136 = load ptr, ptr %9, align 8
  %137 = call i32 @PyDict_SetItemString(ptr noundef %136, ptr noundef @.str.24, ptr noundef @PySet_Type)
  %138 = icmp slt i32 %137, 0
  br i1 %138, label %139, label %140

139:                                              ; preds = %135
  store ptr null, ptr %6, align 8
  br label %222

140:                                              ; preds = %135
  %141 = load ptr, ptr %9, align 8
  %142 = call i32 @PyDict_SetItemString(ptr noundef %141, ptr noundef @.str.25, ptr noundef @PySlice_Type)
  %143 = icmp slt i32 %142, 0
  br i1 %143, label %144, label %145

144:                                              ; preds = %140
  store ptr null, ptr %6, align 8
  br label %222

145:                                              ; preds = %140
  %146 = load ptr, ptr %9, align 8
  %147 = call i32 @PyDict_SetItemString(ptr noundef %146, ptr noundef @.str.26, ptr noundef @PyStaticMethod_Type)
  %148 = icmp slt i32 %147, 0
  br i1 %148, label %149, label %150

149:                                              ; preds = %145
  store ptr null, ptr %6, align 8
  br label %222

150:                                              ; preds = %145
  %151 = load ptr, ptr %9, align 8
  %152 = call i32 @PyDict_SetItemString(ptr noundef %151, ptr noundef @.str.27, ptr noundef @PyUnicode_Type)
  %153 = icmp slt i32 %152, 0
  br i1 %153, label %154, label %155

154:                                              ; preds = %150
  store ptr null, ptr %6, align 8
  br label %222

155:                                              ; preds = %150
  %156 = load ptr, ptr %9, align 8
  %157 = call i32 @PyDict_SetItemString(ptr noundef %156, ptr noundef @.str.28, ptr noundef @PySuper_Type)
  %158 = icmp slt i32 %157, 0
  br i1 %158, label %159, label %160

159:                                              ; preds = %155
  store ptr null, ptr %6, align 8
  br label %222

160:                                              ; preds = %155
  %161 = load ptr, ptr %9, align 8
  %162 = call i32 @PyDict_SetItemString(ptr noundef %161, ptr noundef @.str.29, ptr noundef @PyTuple_Type)
  %163 = icmp slt i32 %162, 0
  br i1 %163, label %164, label %165

164:                                              ; preds = %160
  store ptr null, ptr %6, align 8
  br label %222

165:                                              ; preds = %160
  %166 = load ptr, ptr %9, align 8
  %167 = call i32 @PyDict_SetItemString(ptr noundef %166, ptr noundef @.str.30, ptr noundef @PyType_Type)
  %168 = icmp slt i32 %167, 0
  br i1 %168, label %169, label %170

169:                                              ; preds = %165
  store ptr null, ptr %6, align 8
  br label %222

170:                                              ; preds = %165
  %171 = load ptr, ptr %9, align 8
  %172 = call i32 @PyDict_SetItemString(ptr noundef %171, ptr noundef @.str.2, ptr noundef @PyZip_Type)
  %173 = icmp slt i32 %172, 0
  br i1 %173, label %174, label %175

174:                                              ; preds = %170
  store ptr null, ptr %6, align 8
  br label %222

175:                                              ; preds = %170
  %176 = load ptr, ptr %11, align 8
  %177 = getelementptr inbounds %struct.PyConfig, ptr %176, i32 0, i32 29
  %178 = load i32, ptr %177, align 4
  %179 = icmp eq i32 %178, 0
  %180 = zext i1 %179 to i32
  %181 = sext i32 %180 to i64
  %182 = call ptr @PyBool_FromLong(i64 noundef %181)
  store ptr %182, ptr %10, align 8
  %183 = load ptr, ptr %9, align 8
  %184 = load ptr, ptr %10, align 8
  %185 = call i32 @PyDict_SetItemString(ptr noundef %183, ptr noundef @.str.31, ptr noundef %184)
  %186 = icmp slt i32 %185, 0
  br i1 %186, label %187, label %204

187:                                              ; preds = %175
  %188 = load ptr, ptr %10, align 8
  store ptr %188, ptr %4, align 8
  %189 = load ptr, ptr %4, align 8
  store ptr %189, ptr %3, align 8
  %190 = load ptr, ptr %3, align 8
  %191 = load i32, ptr %190, align 8
  %192 = icmp slt i32 %191, 0
  %193 = zext i1 %192 to i32
  %194 = icmp ne i32 %193, 0
  br i1 %194, label %195, label %196

195:                                              ; preds = %187
  br label %203

196:                                              ; preds = %187
  %197 = load ptr, ptr %4, align 8
  %198 = load i32, ptr %197, align 8
  %199 = add i32 %198, -1
  store i32 %199, ptr %197, align 8
  %200 = icmp eq i32 %199, 0
  br i1 %200, label %201, label %203

201:                                              ; preds = %196
  %202 = load ptr, ptr %4, align 8
  call void @_Py_Dealloc(ptr noundef %202) #7
  br label %203

203:                                              ; preds = %195, %196, %201
  store ptr null, ptr %6, align 8
  br label %222

204:                                              ; preds = %175
  %205 = load ptr, ptr %10, align 8
  store ptr %205, ptr %5, align 8
  %206 = load ptr, ptr %5, align 8
  store ptr %206, ptr %2, align 8
  %207 = load ptr, ptr %2, align 8
  %208 = load i32, ptr %207, align 8
  %209 = icmp slt i32 %208, 0
  %210 = zext i1 %209 to i32
  %211 = icmp ne i32 %210, 0
  br i1 %211, label %212, label %213

212:                                              ; preds = %204
  br label %220

213:                                              ; preds = %204
  %214 = load ptr, ptr %5, align 8
  %215 = load i32, ptr %214, align 8
  %216 = add i32 %215, -1
  store i32 %216, ptr %214, align 8
  %217 = icmp eq i32 %216, 0
  br i1 %217, label %218, label %220

218:                                              ; preds = %213
  %219 = load ptr, ptr %5, align 8
  call void @_Py_Dealloc(ptr noundef %219) #7
  br label %220

220:                                              ; preds = %212, %213, %218
  %221 = load ptr, ptr %8, align 8
  store ptr %221, ptr %6, align 8
  br label %222

222:                                              ; preds = %220, %203, %174, %169, %164, %159, %154, %149, %144, %139, %134, %129, %124, %119, %114, %109, %104, %99, %94, %89, %84, %79, %74, %69, %64, %59, %54, %49, %44, %39, %34, %29, %24, %17
  %223 = load ptr, ptr %6, align 8
  ret ptr %223
}

declare ptr @_PyInterpreterState_GetConfig(ptr noundef) #1

declare ptr @_PyModule_CreateInitialized(ptr noundef, i32 noundef) #1

declare ptr @PyModule_GetDict(ptr noundef) #1

declare i32 @PyDict_SetItemString(ptr noundef, ptr noundef, ptr noundef) #1

declare ptr @PyBool_FromLong(i64 noundef) #1

declare void @PyObject_GC_UnTrack(ptr noundef) #1

declare ptr @PyThreadState_Get() #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_Py_TYPE(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %struct._object, ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %4, align 8
  ret ptr %5
}

declare void @_PyTrash_thread_deposit_object(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @Py_XDECREF(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = icmp ne ptr %5, null
  br i1 %6, label %7, label %24

7:                                                ; preds = %1
  %8 = load ptr, ptr %4, align 8
  store ptr %8, ptr %3, align 8
  %9 = load ptr, ptr %3, align 8
  store ptr %9, ptr %2, align 8
  %10 = load ptr, ptr %2, align 8
  %11 = load i32, ptr %10, align 8
  %12 = icmp slt i32 %11, 0
  %13 = zext i1 %12 to i32
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %16

15:                                               ; preds = %7
  br label %23

16:                                               ; preds = %7
  %17 = load ptr, ptr %3, align 8
  %18 = load i32, ptr %17, align 8
  %19 = add i32 %18, -1
  store i32 %19, ptr %17, align 8
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %21, label %23

21:                                               ; preds = %16
  %22 = load ptr, ptr %3, align 8
  call void @_Py_Dealloc(ptr noundef %22) #7
  br label %23

23:                                               ; preds = %15, %16, %21
  br label %24

24:                                               ; preds = %23, %1
  ret void
}

declare void @_PyTrash_thread_destroy_chain(ptr noundef) #1

declare i32 @PyObject_IsTrue(ptr noundef) #1

declare ptr @PyObject_CallOneArg(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @filter_reduce(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = call ptr @_Py_TYPE(ptr noundef %5)
  %7 = load ptr, ptr %3, align 8
  %8 = getelementptr inbounds %struct.filterobject, ptr %7, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = load ptr, ptr %3, align 8
  %11 = getelementptr inbounds %struct.filterobject, ptr %10, i32 0, i32 2
  %12 = load ptr, ptr %11, align 8
  %13 = call ptr (ptr, ...) @Py_BuildValue(ptr noundef @.str.33, ptr noundef %6, ptr noundef %9, ptr noundef %12)
  ret ptr %13
}

declare ptr @Py_BuildValue(ptr noundef, ...) #1

declare i32 @_PyArg_NoKeywords(ptr noundef, ptr noundef) #1

declare i32 @PyArg_UnpackTuple(ptr noundef, ptr noundef, i64 noundef, i64 noundef, ...) #1

declare ptr @PyObject_GetIter(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_Py_NewRef(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  store ptr %5, ptr %2, align 8
  %6 = load ptr, ptr %2, align 8
  %7 = load i32, ptr %6, align 8
  store i32 %7, ptr %3, align 4
  %8 = load i32, ptr %3, align 4
  %9 = icmp slt i32 %8, 0
  br i1 %9, label %10, label %11

10:                                               ; preds = %1
  br label %15

11:                                               ; preds = %1
  %12 = load i32, ptr %3, align 4
  %13 = add i32 %12, 1
  %14 = load ptr, ptr %2, align 8
  store i32 %13, ptr %14, align 8
  br label %15

15:                                               ; preds = %10, %11
  %16 = load ptr, ptr %4, align 8
  ret ptr %16
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @PyType_Check(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_Py_TYPE(ptr noundef %3)
  %5 = call i32 @PyType_HasFeature(ptr noundef %4, i64 noundef 2147483648)
  ret i32 %5
}

; Function Attrs: cold noreturn
declare void @__assert_rtn(ptr noundef, ptr noundef, i32 noundef, ptr noundef) #2

declare i32 @_PyArg_NoKwnames(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @_PyVectorcall_NARGS(i64 noundef %0) #0 {
  %2 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  %3 = load i64, ptr %2, align 8
  %4 = and i64 %3, 9223372036854775807
  ret i64 %4
}

declare i32 @_PyArg_CheckPositional(ptr noundef, i64 noundef, i64 noundef, i64 noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @PyType_HasFeature(ptr noundef %0, i64 noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = getelementptr inbounds %struct._typeobject, ptr %6, i32 0, i32 19
  %8 = load i64, ptr %7, align 8
  store i64 %8, ptr %5, align 8
  %9 = load i64, ptr %5, align 8
  %10 = load i64, ptr %4, align 8
  %11 = and i64 %9, %10
  %12 = icmp ne i64 %11, 0
  %13 = zext i1 %12 to i32
  ret i32 %13
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_PyThreadState_GET() #0 {
  %1 = call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @_Py_tss_tstate)
  %2 = load ptr, ptr %1, align 8
  ret ptr %2
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @PyTuple_GET_SIZE(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call ptr @_Py_TYPE(ptr noundef %4)
  %6 = call i32 @PyType_HasFeature(ptr noundef %5, i64 noundef 67108864)
  %7 = icmp ne i32 %6, 0
  %8 = xor i1 %7, true
  %9 = zext i1 %8 to i32
  %10 = sext i32 %9 to i64
  %11 = icmp ne i64 %10, 0
  br i1 %11, label %12, label %14

12:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyTuple_GET_SIZE, ptr noundef @.str.41, i32 noundef 22, ptr noundef @.str.42) #8
  unreachable

13:                                               ; No predecessors!
  br label %15

14:                                               ; preds = %1
  br label %15

15:                                               ; preds = %14, %13
  %16 = load ptr, ptr %2, align 8
  store ptr %16, ptr %3, align 8
  %17 = load ptr, ptr %3, align 8
  %18 = call i64 @Py_SIZE(ptr noundef %17)
  ret i64 %18
}

declare ptr @PyMem_Malloc(i64 noundef) #1

declare ptr @_PyErr_NoMemory(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_PyObject_VectorcallTstate(ptr noundef %0, ptr noundef %1, ptr noundef %2, i64 noundef %3, ptr noundef %4) #0 {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i64, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca i64, align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  store i64 %3, ptr %10, align 8
  store ptr %4, ptr %11, align 8
  %15 = load ptr, ptr %11, align 8
  %16 = icmp eq ptr %15, null
  br i1 %16, label %22, label %17

17:                                               ; preds = %5
  %18 = load ptr, ptr %11, align 8
  %19 = call ptr @_Py_TYPE(ptr noundef %18)
  %20 = call i32 @PyType_HasFeature(ptr noundef %19, i64 noundef 67108864)
  %21 = icmp ne i32 %20, 0
  br label %22

22:                                               ; preds = %17, %5
  %23 = phi i1 [ true, %5 ], [ %21, %17 ]
  %24 = xor i1 %23, true
  %25 = zext i1 %24 to i32
  %26 = sext i32 %25 to i64
  %27 = icmp ne i64 %26, 0
  br i1 %27, label %28, label %30

28:                                               ; preds = %22
  call void @__assert_rtn(ptr noundef @__func__._PyObject_VectorcallTstate, ptr noundef @.str.46, i32 noundef 159, ptr noundef @.str.47) #8
  unreachable

29:                                               ; No predecessors!
  br label %31

30:                                               ; preds = %22
  br label %31

31:                                               ; preds = %30, %29
  %32 = load ptr, ptr %9, align 8
  %33 = icmp ne ptr %32, null
  br i1 %33, label %38, label %34

34:                                               ; preds = %31
  %35 = load i64, ptr %10, align 8
  %36 = call i64 @_PyVectorcall_NARGS(i64 noundef %35)
  %37 = icmp eq i64 %36, 0
  br label %38

38:                                               ; preds = %34, %31
  %39 = phi i1 [ true, %31 ], [ %37, %34 ]
  %40 = xor i1 %39, true
  %41 = zext i1 %40 to i32
  %42 = sext i32 %41 to i64
  %43 = icmp ne i64 %42, 0
  br i1 %43, label %44, label %46

44:                                               ; preds = %38
  call void @__assert_rtn(ptr noundef @__func__._PyObject_VectorcallTstate, ptr noundef @.str.46, i32 noundef 160, ptr noundef @.str.48) #8
  unreachable

45:                                               ; No predecessors!
  br label %47

46:                                               ; preds = %38
  br label %47

47:                                               ; preds = %46, %45
  %48 = load ptr, ptr %8, align 8
  %49 = call ptr @_PyVectorcall_FunctionInline(ptr noundef %48)
  store ptr %49, ptr %12, align 8
  %50 = load ptr, ptr %12, align 8
  %51 = icmp eq ptr %50, null
  br i1 %51, label %52, label %61

52:                                               ; preds = %47
  %53 = load i64, ptr %10, align 8
  %54 = call i64 @_PyVectorcall_NARGS(i64 noundef %53)
  store i64 %54, ptr %14, align 8
  %55 = load ptr, ptr %7, align 8
  %56 = load ptr, ptr %8, align 8
  %57 = load ptr, ptr %9, align 8
  %58 = load i64, ptr %14, align 8
  %59 = load ptr, ptr %11, align 8
  %60 = call ptr @_PyObject_MakeTpCall(ptr noundef %55, ptr noundef %56, ptr noundef %57, i64 noundef %58, ptr noundef %59)
  store ptr %60, ptr %6, align 8
  br label %72

61:                                               ; preds = %47
  %62 = load ptr, ptr %12, align 8
  %63 = load ptr, ptr %8, align 8
  %64 = load ptr, ptr %9, align 8
  %65 = load i64, ptr %10, align 8
  %66 = load ptr, ptr %11, align 8
  %67 = call ptr %62(ptr noundef %63, ptr noundef %64, i64 noundef %65, ptr noundef %66)
  store ptr %67, ptr %13, align 8
  %68 = load ptr, ptr %7, align 8
  %69 = load ptr, ptr %8, align 8
  %70 = load ptr, ptr %13, align 8
  %71 = call ptr @_Py_CheckFunctionResult(ptr noundef %68, ptr noundef %69, ptr noundef %70, ptr noundef null)
  store ptr %71, ptr %6, align 8
  br label %72

72:                                               ; preds = %61, %52
  %73 = load ptr, ptr %6, align 8
  ret ptr %73
}

declare void @PyMem_Free(ptr noundef) #1

declare ptr @PyErr_Occurred() #1

declare i32 @PyErr_ExceptionMatches(ptr noundef) #1

declare void @PyErr_Clear() #1

declare ptr @PyErr_Format(ptr noundef, ptr noundef, ...) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #3

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @Py_SIZE(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_Py_TYPE(ptr noundef %3)
  %5 = icmp ne ptr %4, @PyLong_Type
  %6 = xor i1 %5, true
  %7 = zext i1 %6 to i32
  %8 = sext i32 %7 to i64
  %9 = icmp ne i64 %8, 0
  br i1 %9, label %10, label %12

10:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.Py_SIZE, ptr noundef @.str.43, i32 noundef 282, ptr noundef @.str.44) #8
  unreachable

11:                                               ; No predecessors!
  br label %13

12:                                               ; preds = %1
  br label %13

13:                                               ; preds = %12, %11
  %14 = load ptr, ptr %2, align 8
  %15 = call ptr @_Py_TYPE(ptr noundef %14)
  %16 = icmp ne ptr %15, @PyBool_Type
  %17 = xor i1 %16, true
  %18 = zext i1 %17 to i32
  %19 = sext i32 %18 to i64
  %20 = icmp ne i64 %19, 0
  br i1 %20, label %21, label %23

21:                                               ; preds = %13
  call void @__assert_rtn(ptr noundef @__func__.Py_SIZE, ptr noundef @.str.43, i32 noundef 283, ptr noundef @.str.45) #8
  unreachable

22:                                               ; No predecessors!
  br label %24

23:                                               ; preds = %13
  br label %24

24:                                               ; preds = %23, %22
  %25 = load ptr, ptr %2, align 8
  %26 = getelementptr inbounds %struct.PyVarObject, ptr %25, i32 0, i32 1
  %27 = load i64, ptr %26, align 8
  ret i64 %27
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_PyVectorcall_FunctionInline(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = icmp ne ptr %7, null
  %9 = xor i1 %8, true
  %10 = zext i1 %9 to i32
  %11 = sext i32 %10 to i64
  %12 = icmp ne i64 %11, 0
  br i1 %12, label %13, label %15

13:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__._PyVectorcall_FunctionInline, ptr noundef @.str.46, i32 noundef 116, ptr noundef @.str.49) #8
  unreachable

14:                                               ; No predecessors!
  br label %16

15:                                               ; preds = %1
  br label %16

16:                                               ; preds = %15, %14
  %17 = load ptr, ptr %3, align 8
  %18 = call ptr @_Py_TYPE(ptr noundef %17)
  store ptr %18, ptr %4, align 8
  %19 = load ptr, ptr %4, align 8
  %20 = call i32 @PyType_HasFeature(ptr noundef %19, i64 noundef 2048)
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %23, label %22

22:                                               ; preds = %16
  store ptr null, ptr %2, align 8
  br label %52

23:                                               ; preds = %16
  %24 = load ptr, ptr %3, align 8
  %25 = call i32 @PyCallable_Check(ptr noundef %24)
  %26 = icmp ne i32 %25, 0
  %27 = xor i1 %26, true
  %28 = zext i1 %27 to i32
  %29 = sext i32 %28 to i64
  %30 = icmp ne i64 %29, 0
  br i1 %30, label %31, label %33

31:                                               ; preds = %23
  call void @__assert_rtn(ptr noundef @__func__._PyVectorcall_FunctionInline, ptr noundef @.str.46, i32 noundef 122, ptr noundef @.str.50) #8
  unreachable

32:                                               ; No predecessors!
  br label %34

33:                                               ; preds = %23
  br label %34

34:                                               ; preds = %33, %32
  %35 = load ptr, ptr %4, align 8
  %36 = getelementptr inbounds %struct._typeobject, ptr %35, i32 0, i32 5
  %37 = load i64, ptr %36, align 8
  store i64 %37, ptr %5, align 8
  %38 = load i64, ptr %5, align 8
  %39 = icmp sgt i64 %38, 0
  %40 = xor i1 %39, true
  %41 = zext i1 %40 to i32
  %42 = sext i32 %41 to i64
  %43 = icmp ne i64 %42, 0
  br i1 %43, label %44, label %46

44:                                               ; preds = %34
  call void @__assert_rtn(ptr noundef @__func__._PyVectorcall_FunctionInline, ptr noundef @.str.46, i32 noundef 125, ptr noundef @.str.51) #8
  unreachable

45:                                               ; No predecessors!
  br label %47

46:                                               ; preds = %34
  br label %47

47:                                               ; preds = %46, %45
  %48 = load ptr, ptr %3, align 8
  %49 = load i64, ptr %5, align 8
  %50 = getelementptr inbounds i8, ptr %48, i64 %49
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %6, ptr align 1 %50, i64 8, i1 false)
  %51 = load ptr, ptr %6, align 8
  store ptr %51, ptr %2, align 8
  br label %52

52:                                               ; preds = %47, %22
  %53 = load ptr, ptr %2, align 8
  ret ptr %53
}

declare ptr @_PyObject_MakeTpCall(ptr noundef, ptr noundef, ptr noundef, i64 noundef, ptr noundef) #1

declare ptr @_Py_CheckFunctionResult(ptr noundef, ptr noundef, ptr noundef, ptr noundef) #1

declare i32 @PyCallable_Check(ptr noundef) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #4

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @map_reduce(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %10 = load ptr, ptr %4, align 8
  %11 = getelementptr inbounds %struct.mapobject, ptr %10, i32 0, i32 1
  %12 = load ptr, ptr %11, align 8
  %13 = call i64 @PyTuple_GET_SIZE(ptr noundef %12)
  store i64 %13, ptr %6, align 8
  %14 = load i64, ptr %6, align 8
  %15 = add nsw i64 %14, 1
  %16 = call ptr @PyTuple_New(i64 noundef %15)
  store ptr %16, ptr %7, align 8
  %17 = load ptr, ptr %7, align 8
  %18 = icmp eq ptr %17, null
  br i1 %18, label %19, label %20

19:                                               ; preds = %2
  store ptr null, ptr %3, align 8
  br label %75

20:                                               ; preds = %2
  %21 = load ptr, ptr %7, align 8
  %22 = load ptr, ptr %4, align 8
  %23 = getelementptr inbounds %struct.mapobject, ptr %22, i32 0, i32 2
  %24 = load ptr, ptr %23, align 8
  %25 = call ptr @_Py_NewRef(ptr noundef %24)
  call void @PyTuple_SET_ITEM(ptr noundef %21, i64 noundef 0, ptr noundef %25)
  store i64 0, ptr %8, align 8
  br label %26

26:                                               ; preds = %57, %20
  %27 = load i64, ptr %8, align 8
  %28 = load i64, ptr %6, align 8
  %29 = icmp slt i64 %27, %28
  br i1 %29, label %30, label %60

30:                                               ; preds = %26
  %31 = load ptr, ptr %4, align 8
  %32 = getelementptr inbounds %struct.mapobject, ptr %31, i32 0, i32 1
  %33 = load ptr, ptr %32, align 8
  %34 = call ptr @_Py_TYPE(ptr noundef %33)
  %35 = call i32 @PyType_HasFeature(ptr noundef %34, i64 noundef 67108864)
  %36 = icmp ne i32 %35, 0
  %37 = xor i1 %36, true
  %38 = zext i1 %37 to i32
  %39 = sext i32 %38 to i64
  %40 = icmp ne i64 %39, 0
  br i1 %40, label %41, label %43

41:                                               ; preds = %30
  call void @__assert_rtn(ptr noundef @__func__.map_reduce, ptr noundef @.str.34, i32 noundef 1535, ptr noundef @.str.36) #8
  unreachable

42:                                               ; No predecessors!
  br label %44

43:                                               ; preds = %30
  br label %44

44:                                               ; preds = %43, %42
  %45 = load ptr, ptr %4, align 8
  %46 = getelementptr inbounds %struct.mapobject, ptr %45, i32 0, i32 1
  %47 = load ptr, ptr %46, align 8
  %48 = getelementptr inbounds %struct.PyTupleObject, ptr %47, i32 0, i32 1
  %49 = load i64, ptr %8, align 8
  %50 = getelementptr inbounds [1 x ptr], ptr %48, i64 0, i64 %49
  %51 = load ptr, ptr %50, align 8
  store ptr %51, ptr %9, align 8
  %52 = load ptr, ptr %7, align 8
  %53 = load i64, ptr %8, align 8
  %54 = add nsw i64 %53, 1
  %55 = load ptr, ptr %9, align 8
  %56 = call ptr @_Py_NewRef(ptr noundef %55)
  call void @PyTuple_SET_ITEM(ptr noundef %52, i64 noundef %54, ptr noundef %56)
  br label %57

57:                                               ; preds = %44
  %58 = load i64, ptr %8, align 8
  %59 = add nsw i64 %58, 1
  store i64 %59, ptr %8, align 8
  br label %26, !llvm.loop !17

60:                                               ; preds = %26
  %61 = load ptr, ptr %4, align 8
  %62 = getelementptr inbounds %struct.mapobject, ptr %61, i32 0, i32 3
  %63 = load i32, ptr %62, align 8
  %64 = icmp ne i32 %63, 0
  br i1 %64, label %65, label %70

65:                                               ; preds = %60
  %66 = load ptr, ptr %4, align 8
  %67 = call ptr @_Py_TYPE(ptr noundef %66)
  %68 = load ptr, ptr %7, align 8
  %69 = call ptr (ptr, ...) @Py_BuildValue(ptr noundef @.str.53, ptr noundef %67, ptr noundef %68, ptr noundef @_Py_TrueStruct)
  store ptr %69, ptr %3, align 8
  br label %75

70:                                               ; preds = %60
  %71 = load ptr, ptr %4, align 8
  %72 = call ptr @_Py_TYPE(ptr noundef %71)
  %73 = load ptr, ptr %7, align 8
  %74 = call ptr (ptr, ...) @Py_BuildValue(ptr noundef @.str.54, ptr noundef %72, ptr noundef %73)
  store ptr %74, ptr %3, align 8
  br label %75

75:                                               ; preds = %70, %65, %19
  %76 = load ptr, ptr %3, align 8
  ret ptr %76
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @map_setstate(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = call i32 @PyObject_IsTrue(ptr noundef %7)
  store i32 %8, ptr %6, align 4
  %9 = load i32, ptr %6, align 4
  %10 = icmp slt i32 %9, 0
  br i1 %10, label %11, label %12

11:                                               ; preds = %2
  store ptr null, ptr %3, align 8
  br label %16

12:                                               ; preds = %2
  %13 = load i32, ptr %6, align 4
  %14 = load ptr, ptr %4, align 8
  %15 = getelementptr inbounds %struct.mapobject, ptr %14, i32 0, i32 3
  store i32 %13, ptr %15, align 8
  store ptr @_Py_NoneStruct, ptr %3, align 8
  br label %16

16:                                               ; preds = %12, %11
  %17 = load ptr, ptr %3, align 8
  ret ptr %17
}

declare ptr @PyTuple_New(i64 noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @PyTuple_SET_ITEM(ptr noundef %0, i64 noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store i64 %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = call ptr @_Py_TYPE(ptr noundef %8)
  %10 = call i32 @PyType_HasFeature(ptr noundef %9, i64 noundef 67108864)
  %11 = icmp ne i32 %10, 0
  %12 = xor i1 %11, true
  %13 = zext i1 %12 to i32
  %14 = sext i32 %13 to i64
  %15 = icmp ne i64 %14, 0
  br i1 %15, label %16, label %18

16:                                               ; preds = %3
  call void @__assert_rtn(ptr noundef @__func__.PyTuple_SET_ITEM, ptr noundef @.str.41, i32 noundef 32, ptr noundef @.str.42) #8
  unreachable

17:                                               ; No predecessors!
  br label %19

18:                                               ; preds = %3
  br label %19

19:                                               ; preds = %18, %17
  %20 = load ptr, ptr %4, align 8
  store ptr %20, ptr %7, align 8
  %21 = load i64, ptr %5, align 8
  %22 = icmp sle i64 0, %21
  %23 = xor i1 %22, true
  %24 = zext i1 %23 to i32
  %25 = sext i32 %24 to i64
  %26 = icmp ne i64 %25, 0
  br i1 %26, label %27, label %29

27:                                               ; preds = %19
  call void @__assert_rtn(ptr noundef @__func__.PyTuple_SET_ITEM, ptr noundef @.str.41, i32 noundef 33, ptr noundef @.str.55) #8
  unreachable

28:                                               ; No predecessors!
  br label %30

29:                                               ; preds = %19
  br label %30

30:                                               ; preds = %29, %28
  %31 = load i64, ptr %5, align 8
  %32 = load ptr, ptr %7, align 8
  %33 = call i64 @Py_SIZE(ptr noundef %32)
  %34 = icmp slt i64 %31, %33
  %35 = xor i1 %34, true
  %36 = zext i1 %35 to i32
  %37 = sext i32 %36 to i64
  %38 = icmp ne i64 %37, 0
  br i1 %38, label %39, label %41

39:                                               ; preds = %30
  call void @__assert_rtn(ptr noundef @__func__.PyTuple_SET_ITEM, ptr noundef @.str.41, i32 noundef 34, ptr noundef @.str.56) #8
  unreachable

40:                                               ; No predecessors!
  br label %42

41:                                               ; preds = %30
  br label %42

42:                                               ; preds = %41, %40
  %43 = load ptr, ptr %6, align 8
  %44 = load ptr, ptr %7, align 8
  %45 = getelementptr inbounds %struct.PyTupleObject, ptr %44, i32 0, i32 1
  %46 = load i64, ptr %5, align 8
  %47 = getelementptr inbounds [1 x ptr], ptr %45, i64 0, i64 %46
  store ptr %43, ptr %47, align 8
  ret void
}

declare i32 @PyArg_ParseTupleAndKeywords(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ...) #1

declare i64 @PyTuple_Size(ptr noundef) #1

declare void @PyErr_SetString(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @_PyObject_IsUniquelyReferenced(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call i64 @_Py_REFCNT(ptr noundef %3)
  %5 = icmp eq i64 %4, 1
  %6 = zext i1 %5 to i32
  ret i32 %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @_PyObject_GC_IS_TRACKED(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call ptr @_Py_AS_GC(ptr noundef %4)
  store ptr %5, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = getelementptr inbounds %struct.PyGC_Head, ptr %6, i32 0, i32 0
  %8 = load i64, ptr %7, align 8
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  ret i32 %10
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @_PyObject_GC_TRACK(ptr noundef %0, i32 noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca i64, align 8
  store ptr %0, ptr %4, align 8
  store i32 %1, ptr %5, align 4
  store ptr %2, ptr %6, align 8
  %12 = load ptr, ptr %6, align 8
  %13 = call i32 @_PyObject_GC_IS_TRACKED(ptr noundef %12)
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %16, label %15

15:                                               ; preds = %3
  br label %21

16:                                               ; preds = %3
  %17 = load ptr, ptr %6, align 8
  %18 = load ptr, ptr %4, align 8
  %19 = load i32, ptr %5, align 4
  call void @_PyObject_AssertFailed(ptr noundef %17, ptr noundef @.str.66, ptr noundef @.str.67, ptr noundef %18, i32 noundef %19, ptr noundef @__func__._PyObject_GC_TRACK) #9
  unreachable

20:                                               ; No predecessors!
  br label %21

21:                                               ; preds = %20, %15
  %22 = load ptr, ptr %6, align 8
  %23 = call ptr @_Py_AS_GC(ptr noundef %22)
  store ptr %23, ptr %7, align 8
  %24 = load ptr, ptr %7, align 8
  %25 = getelementptr inbounds %struct.PyGC_Head, ptr %24, i32 0, i32 1
  %26 = load i64, ptr %25, align 8
  %27 = and i64 %26, 2
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %29, label %30

29:                                               ; preds = %21
  br label %35

30:                                               ; preds = %21
  %31 = load ptr, ptr %6, align 8
  %32 = load ptr, ptr %4, align 8
  %33 = load i32, ptr %5, align 4
  call void @_PyObject_AssertFailed(ptr noundef %31, ptr noundef @.str.68, ptr noundef @.str.69, ptr noundef %32, i32 noundef %33, ptr noundef @__func__._PyObject_GC_TRACK) #9
  unreachable

34:                                               ; No predecessors!
  br label %35

35:                                               ; preds = %34, %29
  %36 = call ptr @_PyInterpreterState_GET()
  store ptr %36, ptr %8, align 8
  %37 = load ptr, ptr %8, align 8
  %38 = getelementptr inbounds %struct._is, ptr %37, i32 0, i32 14
  %39 = getelementptr inbounds %struct._gc_runtime_state, ptr %38, i32 0, i32 4
  %40 = getelementptr inbounds %struct.gc_generation, ptr %39, i32 0, i32 0
  store ptr %40, ptr %9, align 8
  %41 = load ptr, ptr %9, align 8
  %42 = getelementptr inbounds %struct.PyGC_Head, ptr %41, i32 0, i32 1
  %43 = load i64, ptr %42, align 8
  %44 = inttoptr i64 %43 to ptr
  store ptr %44, ptr %10, align 8
  %45 = load ptr, ptr %10, align 8
  %46 = load ptr, ptr %7, align 8
  call void @_PyGCHead_SET_NEXT(ptr noundef %45, ptr noundef %46)
  %47 = load ptr, ptr %7, align 8
  %48 = load ptr, ptr %10, align 8
  call void @_PyGCHead_SET_PREV(ptr noundef %47, ptr noundef %48)
  %49 = load ptr, ptr %8, align 8
  %50 = getelementptr inbounds %struct._is, ptr %49, i32 0, i32 14
  %51 = getelementptr inbounds %struct._gc_runtime_state, ptr %50, i32 0, i32 13
  %52 = load i32, ptr %51, align 8
  %53 = xor i32 1, %52
  %54 = sext i32 %53 to i64
  store i64 %54, ptr %11, align 8
  %55 = load ptr, ptr %9, align 8
  %56 = ptrtoint ptr %55 to i64
  %57 = load i64, ptr %11, align 8
  %58 = or i64 %56, %57
  %59 = load ptr, ptr %7, align 8
  %60 = getelementptr inbounds %struct.PyGC_Head, ptr %59, i32 0, i32 0
  store i64 %58, ptr %60, align 8
  %61 = load ptr, ptr %7, align 8
  %62 = ptrtoint ptr %61 to i64
  %63 = load ptr, ptr %9, align 8
  %64 = getelementptr inbounds %struct.PyGC_Head, ptr %63, i32 0, i32 1
  store i64 %62, ptr %64, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @_Py_REFCNT(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %struct._object, ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds %struct.anon, ptr %4, i32 0, i32 0
  %6 = load i32, ptr %5, align 8
  %7 = zext i32 %6 to i64
  ret i64 %7
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_Py_AS_GC(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds i8, ptr %4, i64 -16
  store ptr %5, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  ret ptr %6
}

; Function Attrs: noreturn
declare void @_PyObject_AssertFailed(ptr noundef, ptr noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef) #5

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_PyInterpreterState_GET() #0 {
  %1 = alloca ptr, align 8
  %2 = call ptr @_PyThreadState_GET()
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = getelementptr inbounds %struct._ts, ptr %3, i32 0, i32 2
  %5 = load ptr, ptr %4, align 8
  ret ptr %5
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @_PyGCHead_SET_NEXT(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = ptrtoint ptr %6 to i64
  store i64 %7, ptr %5, align 8
  %8 = load i64, ptr %5, align 8
  %9 = and i64 %8, 3
  %10 = icmp eq i64 %9, 0
  %11 = xor i1 %10, true
  %12 = zext i1 %11 to i32
  %13 = sext i32 %12 to i64
  %14 = icmp ne i64 %13, 0
  br i1 %14, label %15, label %17

15:                                               ; preds = %2
  call void @__assert_rtn(ptr noundef @__func__._PyGCHead_SET_NEXT, ptr noundef @.str.70, i32 noundef 175, ptr noundef @.str.71) #8
  unreachable

16:                                               ; No predecessors!
  br label %18

17:                                               ; preds = %2
  br label %18

18:                                               ; preds = %17, %16
  %19 = load ptr, ptr %3, align 8
  %20 = getelementptr inbounds %struct.PyGC_Head, ptr %19, i32 0, i32 0
  %21 = load i64, ptr %20, align 8
  %22 = and i64 %21, 3
  %23 = load i64, ptr %5, align 8
  %24 = or i64 %22, %23
  %25 = load ptr, ptr %3, align 8
  %26 = getelementptr inbounds %struct.PyGC_Head, ptr %25, i32 0, i32 0
  store i64 %24, ptr %26, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @_PyGCHead_SET_PREV(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = ptrtoint ptr %6 to i64
  store i64 %7, ptr %5, align 8
  %8 = load i64, ptr %5, align 8
  %9 = and i64 %8, 3
  %10 = icmp eq i64 %9, 0
  %11 = xor i1 %10, true
  %12 = zext i1 %11 to i32
  %13 = sext i32 %12 to i64
  %14 = icmp ne i64 %13, 0
  br i1 %14, label %15, label %17

15:                                               ; preds = %2
  call void @__assert_rtn(ptr noundef @__func__._PyGCHead_SET_PREV, ptr noundef @.str.70, i32 noundef 187, ptr noundef @.str.72) #8
  unreachable

16:                                               ; No predecessors!
  br label %18

17:                                               ; preds = %2
  br label %18

18:                                               ; preds = %17, %16
  %19 = load ptr, ptr %3, align 8
  %20 = getelementptr inbounds %struct.PyGC_Head, ptr %19, i32 0, i32 1
  %21 = load i64, ptr %20, align 8
  %22 = and i64 %21, 3
  %23 = load i64, ptr %5, align 8
  %24 = or i64 %22, %23
  %25 = load ptr, ptr %3, align 8
  %26 = getelementptr inbounds %struct.PyGC_Head, ptr %25, i32 0, i32 1
  store i64 %24, ptr %26, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @zip_reduce(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds %struct.zipobject, ptr %6, i32 0, i32 4
  %8 = load i32, ptr %7, align 8
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %10, label %17

10:                                               ; preds = %2
  %11 = load ptr, ptr %4, align 8
  %12 = call ptr @_Py_TYPE(ptr noundef %11)
  %13 = load ptr, ptr %4, align 8
  %14 = getelementptr inbounds %struct.zipobject, ptr %13, i32 0, i32 2
  %15 = load ptr, ptr %14, align 8
  %16 = call ptr (i64, ...) @PyTuple_Pack(i64 noundef 3, ptr noundef %12, ptr noundef %15, ptr noundef @_Py_TrueStruct)
  store ptr %16, ptr %3, align 8
  br label %24

17:                                               ; preds = %2
  %18 = load ptr, ptr %4, align 8
  %19 = call ptr @_Py_TYPE(ptr noundef %18)
  %20 = load ptr, ptr %4, align 8
  %21 = getelementptr inbounds %struct.zipobject, ptr %20, i32 0, i32 2
  %22 = load ptr, ptr %21, align 8
  %23 = call ptr (i64, ...) @PyTuple_Pack(i64 noundef 2, ptr noundef %19, ptr noundef %22)
  store ptr %23, ptr %3, align 8
  br label %24

24:                                               ; preds = %17, %10
  %25 = load ptr, ptr %3, align 8
  ret ptr %25
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @zip_setstate(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = call i32 @PyObject_IsTrue(ptr noundef %7)
  store i32 %8, ptr %6, align 4
  %9 = load i32, ptr %6, align 4
  %10 = icmp slt i32 %9, 0
  br i1 %10, label %11, label %12

11:                                               ; preds = %2
  store ptr null, ptr %3, align 8
  br label %16

12:                                               ; preds = %2
  %13 = load i32, ptr %6, align 4
  %14 = load ptr, ptr %4, align 8
  %15 = getelementptr inbounds %struct.zipobject, ptr %14, i32 0, i32 4
  store i32 %13, ptr %15, align 8
  store ptr @_Py_NoneStruct, ptr %3, align 8
  br label %16

16:                                               ; preds = %12, %11
  %17 = load ptr, ptr %3, align 8
  ret ptr %17
}

declare ptr @PyTuple_Pack(i64 noundef, ...) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin___build_class__(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i32, align 4
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca i64, align 8
  %25 = alloca ptr, align 8
  %26 = alloca ptr, align 8
  %27 = alloca ptr, align 8
  %28 = alloca ptr, align 8
  %29 = alloca ptr, align 8
  %30 = alloca ptr, align 8
  %31 = alloca ptr, align 8
  %32 = alloca ptr, align 8
  %33 = alloca ptr, align 8
  %34 = alloca ptr, align 8
  %35 = alloca ptr, align 8
  %36 = alloca ptr, align 8
  %37 = alloca i32, align 4
  %38 = alloca ptr, align 8
  %39 = alloca ptr, align 8
  %40 = alloca ptr, align 8
  %41 = alloca [2 x ptr], align 8
  %42 = alloca ptr, align 8
  %43 = alloca [3 x ptr], align 8
  %44 = alloca ptr, align 8
  %45 = alloca ptr, align 8
  %46 = alloca ptr, align 8
  %47 = alloca ptr, align 8
  %48 = alloca ptr, align 8
  store ptr %0, ptr %22, align 8
  store ptr %1, ptr %23, align 8
  store i64 %2, ptr %24, align 8
  store ptr %3, ptr %25, align 8
  store ptr null, ptr %30, align 8
  store ptr null, ptr %31, align 8
  store ptr null, ptr %32, align 8
  store ptr null, ptr %33, align 8
  store ptr null, ptr %34, align 8
  store ptr null, ptr %35, align 8
  store ptr null, ptr %36, align 8
  store i32 0, ptr %37, align 4
  %49 = load i64, ptr %24, align 8
  %50 = icmp slt i64 %49, 2
  br i1 %50, label %51, label %53

51:                                               ; preds = %4
  %52 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %52, ptr noundef @.str.119)
  store ptr null, ptr %21, align 8
  br label %416

53:                                               ; preds = %4
  %54 = load ptr, ptr %23, align 8
  %55 = getelementptr inbounds ptr, ptr %54, i64 0
  %56 = load ptr, ptr %55, align 8
  store ptr %56, ptr %26, align 8
  %57 = load ptr, ptr %26, align 8
  %58 = call i32 @Py_IS_TYPE(ptr noundef %57, ptr noundef @PyFunction_Type)
  %59 = icmp ne i32 %58, 0
  br i1 %59, label %62, label %60

60:                                               ; preds = %53
  %61 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %61, ptr noundef @.str.120)
  store ptr null, ptr %21, align 8
  br label %416

62:                                               ; preds = %53
  %63 = load ptr, ptr %23, align 8
  %64 = getelementptr inbounds ptr, ptr %63, i64 1
  %65 = load ptr, ptr %64, align 8
  store ptr %65, ptr %27, align 8
  %66 = load ptr, ptr %27, align 8
  %67 = call ptr @_Py_TYPE(ptr noundef %66)
  %68 = call i32 @PyType_HasFeature(ptr noundef %67, i64 noundef 268435456)
  %69 = icmp ne i32 %68, 0
  br i1 %69, label %72, label %70

70:                                               ; preds = %62
  %71 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %71, ptr noundef @.str.121)
  store ptr null, ptr %21, align 8
  br label %416

72:                                               ; preds = %62
  %73 = load ptr, ptr %23, align 8
  %74 = getelementptr inbounds ptr, ptr %73, i64 2
  %75 = load i64, ptr %24, align 8
  %76 = sub nsw i64 %75, 2
  %77 = call ptr @_PyTuple_FromArray(ptr noundef %74, i64 noundef %76)
  store ptr %77, ptr %34, align 8
  %78 = load ptr, ptr %34, align 8
  %79 = icmp eq ptr %78, null
  br i1 %79, label %80, label %81

80:                                               ; preds = %72
  store ptr null, ptr %21, align 8
  br label %416

81:                                               ; preds = %72
  %82 = load ptr, ptr %34, align 8
  %83 = load ptr, ptr %23, align 8
  %84 = getelementptr inbounds ptr, ptr %83, i64 2
  %85 = load i64, ptr %24, align 8
  %86 = sub nsw i64 %85, 2
  %87 = call ptr @update_bases(ptr noundef %82, ptr noundef %84, i64 noundef %86)
  store ptr %87, ptr %36, align 8
  %88 = load ptr, ptr %36, align 8
  %89 = icmp eq ptr %88, null
  br i1 %89, label %90, label %107

90:                                               ; preds = %81
  %91 = load ptr, ptr %34, align 8
  store ptr %91, ptr %14, align 8
  %92 = load ptr, ptr %14, align 8
  store ptr %92, ptr %11, align 8
  %93 = load ptr, ptr %11, align 8
  %94 = load i32, ptr %93, align 8
  %95 = icmp slt i32 %94, 0
  %96 = zext i1 %95 to i32
  %97 = icmp ne i32 %96, 0
  br i1 %97, label %98, label %99

98:                                               ; preds = %90
  br label %106

99:                                               ; preds = %90
  %100 = load ptr, ptr %14, align 8
  %101 = load i32, ptr %100, align 8
  %102 = add i32 %101, -1
  store i32 %102, ptr %100, align 8
  %103 = icmp eq i32 %102, 0
  br i1 %103, label %104, label %106

104:                                              ; preds = %99
  %105 = load ptr, ptr %14, align 8
  call void @_Py_Dealloc(ptr noundef %105) #7
  br label %106

106:                                              ; preds = %98, %99, %104
  store ptr null, ptr %21, align 8
  br label %416

107:                                              ; preds = %81
  %108 = load ptr, ptr %25, align 8
  %109 = icmp eq ptr %108, null
  br i1 %109, label %110, label %111

110:                                              ; preds = %107
  store ptr null, ptr %33, align 8
  store ptr null, ptr %35, align 8
  br label %132

111:                                              ; preds = %107
  %112 = load ptr, ptr %23, align 8
  %113 = load i64, ptr %24, align 8
  %114 = getelementptr inbounds ptr, ptr %112, i64 %113
  %115 = load ptr, ptr %25, align 8
  %116 = call ptr @_PyStack_AsDict(ptr noundef %114, ptr noundef %115)
  store ptr %116, ptr %35, align 8
  %117 = load ptr, ptr %35, align 8
  %118 = icmp eq ptr %117, null
  br i1 %118, label %119, label %120

119:                                              ; preds = %111
  br label %373

120:                                              ; preds = %111
  %121 = load ptr, ptr %35, align 8
  %122 = call i32 @PyDict_Pop(ptr noundef %121, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 505), ptr noundef %33)
  %123 = icmp slt i32 %122, 0
  br i1 %123, label %124, label %125

124:                                              ; preds = %120
  br label %373

125:                                              ; preds = %120
  %126 = load ptr, ptr %33, align 8
  %127 = icmp ne ptr %126, null
  br i1 %127, label %128, label %131

128:                                              ; preds = %125
  %129 = load ptr, ptr %33, align 8
  %130 = call i32 @PyType_Check(ptr noundef %129)
  store i32 %130, ptr %37, align 4
  br label %131

131:                                              ; preds = %128, %125
  br label %132

132:                                              ; preds = %131, %110
  %133 = load ptr, ptr %33, align 8
  %134 = icmp eq ptr %133, null
  br i1 %134, label %135, label %171

135:                                              ; preds = %132
  %136 = load ptr, ptr %36, align 8
  %137 = call i64 @PyTuple_GET_SIZE(ptr noundef %136)
  %138 = icmp eq i64 %137, 0
  br i1 %138, label %139, label %140

139:                                              ; preds = %135
  store ptr @PyType_Type, ptr %33, align 8
  br label %159

140:                                              ; preds = %135
  %141 = load ptr, ptr %36, align 8
  %142 = call ptr @_Py_TYPE(ptr noundef %141)
  %143 = call i32 @PyType_HasFeature(ptr noundef %142, i64 noundef 67108864)
  %144 = icmp ne i32 %143, 0
  %145 = xor i1 %144, true
  %146 = zext i1 %145 to i32
  %147 = sext i32 %146 to i64
  %148 = icmp ne i64 %147, 0
  br i1 %148, label %149, label %151

149:                                              ; preds = %140
  call void @__assert_rtn(ptr noundef @__func__.builtin___build_class__, ptr noundef @.str.34, i32 noundef 159, ptr noundef @.str.122) #8
  unreachable

150:                                              ; No predecessors!
  br label %152

151:                                              ; preds = %140
  br label %152

152:                                              ; preds = %151, %150
  %153 = load ptr, ptr %36, align 8
  %154 = getelementptr inbounds %struct.PyTupleObject, ptr %153, i32 0, i32 1
  %155 = getelementptr inbounds [1 x ptr], ptr %154, i64 0, i64 0
  %156 = load ptr, ptr %155, align 8
  store ptr %156, ptr %38, align 8
  %157 = load ptr, ptr %38, align 8
  %158 = call ptr @_Py_TYPE(ptr noundef %157)
  store ptr %158, ptr %33, align 8
  br label %159

159:                                              ; preds = %152, %139
  %160 = load ptr, ptr %33, align 8
  store ptr %160, ptr %12, align 8
  %161 = load ptr, ptr %12, align 8
  %162 = load i32, ptr %161, align 8
  store i32 %162, ptr %13, align 4
  %163 = load i32, ptr %13, align 4
  %164 = icmp slt i32 %163, 0
  br i1 %164, label %165, label %166

165:                                              ; preds = %159
  br label %170

166:                                              ; preds = %159
  %167 = load i32, ptr %13, align 4
  %168 = add i32 %167, 1
  %169 = load ptr, ptr %12, align 8
  store i32 %168, ptr %169, align 8
  br label %170

170:                                              ; preds = %165, %166
  store i32 1, ptr %37, align 4
  br label %171

171:                                              ; preds = %170, %132
  %172 = load i32, ptr %37, align 4
  %173 = icmp ne i32 %172, 0
  br i1 %173, label %174, label %210

174:                                              ; preds = %171
  %175 = load ptr, ptr %33, align 8
  %176 = load ptr, ptr %36, align 8
  %177 = call ptr @_PyType_CalculateMetaclass(ptr noundef %175, ptr noundef %176)
  store ptr %177, ptr %28, align 8
  %178 = load ptr, ptr %28, align 8
  %179 = icmp eq ptr %178, null
  br i1 %179, label %180, label %181

180:                                              ; preds = %174
  br label %373

181:                                              ; preds = %174
  %182 = load ptr, ptr %28, align 8
  %183 = load ptr, ptr %33, align 8
  %184 = icmp ne ptr %182, %183
  br i1 %184, label %185, label %209

185:                                              ; preds = %181
  br label %186

186:                                              ; preds = %185
  store ptr %33, ptr %39, align 8
  %187 = load ptr, ptr %39, align 8
  %188 = load ptr, ptr %187, align 8
  store ptr %188, ptr %40, align 8
  %189 = load ptr, ptr %28, align 8
  %190 = call ptr @_Py_NewRef(ptr noundef %189)
  %191 = load ptr, ptr %39, align 8
  store ptr %190, ptr %191, align 8
  %192 = load ptr, ptr %40, align 8
  store ptr %192, ptr %15, align 8
  %193 = load ptr, ptr %15, align 8
  store ptr %193, ptr %10, align 8
  %194 = load ptr, ptr %10, align 8
  %195 = load i32, ptr %194, align 8
  %196 = icmp slt i32 %195, 0
  %197 = zext i1 %196 to i32
  %198 = icmp ne i32 %197, 0
  br i1 %198, label %199, label %200

199:                                              ; preds = %186
  br label %207

200:                                              ; preds = %186
  %201 = load ptr, ptr %15, align 8
  %202 = load i32, ptr %201, align 8
  %203 = add i32 %202, -1
  store i32 %203, ptr %201, align 8
  %204 = icmp eq i32 %203, 0
  br i1 %204, label %205, label %207

205:                                              ; preds = %200
  %206 = load ptr, ptr %15, align 8
  call void @_Py_Dealloc(ptr noundef %206) #7
  br label %207

207:                                              ; preds = %199, %200, %205
  br label %208

208:                                              ; preds = %207
  br label %209

209:                                              ; preds = %208, %181
  br label %210

210:                                              ; preds = %209, %171
  %211 = load ptr, ptr %33, align 8
  %212 = call i32 @PyObject_GetOptionalAttr(ptr noundef %211, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 123), ptr noundef %29)
  %213 = icmp slt i32 %212, 0
  br i1 %213, label %214, label %215

214:                                              ; preds = %210
  store ptr null, ptr %32, align 8
  br label %245

215:                                              ; preds = %210
  %216 = load ptr, ptr %29, align 8
  %217 = icmp eq ptr %216, null
  br i1 %217, label %218, label %220

218:                                              ; preds = %215
  %219 = call ptr @PyDict_New()
  store ptr %219, ptr %32, align 8
  br label %244

220:                                              ; preds = %215
  %221 = load ptr, ptr %27, align 8
  store ptr %221, ptr %41, align 8
  %222 = getelementptr inbounds ptr, ptr %41, i64 1
  %223 = load ptr, ptr %36, align 8
  store ptr %223, ptr %222, align 8
  %224 = load ptr, ptr %29, align 8
  %225 = getelementptr inbounds [2 x ptr], ptr %41, i64 0, i64 0
  %226 = load ptr, ptr %35, align 8
  %227 = call ptr @PyObject_VectorcallDict(ptr noundef %224, ptr noundef %225, i64 noundef 2, ptr noundef %226)
  store ptr %227, ptr %32, align 8
  %228 = load ptr, ptr %29, align 8
  store ptr %228, ptr %16, align 8
  %229 = load ptr, ptr %16, align 8
  store ptr %229, ptr %9, align 8
  %230 = load ptr, ptr %9, align 8
  %231 = load i32, ptr %230, align 8
  %232 = icmp slt i32 %231, 0
  %233 = zext i1 %232 to i32
  %234 = icmp ne i32 %233, 0
  br i1 %234, label %235, label %236

235:                                              ; preds = %220
  br label %243

236:                                              ; preds = %220
  %237 = load ptr, ptr %16, align 8
  %238 = load i32, ptr %237, align 8
  %239 = add i32 %238, -1
  store i32 %239, ptr %237, align 8
  %240 = icmp eq i32 %239, 0
  br i1 %240, label %241, label %243

241:                                              ; preds = %236
  %242 = load ptr, ptr %16, align 8
  call void @_Py_Dealloc(ptr noundef %242) #7
  br label %243

243:                                              ; preds = %235, %236, %241
  br label %244

244:                                              ; preds = %243, %218
  br label %245

245:                                              ; preds = %244, %214
  %246 = load ptr, ptr %32, align 8
  %247 = icmp eq ptr %246, null
  br i1 %247, label %248, label %249

248:                                              ; preds = %245
  br label %373

249:                                              ; preds = %245
  %250 = load ptr, ptr %32, align 8
  %251 = call i32 @PyMapping_Check(ptr noundef %250)
  %252 = icmp ne i32 %251, 0
  br i1 %252, label %269, label %253

253:                                              ; preds = %249
  %254 = load ptr, ptr @PyExc_TypeError, align 8
  %255 = load i32, ptr %37, align 4
  %256 = icmp ne i32 %255, 0
  br i1 %256, label %257, label %261

257:                                              ; preds = %253
  %258 = load ptr, ptr %33, align 8
  %259 = getelementptr inbounds %struct._typeobject, ptr %258, i32 0, i32 1
  %260 = load ptr, ptr %259, align 8
  br label %262

261:                                              ; preds = %253
  br label %262

262:                                              ; preds = %261, %257
  %263 = phi ptr [ %260, %257 ], [ @.str.124, %261 ]
  %264 = load ptr, ptr %32, align 8
  %265 = call ptr @_Py_TYPE(ptr noundef %264)
  %266 = getelementptr inbounds %struct._typeobject, ptr %265, i32 0, i32 1
  %267 = load ptr, ptr %266, align 8
  %268 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %254, ptr noundef @.str.123, ptr noundef %263, ptr noundef %267)
  br label %373

269:                                              ; preds = %249
  %270 = call ptr @_PyThreadState_GET()
  store ptr %270, ptr %42, align 8
  %271 = load ptr, ptr %42, align 8
  %272 = load ptr, ptr %26, align 8
  %273 = load ptr, ptr %32, align 8
  %274 = call ptr @_PyEval_Vector(ptr noundef %271, ptr noundef %272, ptr noundef %273, ptr noundef null, i64 noundef 0, ptr noundef null)
  store ptr %274, ptr %31, align 8
  %275 = load ptr, ptr %31, align 8
  %276 = icmp ne ptr %275, null
  br i1 %276, label %277, label %372

277:                                              ; preds = %269
  %278 = load ptr, ptr %36, align 8
  %279 = load ptr, ptr %34, align 8
  %280 = icmp ne ptr %278, %279
  br i1 %280, label %281, label %288

281:                                              ; preds = %277
  %282 = load ptr, ptr %32, align 8
  %283 = load ptr, ptr %34, align 8
  %284 = call i32 @PyMapping_SetItemString(ptr noundef %282, ptr noundef @.str.125, ptr noundef %283)
  %285 = icmp slt i32 %284, 0
  br i1 %285, label %286, label %287

286:                                              ; preds = %281
  br label %373

287:                                              ; preds = %281
  br label %288

288:                                              ; preds = %287, %277
  %289 = load ptr, ptr %27, align 8
  store ptr %289, ptr %43, align 8
  %290 = getelementptr inbounds ptr, ptr %43, i64 1
  %291 = load ptr, ptr %36, align 8
  store ptr %291, ptr %290, align 8
  %292 = getelementptr inbounds ptr, ptr %43, i64 2
  %293 = load ptr, ptr %32, align 8
  store ptr %293, ptr %292, align 8
  %294 = load ptr, ptr %33, align 8
  %295 = getelementptr inbounds [3 x ptr], ptr %43, i64 0, i64 0
  %296 = load ptr, ptr %35, align 8
  %297 = call ptr @PyObject_VectorcallDict(ptr noundef %294, ptr noundef %295, i64 noundef 3, ptr noundef %296)
  store ptr %297, ptr %30, align 8
  %298 = load ptr, ptr %30, align 8
  %299 = icmp ne ptr %298, null
  br i1 %299, label %300, label %371

300:                                              ; preds = %288
  %301 = load ptr, ptr %30, align 8
  %302 = call i32 @PyType_Check(ptr noundef %301)
  %303 = icmp ne i32 %302, 0
  br i1 %303, label %304, label %371

304:                                              ; preds = %300
  %305 = load ptr, ptr %31, align 8
  %306 = call i32 @Py_IS_TYPE(ptr noundef %305, ptr noundef @PyCell_Type)
  %307 = icmp ne i32 %306, 0
  br i1 %307, label %308, label %371

308:                                              ; preds = %304
  %309 = load ptr, ptr %31, align 8
  %310 = call ptr @PyCell_GetRef(ptr noundef %309)
  store ptr %310, ptr %44, align 8
  %311 = load ptr, ptr %44, align 8
  %312 = load ptr, ptr %30, align 8
  %313 = icmp ne ptr %311, %312
  br i1 %313, label %314, label %353

314:                                              ; preds = %308
  %315 = load ptr, ptr %44, align 8
  %316 = icmp eq ptr %315, null
  br i1 %316, label %317, label %323

317:                                              ; preds = %314
  store ptr @.str.126, ptr %45, align 8
  %318 = load ptr, ptr @PyExc_RuntimeError, align 8
  %319 = load ptr, ptr %45, align 8
  %320 = load ptr, ptr %27, align 8
  %321 = load ptr, ptr %30, align 8
  %322 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %318, ptr noundef %319, ptr noundef %320, ptr noundef %321)
  br label %330

323:                                              ; preds = %314
  store ptr @.str.127, ptr %46, align 8
  %324 = load ptr, ptr @PyExc_TypeError, align 8
  %325 = load ptr, ptr %46, align 8
  %326 = load ptr, ptr %44, align 8
  %327 = load ptr, ptr %27, align 8
  %328 = load ptr, ptr %30, align 8
  %329 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %324, ptr noundef %325, ptr noundef %326, ptr noundef %327, ptr noundef %328)
  br label %330

330:                                              ; preds = %323, %317
  %331 = load ptr, ptr %44, align 8
  call void @Py_XDECREF(ptr noundef %331)
  br label %332

332:                                              ; preds = %330
  store ptr %30, ptr %47, align 8
  %333 = load ptr, ptr %47, align 8
  %334 = load ptr, ptr %333, align 8
  store ptr %334, ptr %48, align 8
  %335 = load ptr, ptr %47, align 8
  store ptr null, ptr %335, align 8
  %336 = load ptr, ptr %48, align 8
  store ptr %336, ptr %17, align 8
  %337 = load ptr, ptr %17, align 8
  store ptr %337, ptr %8, align 8
  %338 = load ptr, ptr %8, align 8
  %339 = load i32, ptr %338, align 8
  %340 = icmp slt i32 %339, 0
  %341 = zext i1 %340 to i32
  %342 = icmp ne i32 %341, 0
  br i1 %342, label %343, label %344

343:                                              ; preds = %332
  br label %351

344:                                              ; preds = %332
  %345 = load ptr, ptr %17, align 8
  %346 = load i32, ptr %345, align 8
  %347 = add i32 %346, -1
  store i32 %347, ptr %345, align 8
  %348 = icmp eq i32 %347, 0
  br i1 %348, label %349, label %351

349:                                              ; preds = %344
  %350 = load ptr, ptr %17, align 8
  call void @_Py_Dealloc(ptr noundef %350) #7
  br label %351

351:                                              ; preds = %343, %344, %349
  br label %352

352:                                              ; preds = %351
  br label %373

353:                                              ; preds = %308
  %354 = load ptr, ptr %44, align 8
  store ptr %354, ptr %18, align 8
  %355 = load ptr, ptr %18, align 8
  store ptr %355, ptr %7, align 8
  %356 = load ptr, ptr %7, align 8
  %357 = load i32, ptr %356, align 8
  %358 = icmp slt i32 %357, 0
  %359 = zext i1 %358 to i32
  %360 = icmp ne i32 %359, 0
  br i1 %360, label %361, label %362

361:                                              ; preds = %353
  br label %369

362:                                              ; preds = %353
  %363 = load ptr, ptr %18, align 8
  %364 = load i32, ptr %363, align 8
  %365 = add i32 %364, -1
  store i32 %365, ptr %363, align 8
  %366 = icmp eq i32 %365, 0
  br i1 %366, label %367, label %369

367:                                              ; preds = %362
  %368 = load ptr, ptr %18, align 8
  call void @_Py_Dealloc(ptr noundef %368) #7
  br label %369

369:                                              ; preds = %361, %362, %367
  br label %370

370:                                              ; preds = %369
  br label %371

371:                                              ; preds = %370, %304, %300, %288
  br label %372

372:                                              ; preds = %371, %269
  br label %373

373:                                              ; preds = %372, %352, %286, %262, %248, %180, %124, %119
  %374 = load ptr, ptr %31, align 8
  call void @Py_XDECREF(ptr noundef %374)
  %375 = load ptr, ptr %32, align 8
  call void @Py_XDECREF(ptr noundef %375)
  %376 = load ptr, ptr %33, align 8
  call void @Py_XDECREF(ptr noundef %376)
  %377 = load ptr, ptr %35, align 8
  call void @Py_XDECREF(ptr noundef %377)
  %378 = load ptr, ptr %36, align 8
  %379 = load ptr, ptr %34, align 8
  %380 = icmp ne ptr %378, %379
  br i1 %380, label %381, label %398

381:                                              ; preds = %373
  %382 = load ptr, ptr %34, align 8
  store ptr %382, ptr %19, align 8
  %383 = load ptr, ptr %19, align 8
  store ptr %383, ptr %6, align 8
  %384 = load ptr, ptr %6, align 8
  %385 = load i32, ptr %384, align 8
  %386 = icmp slt i32 %385, 0
  %387 = zext i1 %386 to i32
  %388 = icmp ne i32 %387, 0
  br i1 %388, label %389, label %390

389:                                              ; preds = %381
  br label %397

390:                                              ; preds = %381
  %391 = load ptr, ptr %19, align 8
  %392 = load i32, ptr %391, align 8
  %393 = add i32 %392, -1
  store i32 %393, ptr %391, align 8
  %394 = icmp eq i32 %393, 0
  br i1 %394, label %395, label %397

395:                                              ; preds = %390
  %396 = load ptr, ptr %19, align 8
  call void @_Py_Dealloc(ptr noundef %396) #7
  br label %397

397:                                              ; preds = %389, %390, %395
  br label %398

398:                                              ; preds = %397, %373
  %399 = load ptr, ptr %36, align 8
  store ptr %399, ptr %20, align 8
  %400 = load ptr, ptr %20, align 8
  store ptr %400, ptr %5, align 8
  %401 = load ptr, ptr %5, align 8
  %402 = load i32, ptr %401, align 8
  %403 = icmp slt i32 %402, 0
  %404 = zext i1 %403 to i32
  %405 = icmp ne i32 %404, 0
  br i1 %405, label %406, label %407

406:                                              ; preds = %398
  br label %414

407:                                              ; preds = %398
  %408 = load ptr, ptr %20, align 8
  %409 = load i32, ptr %408, align 8
  %410 = add i32 %409, -1
  store i32 %410, ptr %408, align 8
  %411 = icmp eq i32 %410, 0
  br i1 %411, label %412, label %414

412:                                              ; preds = %407
  %413 = load ptr, ptr %20, align 8
  call void @_Py_Dealloc(ptr noundef %413) #7
  br label %414

414:                                              ; preds = %406, %407, %412
  %415 = load ptr, ptr %30, align 8
  store ptr %415, ptr %21, align 8
  br label %416

416:                                              ; preds = %414, %106, %80, %70, %60, %51
  %417 = load ptr, ptr %21, align 8
  ret ptr %417
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin___import__(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [5 x ptr], align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store ptr null, ptr %9, align 8
  %17 = load i64, ptr %7, align 8
  %18 = load ptr, ptr %8, align 8
  %19 = icmp ne ptr %18, null
  br i1 %19, label %20, label %23

20:                                               ; preds = %4
  %21 = load ptr, ptr %8, align 8
  %22 = call i64 @PyTuple_GET_SIZE(ptr noundef %21)
  br label %24

23:                                               ; preds = %4
  br label %24

24:                                               ; preds = %23, %20
  %25 = phi i64 [ %22, %20 ], [ 0, %23 ]
  %26 = add nsw i64 %17, %25
  %27 = sub nsw i64 %26, 1
  store i64 %27, ptr %11, align 8
  store ptr null, ptr %13, align 8
  store ptr null, ptr %14, align 8
  store ptr null, ptr %15, align 8
  store i32 0, ptr %16, align 4
  %28 = load ptr, ptr %8, align 8
  %29 = icmp eq ptr %28, null
  br i1 %29, label %30, label %41

30:                                               ; preds = %24
  %31 = load i64, ptr %7, align 8
  %32 = icmp sle i64 1, %31
  br i1 %32, label %33, label %41

33:                                               ; preds = %30
  %34 = load i64, ptr %7, align 8
  %35 = icmp sle i64 %34, 5
  br i1 %35, label %36, label %41

36:                                               ; preds = %33
  %37 = load ptr, ptr %6, align 8
  %38 = icmp ne ptr %37, null
  br i1 %38, label %39, label %41

39:                                               ; preds = %36
  %40 = load ptr, ptr %6, align 8
  br label %47

41:                                               ; preds = %36, %33, %30, %24
  %42 = load ptr, ptr %6, align 8
  %43 = load i64, ptr %7, align 8
  %44 = load ptr, ptr %8, align 8
  %45 = getelementptr inbounds [5 x ptr], ptr %10, i64 0, i64 0
  %46 = call ptr @_PyArg_UnpackKeywords(ptr noundef %42, i64 noundef %43, ptr noundef null, ptr noundef %44, ptr noundef @builtin___import__._parser, i32 noundef 1, i32 noundef 5, i32 noundef 0, i32 noundef 0, ptr noundef %45)
  br label %47

47:                                               ; preds = %41, %39
  %48 = phi ptr [ %40, %39 ], [ %46, %41 ]
  store ptr %48, ptr %6, align 8
  %49 = load ptr, ptr %6, align 8
  %50 = icmp ne ptr %49, null
  br i1 %50, label %52, label %51

51:                                               ; preds = %47
  br label %121

52:                                               ; preds = %47
  %53 = load ptr, ptr %6, align 8
  %54 = getelementptr inbounds ptr, ptr %53, i64 0
  %55 = load ptr, ptr %54, align 8
  store ptr %55, ptr %12, align 8
  %56 = load i64, ptr %11, align 8
  %57 = icmp ne i64 %56, 0
  br i1 %57, label %59, label %58

58:                                               ; preds = %52
  br label %113

59:                                               ; preds = %52
  %60 = load ptr, ptr %6, align 8
  %61 = getelementptr inbounds ptr, ptr %60, i64 1
  %62 = load ptr, ptr %61, align 8
  %63 = icmp ne ptr %62, null
  br i1 %63, label %64, label %73

64:                                               ; preds = %59
  %65 = load ptr, ptr %6, align 8
  %66 = getelementptr inbounds ptr, ptr %65, i64 1
  %67 = load ptr, ptr %66, align 8
  store ptr %67, ptr %13, align 8
  %68 = load i64, ptr %11, align 8
  %69 = add nsw i64 %68, -1
  store i64 %69, ptr %11, align 8
  %70 = icmp ne i64 %69, 0
  br i1 %70, label %72, label %71

71:                                               ; preds = %64
  br label %113

72:                                               ; preds = %64
  br label %73

73:                                               ; preds = %72, %59
  %74 = load ptr, ptr %6, align 8
  %75 = getelementptr inbounds ptr, ptr %74, i64 2
  %76 = load ptr, ptr %75, align 8
  %77 = icmp ne ptr %76, null
  br i1 %77, label %78, label %87

78:                                               ; preds = %73
  %79 = load ptr, ptr %6, align 8
  %80 = getelementptr inbounds ptr, ptr %79, i64 2
  %81 = load ptr, ptr %80, align 8
  store ptr %81, ptr %14, align 8
  %82 = load i64, ptr %11, align 8
  %83 = add nsw i64 %82, -1
  store i64 %83, ptr %11, align 8
  %84 = icmp ne i64 %83, 0
  br i1 %84, label %86, label %85

85:                                               ; preds = %78
  br label %113

86:                                               ; preds = %78
  br label %87

87:                                               ; preds = %86, %73
  %88 = load ptr, ptr %6, align 8
  %89 = getelementptr inbounds ptr, ptr %88, i64 3
  %90 = load ptr, ptr %89, align 8
  %91 = icmp ne ptr %90, null
  br i1 %91, label %92, label %101

92:                                               ; preds = %87
  %93 = load ptr, ptr %6, align 8
  %94 = getelementptr inbounds ptr, ptr %93, i64 3
  %95 = load ptr, ptr %94, align 8
  store ptr %95, ptr %15, align 8
  %96 = load i64, ptr %11, align 8
  %97 = add nsw i64 %96, -1
  store i64 %97, ptr %11, align 8
  %98 = icmp ne i64 %97, 0
  br i1 %98, label %100, label %99

99:                                               ; preds = %92
  br label %113

100:                                              ; preds = %92
  br label %101

101:                                              ; preds = %100, %87
  %102 = load ptr, ptr %6, align 8
  %103 = getelementptr inbounds ptr, ptr %102, i64 4
  %104 = load ptr, ptr %103, align 8
  %105 = call i32 @PyLong_AsInt(ptr noundef %104)
  store i32 %105, ptr %16, align 4
  %106 = load i32, ptr %16, align 4
  %107 = icmp eq i32 %106, -1
  br i1 %107, label %108, label %112

108:                                              ; preds = %101
  %109 = call ptr @PyErr_Occurred()
  %110 = icmp ne ptr %109, null
  br i1 %110, label %111, label %112

111:                                              ; preds = %108
  br label %121

112:                                              ; preds = %108, %101
  br label %113

113:                                              ; preds = %112, %99, %85, %71, %58
  %114 = load ptr, ptr %5, align 8
  %115 = load ptr, ptr %12, align 8
  %116 = load ptr, ptr %13, align 8
  %117 = load ptr, ptr %14, align 8
  %118 = load ptr, ptr %15, align 8
  %119 = load i32, ptr %16, align 4
  %120 = call ptr @builtin___import___impl(ptr noundef %114, ptr noundef %115, ptr noundef %116, ptr noundef %117, ptr noundef %118, i32 noundef %119)
  store ptr %120, ptr %9, align 8
  br label %121

121:                                              ; preds = %113, %111, %51
  %122 = load ptr, ptr %9, align 8
  ret ptr %122
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_abs(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call ptr @PyNumber_Absolute(ptr noundef %5)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_all(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca i32, align 4
  store ptr %0, ptr %12, align 8
  store ptr %1, ptr %13, align 8
  %18 = load ptr, ptr %13, align 8
  %19 = call ptr @PyObject_GetIter(ptr noundef %18)
  store ptr %19, ptr %14, align 8
  %20 = load ptr, ptr %14, align 8
  %21 = icmp eq ptr %20, null
  br i1 %21, label %22, label %23

22:                                               ; preds = %2
  store ptr null, ptr %11, align 8
  br label %121

23:                                               ; preds = %2
  %24 = load ptr, ptr %14, align 8
  %25 = call ptr @_Py_TYPE(ptr noundef %24)
  %26 = getelementptr inbounds %struct._typeobject, ptr %25, i32 0, i32 26
  %27 = load ptr, ptr %26, align 8
  store ptr %27, ptr %16, align 8
  br label %28

28:                                               ; preds = %93, %23
  %29 = load ptr, ptr %16, align 8
  %30 = load ptr, ptr %14, align 8
  %31 = call ptr %29(ptr noundef %30)
  store ptr %31, ptr %15, align 8
  %32 = load ptr, ptr %15, align 8
  %33 = icmp eq ptr %32, null
  br i1 %33, label %34, label %35

34:                                               ; preds = %28
  br label %94

35:                                               ; preds = %28
  %36 = load ptr, ptr %15, align 8
  %37 = call i32 @PyObject_IsTrue(ptr noundef %36)
  store i32 %37, ptr %17, align 4
  %38 = load ptr, ptr %15, align 8
  store ptr %38, ptr %7, align 8
  %39 = load ptr, ptr %7, align 8
  store ptr %39, ptr %6, align 8
  %40 = load ptr, ptr %6, align 8
  %41 = load i32, ptr %40, align 8
  %42 = icmp slt i32 %41, 0
  %43 = zext i1 %42 to i32
  %44 = icmp ne i32 %43, 0
  br i1 %44, label %45, label %46

45:                                               ; preds = %35
  br label %53

46:                                               ; preds = %35
  %47 = load ptr, ptr %7, align 8
  %48 = load i32, ptr %47, align 8
  %49 = add i32 %48, -1
  store i32 %49, ptr %47, align 8
  %50 = icmp eq i32 %49, 0
  br i1 %50, label %51, label %53

51:                                               ; preds = %46
  %52 = load ptr, ptr %7, align 8
  call void @_Py_Dealloc(ptr noundef %52) #7
  br label %53

53:                                               ; preds = %45, %46, %51
  %54 = load i32, ptr %17, align 4
  %55 = icmp slt i32 %54, 0
  br i1 %55, label %56, label %73

56:                                               ; preds = %53
  %57 = load ptr, ptr %14, align 8
  store ptr %57, ptr %8, align 8
  %58 = load ptr, ptr %8, align 8
  store ptr %58, ptr %5, align 8
  %59 = load ptr, ptr %5, align 8
  %60 = load i32, ptr %59, align 8
  %61 = icmp slt i32 %60, 0
  %62 = zext i1 %61 to i32
  %63 = icmp ne i32 %62, 0
  br i1 %63, label %64, label %65

64:                                               ; preds = %56
  br label %72

65:                                               ; preds = %56
  %66 = load ptr, ptr %8, align 8
  %67 = load i32, ptr %66, align 8
  %68 = add i32 %67, -1
  store i32 %68, ptr %66, align 8
  %69 = icmp eq i32 %68, 0
  br i1 %69, label %70, label %72

70:                                               ; preds = %65
  %71 = load ptr, ptr %8, align 8
  call void @_Py_Dealloc(ptr noundef %71) #7
  br label %72

72:                                               ; preds = %64, %65, %70
  store ptr null, ptr %11, align 8
  br label %121

73:                                               ; preds = %53
  %74 = load i32, ptr %17, align 4
  %75 = icmp eq i32 %74, 0
  br i1 %75, label %76, label %93

76:                                               ; preds = %73
  %77 = load ptr, ptr %14, align 8
  store ptr %77, ptr %9, align 8
  %78 = load ptr, ptr %9, align 8
  store ptr %78, ptr %4, align 8
  %79 = load ptr, ptr %4, align 8
  %80 = load i32, ptr %79, align 8
  %81 = icmp slt i32 %80, 0
  %82 = zext i1 %81 to i32
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %84, label %85

84:                                               ; preds = %76
  br label %92

85:                                               ; preds = %76
  %86 = load ptr, ptr %9, align 8
  %87 = load i32, ptr %86, align 8
  %88 = add i32 %87, -1
  store i32 %88, ptr %86, align 8
  %89 = icmp eq i32 %88, 0
  br i1 %89, label %90, label %92

90:                                               ; preds = %85
  %91 = load ptr, ptr %9, align 8
  call void @_Py_Dealloc(ptr noundef %91) #7
  br label %92

92:                                               ; preds = %84, %85, %90
  store ptr @_Py_FalseStruct, ptr %11, align 8
  br label %121

93:                                               ; preds = %73
  br label %28

94:                                               ; preds = %34
  %95 = load ptr, ptr %14, align 8
  store ptr %95, ptr %10, align 8
  %96 = load ptr, ptr %10, align 8
  store ptr %96, ptr %3, align 8
  %97 = load ptr, ptr %3, align 8
  %98 = load i32, ptr %97, align 8
  %99 = icmp slt i32 %98, 0
  %100 = zext i1 %99 to i32
  %101 = icmp ne i32 %100, 0
  br i1 %101, label %102, label %103

102:                                              ; preds = %94
  br label %110

103:                                              ; preds = %94
  %104 = load ptr, ptr %10, align 8
  %105 = load i32, ptr %104, align 8
  %106 = add i32 %105, -1
  store i32 %106, ptr %104, align 8
  %107 = icmp eq i32 %106, 0
  br i1 %107, label %108, label %110

108:                                              ; preds = %103
  %109 = load ptr, ptr %10, align 8
  call void @_Py_Dealloc(ptr noundef %109) #7
  br label %110

110:                                              ; preds = %102, %103, %108
  %111 = call ptr @PyErr_Occurred()
  %112 = icmp ne ptr %111, null
  br i1 %112, label %113, label %120

113:                                              ; preds = %110
  %114 = load ptr, ptr @PyExc_StopIteration, align 8
  %115 = call i32 @PyErr_ExceptionMatches(ptr noundef %114)
  %116 = icmp ne i32 %115, 0
  br i1 %116, label %117, label %118

117:                                              ; preds = %113
  call void @PyErr_Clear()
  br label %119

118:                                              ; preds = %113
  store ptr null, ptr %11, align 8
  br label %121

119:                                              ; preds = %117
  br label %120

120:                                              ; preds = %119, %110
  store ptr @_Py_TrueStruct, ptr %11, align 8
  br label %121

121:                                              ; preds = %120, %118, %92, %72, %22
  %122 = load ptr, ptr %11, align 8
  ret ptr %122
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_any(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca i32, align 4
  store ptr %0, ptr %12, align 8
  store ptr %1, ptr %13, align 8
  %18 = load ptr, ptr %13, align 8
  %19 = call ptr @PyObject_GetIter(ptr noundef %18)
  store ptr %19, ptr %14, align 8
  %20 = load ptr, ptr %14, align 8
  %21 = icmp eq ptr %20, null
  br i1 %21, label %22, label %23

22:                                               ; preds = %2
  store ptr null, ptr %11, align 8
  br label %121

23:                                               ; preds = %2
  %24 = load ptr, ptr %14, align 8
  %25 = call ptr @_Py_TYPE(ptr noundef %24)
  %26 = getelementptr inbounds %struct._typeobject, ptr %25, i32 0, i32 26
  %27 = load ptr, ptr %26, align 8
  store ptr %27, ptr %16, align 8
  br label %28

28:                                               ; preds = %93, %23
  %29 = load ptr, ptr %16, align 8
  %30 = load ptr, ptr %14, align 8
  %31 = call ptr %29(ptr noundef %30)
  store ptr %31, ptr %15, align 8
  %32 = load ptr, ptr %15, align 8
  %33 = icmp eq ptr %32, null
  br i1 %33, label %34, label %35

34:                                               ; preds = %28
  br label %94

35:                                               ; preds = %28
  %36 = load ptr, ptr %15, align 8
  %37 = call i32 @PyObject_IsTrue(ptr noundef %36)
  store i32 %37, ptr %17, align 4
  %38 = load ptr, ptr %15, align 8
  store ptr %38, ptr %7, align 8
  %39 = load ptr, ptr %7, align 8
  store ptr %39, ptr %6, align 8
  %40 = load ptr, ptr %6, align 8
  %41 = load i32, ptr %40, align 8
  %42 = icmp slt i32 %41, 0
  %43 = zext i1 %42 to i32
  %44 = icmp ne i32 %43, 0
  br i1 %44, label %45, label %46

45:                                               ; preds = %35
  br label %53

46:                                               ; preds = %35
  %47 = load ptr, ptr %7, align 8
  %48 = load i32, ptr %47, align 8
  %49 = add i32 %48, -1
  store i32 %49, ptr %47, align 8
  %50 = icmp eq i32 %49, 0
  br i1 %50, label %51, label %53

51:                                               ; preds = %46
  %52 = load ptr, ptr %7, align 8
  call void @_Py_Dealloc(ptr noundef %52) #7
  br label %53

53:                                               ; preds = %45, %46, %51
  %54 = load i32, ptr %17, align 4
  %55 = icmp slt i32 %54, 0
  br i1 %55, label %56, label %73

56:                                               ; preds = %53
  %57 = load ptr, ptr %14, align 8
  store ptr %57, ptr %8, align 8
  %58 = load ptr, ptr %8, align 8
  store ptr %58, ptr %5, align 8
  %59 = load ptr, ptr %5, align 8
  %60 = load i32, ptr %59, align 8
  %61 = icmp slt i32 %60, 0
  %62 = zext i1 %61 to i32
  %63 = icmp ne i32 %62, 0
  br i1 %63, label %64, label %65

64:                                               ; preds = %56
  br label %72

65:                                               ; preds = %56
  %66 = load ptr, ptr %8, align 8
  %67 = load i32, ptr %66, align 8
  %68 = add i32 %67, -1
  store i32 %68, ptr %66, align 8
  %69 = icmp eq i32 %68, 0
  br i1 %69, label %70, label %72

70:                                               ; preds = %65
  %71 = load ptr, ptr %8, align 8
  call void @_Py_Dealloc(ptr noundef %71) #7
  br label %72

72:                                               ; preds = %64, %65, %70
  store ptr null, ptr %11, align 8
  br label %121

73:                                               ; preds = %53
  %74 = load i32, ptr %17, align 4
  %75 = icmp sgt i32 %74, 0
  br i1 %75, label %76, label %93

76:                                               ; preds = %73
  %77 = load ptr, ptr %14, align 8
  store ptr %77, ptr %9, align 8
  %78 = load ptr, ptr %9, align 8
  store ptr %78, ptr %4, align 8
  %79 = load ptr, ptr %4, align 8
  %80 = load i32, ptr %79, align 8
  %81 = icmp slt i32 %80, 0
  %82 = zext i1 %81 to i32
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %84, label %85

84:                                               ; preds = %76
  br label %92

85:                                               ; preds = %76
  %86 = load ptr, ptr %9, align 8
  %87 = load i32, ptr %86, align 8
  %88 = add i32 %87, -1
  store i32 %88, ptr %86, align 8
  %89 = icmp eq i32 %88, 0
  br i1 %89, label %90, label %92

90:                                               ; preds = %85
  %91 = load ptr, ptr %9, align 8
  call void @_Py_Dealloc(ptr noundef %91) #7
  br label %92

92:                                               ; preds = %84, %85, %90
  store ptr @_Py_TrueStruct, ptr %11, align 8
  br label %121

93:                                               ; preds = %73
  br label %28

94:                                               ; preds = %34
  %95 = load ptr, ptr %14, align 8
  store ptr %95, ptr %10, align 8
  %96 = load ptr, ptr %10, align 8
  store ptr %96, ptr %3, align 8
  %97 = load ptr, ptr %3, align 8
  %98 = load i32, ptr %97, align 8
  %99 = icmp slt i32 %98, 0
  %100 = zext i1 %99 to i32
  %101 = icmp ne i32 %100, 0
  br i1 %101, label %102, label %103

102:                                              ; preds = %94
  br label %110

103:                                              ; preds = %94
  %104 = load ptr, ptr %10, align 8
  %105 = load i32, ptr %104, align 8
  %106 = add i32 %105, -1
  store i32 %106, ptr %104, align 8
  %107 = icmp eq i32 %106, 0
  br i1 %107, label %108, label %110

108:                                              ; preds = %103
  %109 = load ptr, ptr %10, align 8
  call void @_Py_Dealloc(ptr noundef %109) #7
  br label %110

110:                                              ; preds = %102, %103, %108
  %111 = call ptr @PyErr_Occurred()
  %112 = icmp ne ptr %111, null
  br i1 %112, label %113, label %120

113:                                              ; preds = %110
  %114 = load ptr, ptr @PyExc_StopIteration, align 8
  %115 = call i32 @PyErr_ExceptionMatches(ptr noundef %114)
  %116 = icmp ne i32 %115, 0
  br i1 %116, label %117, label %118

117:                                              ; preds = %113
  call void @PyErr_Clear()
  br label %119

118:                                              ; preds = %113
  store ptr null, ptr %11, align 8
  br label %121

119:                                              ; preds = %117
  br label %120

120:                                              ; preds = %119, %110
  store ptr @_Py_FalseStruct, ptr %11, align 8
  br label %121

121:                                              ; preds = %120, %118, %92, %72, %22
  %122 = load ptr, ptr %11, align 8
  ret ptr %122
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_ascii(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call ptr @PyObject_ASCII(ptr noundef %5)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_bin(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call ptr @PyNumber_ToBase(ptr noundef %5, i32 noundef 2)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_breakpoint(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i64, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  store ptr %0, ptr %10, align 8
  store ptr %1, ptr %11, align 8
  store i64 %2, ptr %12, align 8
  store ptr %3, ptr %13, align 8
  %16 = call ptr @PySys_GetObject(ptr noundef @.str.135)
  store ptr %16, ptr %14, align 8
  %17 = load ptr, ptr %14, align 8
  %18 = icmp eq ptr %17, null
  br i1 %18, label %19, label %21

19:                                               ; preds = %4
  %20 = load ptr, ptr @PyExc_RuntimeError, align 8
  call void @PyErr_SetString(ptr noundef %20, ptr noundef @.str.136)
  store ptr null, ptr %9, align 8
  br label %60

21:                                               ; preds = %4
  %22 = load ptr, ptr %14, align 8
  %23 = call i32 (ptr, ptr, ...) @PySys_Audit(ptr noundef @.str.137, ptr noundef @.str.138, ptr noundef %22)
  %24 = icmp slt i32 %23, 0
  br i1 %24, label %25, label %26

25:                                               ; preds = %21
  store ptr null, ptr %9, align 8
  br label %60

26:                                               ; preds = %21
  %27 = load ptr, ptr %14, align 8
  store ptr %27, ptr %6, align 8
  %28 = load ptr, ptr %6, align 8
  %29 = load i32, ptr %28, align 8
  store i32 %29, ptr %7, align 4
  %30 = load i32, ptr %7, align 4
  %31 = icmp slt i32 %30, 0
  br i1 %31, label %32, label %33

32:                                               ; preds = %26
  br label %37

33:                                               ; preds = %26
  %34 = load i32, ptr %7, align 4
  %35 = add i32 %34, 1
  %36 = load ptr, ptr %6, align 8
  store i32 %35, ptr %36, align 8
  br label %37

37:                                               ; preds = %32, %33
  %38 = load ptr, ptr %14, align 8
  %39 = load ptr, ptr %11, align 8
  %40 = load i64, ptr %12, align 8
  %41 = load ptr, ptr %13, align 8
  %42 = call ptr @PyObject_Vectorcall(ptr noundef %38, ptr noundef %39, i64 noundef %40, ptr noundef %41)
  store ptr %42, ptr %15, align 8
  %43 = load ptr, ptr %14, align 8
  store ptr %43, ptr %8, align 8
  %44 = load ptr, ptr %8, align 8
  store ptr %44, ptr %5, align 8
  %45 = load ptr, ptr %5, align 8
  %46 = load i32, ptr %45, align 8
  %47 = icmp slt i32 %46, 0
  %48 = zext i1 %47 to i32
  %49 = icmp ne i32 %48, 0
  br i1 %49, label %50, label %51

50:                                               ; preds = %37
  br label %58

51:                                               ; preds = %37
  %52 = load ptr, ptr %8, align 8
  %53 = load i32, ptr %52, align 8
  %54 = add i32 %53, -1
  store i32 %54, ptr %52, align 8
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %56, label %58

56:                                               ; preds = %51
  %57 = load ptr, ptr %8, align 8
  call void @_Py_Dealloc(ptr noundef %57) #7
  br label %58

58:                                               ; preds = %50, %51, %56
  %59 = load ptr, ptr %15, align 8
  store ptr %59, ptr %9, align 8
  br label %60

60:                                               ; preds = %58, %25, %19
  %61 = load ptr, ptr %9, align 8
  ret ptr %61
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_callable(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call i32 @PyCallable_Check(ptr noundef %5)
  %7 = sext i32 %6 to i64
  %8 = call ptr @PyBool_FromLong(i64 noundef %7)
  ret ptr %8
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_chr(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i64, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = call i64 @PyLong_AsLongAndOverflow(ptr noundef %8, ptr noundef %6)
  store i64 %9, ptr %7, align 8
  %10 = load i64, ptr %7, align 8
  %11 = icmp eq i64 %10, -1
  br i1 %11, label %12, label %16

12:                                               ; preds = %2
  %13 = call ptr @PyErr_Occurred()
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %16

15:                                               ; preds = %12
  store ptr null, ptr %3, align 8
  br label %39

16:                                               ; preds = %12, %2
  %17 = load i32, ptr %6, align 4
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %19, label %25

19:                                               ; preds = %16
  %20 = load i32, ptr %6, align 4
  %21 = icmp slt i32 %20, 0
  %22 = zext i1 %21 to i64
  %23 = select i1 %21, i32 -2147483648, i32 2147483647
  %24 = sext i32 %23 to i64
  store i64 %24, ptr %7, align 8
  br label %35

25:                                               ; preds = %16
  %26 = load i64, ptr %7, align 8
  %27 = icmp slt i64 %26, -2147483648
  br i1 %27, label %28, label %29

28:                                               ; preds = %25
  store i64 -2147483648, ptr %7, align 8
  br label %34

29:                                               ; preds = %25
  %30 = load i64, ptr %7, align 8
  %31 = icmp sgt i64 %30, 2147483647
  br i1 %31, label %32, label %33

32:                                               ; preds = %29
  store i64 2147483647, ptr %7, align 8
  br label %33

33:                                               ; preds = %32, %29
  br label %34

34:                                               ; preds = %33, %28
  br label %35

35:                                               ; preds = %34, %19
  %36 = load i64, ptr %7, align 8
  %37 = trunc i64 %36 to i32
  %38 = call ptr @PyUnicode_FromOrdinal(i32 noundef %37)
  store ptr %38, ptr %3, align 8
  br label %39

39:                                               ; preds = %35, %15
  %40 = load ptr, ptr %3, align 8
  ret ptr %40
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_compile(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [7 x ptr], align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca i64, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store ptr null, ptr %9, align 8
  %20 = load i64, ptr %7, align 8
  %21 = load ptr, ptr %8, align 8
  %22 = icmp ne ptr %21, null
  br i1 %22, label %23, label %26

23:                                               ; preds = %4
  %24 = load ptr, ptr %8, align 8
  %25 = call i64 @PyTuple_GET_SIZE(ptr noundef %24)
  br label %27

26:                                               ; preds = %4
  br label %27

27:                                               ; preds = %26, %23
  %28 = phi i64 [ %25, %23 ], [ 0, %26 ]
  %29 = add nsw i64 %20, %28
  %30 = sub nsw i64 %29, 3
  store i64 %30, ptr %11, align 8
  store i32 0, ptr %15, align 4
  store i32 0, ptr %16, align 4
  store i32 -1, ptr %17, align 4
  store i32 -1, ptr %18, align 4
  %31 = load ptr, ptr %8, align 8
  %32 = icmp eq ptr %31, null
  br i1 %32, label %33, label %44

33:                                               ; preds = %27
  %34 = load i64, ptr %7, align 8
  %35 = icmp sle i64 3, %34
  br i1 %35, label %36, label %44

36:                                               ; preds = %33
  %37 = load i64, ptr %7, align 8
  %38 = icmp sle i64 %37, 6
  br i1 %38, label %39, label %44

39:                                               ; preds = %36
  %40 = load ptr, ptr %6, align 8
  %41 = icmp ne ptr %40, null
  br i1 %41, label %42, label %44

42:                                               ; preds = %39
  %43 = load ptr, ptr %6, align 8
  br label %50

44:                                               ; preds = %39, %36, %33, %27
  %45 = load ptr, ptr %6, align 8
  %46 = load i64, ptr %7, align 8
  %47 = load ptr, ptr %8, align 8
  %48 = getelementptr inbounds [7 x ptr], ptr %10, i64 0, i64 0
  %49 = call ptr @_PyArg_UnpackKeywords(ptr noundef %45, i64 noundef %46, ptr noundef null, ptr noundef %47, ptr noundef @builtin_compile._parser, i32 noundef 3, i32 noundef 6, i32 noundef 0, i32 noundef 0, ptr noundef %48)
  br label %50

50:                                               ; preds = %44, %42
  %51 = phi ptr [ %43, %42 ], [ %49, %44 ]
  store ptr %51, ptr %6, align 8
  %52 = load ptr, ptr %6, align 8
  %53 = icmp ne ptr %52, null
  br i1 %53, label %55, label %54

54:                                               ; preds = %50
  br label %185

55:                                               ; preds = %50
  %56 = load ptr, ptr %6, align 8
  %57 = getelementptr inbounds ptr, ptr %56, i64 0
  %58 = load ptr, ptr %57, align 8
  store ptr %58, ptr %12, align 8
  %59 = load ptr, ptr %6, align 8
  %60 = getelementptr inbounds ptr, ptr %59, i64 1
  %61 = load ptr, ptr %60, align 8
  %62 = call i32 @PyUnicode_FSDecoder(ptr noundef %61, ptr noundef %13)
  %63 = icmp ne i32 %62, 0
  br i1 %63, label %65, label %64

64:                                               ; preds = %55
  br label %185

65:                                               ; preds = %55
  %66 = load ptr, ptr %6, align 8
  %67 = getelementptr inbounds ptr, ptr %66, i64 2
  %68 = load ptr, ptr %67, align 8
  %69 = call ptr @_Py_TYPE(ptr noundef %68)
  %70 = call i32 @PyType_HasFeature(ptr noundef %69, i64 noundef 268435456)
  %71 = icmp ne i32 %70, 0
  br i1 %71, label %76, label %72

72:                                               ; preds = %65
  %73 = load ptr, ptr %6, align 8
  %74 = getelementptr inbounds ptr, ptr %73, i64 2
  %75 = load ptr, ptr %74, align 8
  call void @_PyArg_BadArgument(ptr noundef @.str.85, ptr noundef @.str.146, ptr noundef @.str.27, ptr noundef %75)
  br label %185

76:                                               ; preds = %65
  %77 = load ptr, ptr %6, align 8
  %78 = getelementptr inbounds ptr, ptr %77, i64 2
  %79 = load ptr, ptr %78, align 8
  %80 = call ptr @PyUnicode_AsUTF8AndSize(ptr noundef %79, ptr noundef %19)
  store ptr %80, ptr %14, align 8
  %81 = load ptr, ptr %14, align 8
  %82 = icmp eq ptr %81, null
  br i1 %82, label %83, label %84

83:                                               ; preds = %76
  br label %185

84:                                               ; preds = %76
  %85 = load ptr, ptr %14, align 8
  %86 = call i64 @strlen(ptr noundef %85) #7
  %87 = load i64, ptr %19, align 8
  %88 = icmp ne i64 %86, %87
  br i1 %88, label %89, label %91

89:                                               ; preds = %84
  %90 = load ptr, ptr @PyExc_ValueError, align 8
  call void @PyErr_SetString(ptr noundef %90, ptr noundef @.str.147)
  br label %185

91:                                               ; preds = %84
  %92 = load i64, ptr %11, align 8
  %93 = icmp ne i64 %92, 0
  br i1 %93, label %95, label %94

94:                                               ; preds = %91
  br label %159

95:                                               ; preds = %91
  %96 = load ptr, ptr %6, align 8
  %97 = getelementptr inbounds ptr, ptr %96, i64 3
  %98 = load ptr, ptr %97, align 8
  %99 = icmp ne ptr %98, null
  br i1 %99, label %100, label %117

100:                                              ; preds = %95
  %101 = load ptr, ptr %6, align 8
  %102 = getelementptr inbounds ptr, ptr %101, i64 3
  %103 = load ptr, ptr %102, align 8
  %104 = call i32 @PyLong_AsInt(ptr noundef %103)
  store i32 %104, ptr %15, align 4
  %105 = load i32, ptr %15, align 4
  %106 = icmp eq i32 %105, -1
  br i1 %106, label %107, label %111

107:                                              ; preds = %100
  %108 = call ptr @PyErr_Occurred()
  %109 = icmp ne ptr %108, null
  br i1 %109, label %110, label %111

110:                                              ; preds = %107
  br label %185

111:                                              ; preds = %107, %100
  %112 = load i64, ptr %11, align 8
  %113 = add nsw i64 %112, -1
  store i64 %113, ptr %11, align 8
  %114 = icmp ne i64 %113, 0
  br i1 %114, label %116, label %115

115:                                              ; preds = %111
  br label %159

116:                                              ; preds = %111
  br label %117

117:                                              ; preds = %116, %95
  %118 = load ptr, ptr %6, align 8
  %119 = getelementptr inbounds ptr, ptr %118, i64 4
  %120 = load ptr, ptr %119, align 8
  %121 = icmp ne ptr %120, null
  br i1 %121, label %122, label %136

122:                                              ; preds = %117
  %123 = load ptr, ptr %6, align 8
  %124 = getelementptr inbounds ptr, ptr %123, i64 4
  %125 = load ptr, ptr %124, align 8
  %126 = call i32 @PyObject_IsTrue(ptr noundef %125)
  store i32 %126, ptr %16, align 4
  %127 = load i32, ptr %16, align 4
  %128 = icmp slt i32 %127, 0
  br i1 %128, label %129, label %130

129:                                              ; preds = %122
  br label %185

130:                                              ; preds = %122
  %131 = load i64, ptr %11, align 8
  %132 = add nsw i64 %131, -1
  store i64 %132, ptr %11, align 8
  %133 = icmp ne i64 %132, 0
  br i1 %133, label %135, label %134

134:                                              ; preds = %130
  br label %159

135:                                              ; preds = %130
  br label %136

136:                                              ; preds = %135, %117
  %137 = load ptr, ptr %6, align 8
  %138 = getelementptr inbounds ptr, ptr %137, i64 5
  %139 = load ptr, ptr %138, align 8
  %140 = icmp ne ptr %139, null
  br i1 %140, label %141, label %158

141:                                              ; preds = %136
  %142 = load ptr, ptr %6, align 8
  %143 = getelementptr inbounds ptr, ptr %142, i64 5
  %144 = load ptr, ptr %143, align 8
  %145 = call i32 @PyLong_AsInt(ptr noundef %144)
  store i32 %145, ptr %17, align 4
  %146 = load i32, ptr %17, align 4
  %147 = icmp eq i32 %146, -1
  br i1 %147, label %148, label %152

148:                                              ; preds = %141
  %149 = call ptr @PyErr_Occurred()
  %150 = icmp ne ptr %149, null
  br i1 %150, label %151, label %152

151:                                              ; preds = %148
  br label %185

152:                                              ; preds = %148, %141
  %153 = load i64, ptr %11, align 8
  %154 = add nsw i64 %153, -1
  store i64 %154, ptr %11, align 8
  %155 = icmp ne i64 %154, 0
  br i1 %155, label %157, label %156

156:                                              ; preds = %152
  br label %159

157:                                              ; preds = %152
  br label %158

158:                                              ; preds = %157, %136
  br label %159

159:                                              ; preds = %158, %156, %134, %115, %94
  %160 = load i64, ptr %11, align 8
  %161 = icmp ne i64 %160, 0
  br i1 %161, label %163, label %162

162:                                              ; preds = %159
  br label %175

163:                                              ; preds = %159
  %164 = load ptr, ptr %6, align 8
  %165 = getelementptr inbounds ptr, ptr %164, i64 6
  %166 = load ptr, ptr %165, align 8
  %167 = call i32 @PyLong_AsInt(ptr noundef %166)
  store i32 %167, ptr %18, align 4
  %168 = load i32, ptr %18, align 4
  %169 = icmp eq i32 %168, -1
  br i1 %169, label %170, label %174

170:                                              ; preds = %163
  %171 = call ptr @PyErr_Occurred()
  %172 = icmp ne ptr %171, null
  br i1 %172, label %173, label %174

173:                                              ; preds = %170
  br label %185

174:                                              ; preds = %170, %163
  br label %175

175:                                              ; preds = %174, %162
  %176 = load ptr, ptr %5, align 8
  %177 = load ptr, ptr %12, align 8
  %178 = load ptr, ptr %13, align 8
  %179 = load ptr, ptr %14, align 8
  %180 = load i32, ptr %15, align 4
  %181 = load i32, ptr %16, align 4
  %182 = load i32, ptr %17, align 4
  %183 = load i32, ptr %18, align 4
  %184 = call ptr @builtin_compile_impl(ptr noundef %176, ptr noundef %177, ptr noundef %178, ptr noundef %179, i32 noundef %180, i32 noundef %181, i32 noundef %182, i32 noundef %183)
  store ptr %184, ptr %9, align 8
  br label %185

185:                                              ; preds = %175, %173, %151, %129, %110, %89, %83, %72, %64, %54
  %186 = load ptr, ptr %9, align 8
  ret ptr %186
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_delattr(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  %10 = load i64, ptr %6, align 8
  %11 = icmp sle i64 2, %10
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = load i64, ptr %6, align 8
  %14 = icmp sle i64 %13, 2
  br i1 %14, label %20, label %15

15:                                               ; preds = %12, %3
  %16 = load i64, ptr %6, align 8
  %17 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.86, i64 noundef %16, i64 noundef 2, i64 noundef 2)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %15
  br label %31

20:                                               ; preds = %15, %12
  %21 = load ptr, ptr %5, align 8
  %22 = getelementptr inbounds ptr, ptr %21, i64 0
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %8, align 8
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr inbounds ptr, ptr %24, i64 1
  %26 = load ptr, ptr %25, align 8
  store ptr %26, ptr %9, align 8
  %27 = load ptr, ptr %4, align 8
  %28 = load ptr, ptr %8, align 8
  %29 = load ptr, ptr %9, align 8
  %30 = call ptr @builtin_delattr_impl(ptr noundef %27, ptr noundef %28, ptr noundef %29)
  store ptr %30, ptr %7, align 8
  br label %31

31:                                               ; preds = %20, %19
  %32 = load ptr, ptr %7, align 8
  ret ptr %32
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_dir(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr null, ptr %6, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr noundef %7, ptr noundef @.str.87, i64 noundef 0, i64 noundef 1, ptr noundef %6)
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %2
  store ptr null, ptr %3, align 8
  br label %14

11:                                               ; preds = %2
  %12 = load ptr, ptr %6, align 8
  %13 = call ptr @PyObject_Dir(ptr noundef %12)
  store ptr %13, ptr %3, align 8
  br label %14

14:                                               ; preds = %11, %10
  %15 = load ptr, ptr %3, align 8
  ret ptr %15
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_divmod(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  %10 = load i64, ptr %6, align 8
  %11 = icmp sle i64 2, %10
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = load i64, ptr %6, align 8
  %14 = icmp sle i64 %13, 2
  br i1 %14, label %20, label %15

15:                                               ; preds = %12, %3
  %16 = load i64, ptr %6, align 8
  %17 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.88, i64 noundef %16, i64 noundef 2, i64 noundef 2)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %15
  br label %31

20:                                               ; preds = %15, %12
  %21 = load ptr, ptr %5, align 8
  %22 = getelementptr inbounds ptr, ptr %21, i64 0
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %8, align 8
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr inbounds ptr, ptr %24, i64 1
  %26 = load ptr, ptr %25, align 8
  store ptr %26, ptr %9, align 8
  %27 = load ptr, ptr %4, align 8
  %28 = load ptr, ptr %8, align 8
  %29 = load ptr, ptr %9, align 8
  %30 = call ptr @builtin_divmod_impl(ptr noundef %27, ptr noundef %28, ptr noundef %29)
  store ptr %30, ptr %7, align 8
  br label %31

31:                                               ; preds = %20, %19
  %32 = load ptr, ptr %7, align 8
  ret ptr %32
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_eval(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [3 x ptr], align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store ptr null, ptr %9, align 8
  %15 = load i64, ptr %7, align 8
  %16 = load ptr, ptr %8, align 8
  %17 = icmp ne ptr %16, null
  br i1 %17, label %18, label %21

18:                                               ; preds = %4
  %19 = load ptr, ptr %8, align 8
  %20 = call i64 @PyTuple_GET_SIZE(ptr noundef %19)
  br label %22

21:                                               ; preds = %4
  br label %22

22:                                               ; preds = %21, %18
  %23 = phi i64 [ %20, %18 ], [ 0, %21 ]
  %24 = add nsw i64 %15, %23
  %25 = sub nsw i64 %24, 1
  store i64 %25, ptr %11, align 8
  store ptr @_Py_NoneStruct, ptr %13, align 8
  store ptr @_Py_NoneStruct, ptr %14, align 8
  %26 = load ptr, ptr %8, align 8
  %27 = icmp eq ptr %26, null
  br i1 %27, label %28, label %39

28:                                               ; preds = %22
  %29 = load i64, ptr %7, align 8
  %30 = icmp sle i64 1, %29
  br i1 %30, label %31, label %39

31:                                               ; preds = %28
  %32 = load i64, ptr %7, align 8
  %33 = icmp sle i64 %32, 3
  br i1 %33, label %34, label %39

34:                                               ; preds = %31
  %35 = load ptr, ptr %6, align 8
  %36 = icmp ne ptr %35, null
  br i1 %36, label %37, label %39

37:                                               ; preds = %34
  %38 = load ptr, ptr %6, align 8
  br label %45

39:                                               ; preds = %34, %31, %28, %22
  %40 = load ptr, ptr %6, align 8
  %41 = load i64, ptr %7, align 8
  %42 = load ptr, ptr %8, align 8
  %43 = getelementptr inbounds [3 x ptr], ptr %10, i64 0, i64 0
  %44 = call ptr @_PyArg_UnpackKeywords(ptr noundef %40, i64 noundef %41, ptr noundef null, ptr noundef %42, ptr noundef @builtin_eval._parser, i32 noundef 1, i32 noundef 3, i32 noundef 0, i32 noundef 0, ptr noundef %43)
  br label %45

45:                                               ; preds = %39, %37
  %46 = phi ptr [ %38, %37 ], [ %44, %39 ]
  store ptr %46, ptr %6, align 8
  %47 = load ptr, ptr %6, align 8
  %48 = icmp ne ptr %47, null
  br i1 %48, label %50, label %49

49:                                               ; preds = %45
  br label %81

50:                                               ; preds = %45
  %51 = load ptr, ptr %6, align 8
  %52 = getelementptr inbounds ptr, ptr %51, i64 0
  %53 = load ptr, ptr %52, align 8
  store ptr %53, ptr %12, align 8
  %54 = load i64, ptr %11, align 8
  %55 = icmp ne i64 %54, 0
  br i1 %55, label %57, label %56

56:                                               ; preds = %50
  br label %75

57:                                               ; preds = %50
  %58 = load ptr, ptr %6, align 8
  %59 = getelementptr inbounds ptr, ptr %58, i64 1
  %60 = load ptr, ptr %59, align 8
  %61 = icmp ne ptr %60, null
  br i1 %61, label %62, label %71

62:                                               ; preds = %57
  %63 = load ptr, ptr %6, align 8
  %64 = getelementptr inbounds ptr, ptr %63, i64 1
  %65 = load ptr, ptr %64, align 8
  store ptr %65, ptr %13, align 8
  %66 = load i64, ptr %11, align 8
  %67 = add nsw i64 %66, -1
  store i64 %67, ptr %11, align 8
  %68 = icmp ne i64 %67, 0
  br i1 %68, label %70, label %69

69:                                               ; preds = %62
  br label %75

70:                                               ; preds = %62
  br label %71

71:                                               ; preds = %70, %57
  %72 = load ptr, ptr %6, align 8
  %73 = getelementptr inbounds ptr, ptr %72, i64 2
  %74 = load ptr, ptr %73, align 8
  store ptr %74, ptr %14, align 8
  br label %75

75:                                               ; preds = %71, %69, %56
  %76 = load ptr, ptr %5, align 8
  %77 = load ptr, ptr %12, align 8
  %78 = load ptr, ptr %13, align 8
  %79 = load ptr, ptr %14, align 8
  %80 = call ptr @builtin_eval_impl(ptr noundef %76, ptr noundef %77, ptr noundef %78, ptr noundef %79)
  store ptr %80, ptr %9, align 8
  br label %81

81:                                               ; preds = %75, %49
  %82 = load ptr, ptr %9, align 8
  ret ptr %82
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_exec(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [4 x ptr], align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store ptr null, ptr %9, align 8
  %16 = load i64, ptr %7, align 8
  %17 = load ptr, ptr %8, align 8
  %18 = icmp ne ptr %17, null
  br i1 %18, label %19, label %22

19:                                               ; preds = %4
  %20 = load ptr, ptr %8, align 8
  %21 = call i64 @PyTuple_GET_SIZE(ptr noundef %20)
  br label %23

22:                                               ; preds = %4
  br label %23

23:                                               ; preds = %22, %19
  %24 = phi i64 [ %21, %19 ], [ 0, %22 ]
  %25 = add nsw i64 %16, %24
  %26 = sub nsw i64 %25, 1
  store i64 %26, ptr %11, align 8
  store ptr @_Py_NoneStruct, ptr %13, align 8
  store ptr @_Py_NoneStruct, ptr %14, align 8
  store ptr null, ptr %15, align 8
  %27 = load ptr, ptr %8, align 8
  %28 = icmp eq ptr %27, null
  br i1 %28, label %29, label %40

29:                                               ; preds = %23
  %30 = load i64, ptr %7, align 8
  %31 = icmp sle i64 1, %30
  br i1 %31, label %32, label %40

32:                                               ; preds = %29
  %33 = load i64, ptr %7, align 8
  %34 = icmp sle i64 %33, 3
  br i1 %34, label %35, label %40

35:                                               ; preds = %32
  %36 = load ptr, ptr %6, align 8
  %37 = icmp ne ptr %36, null
  br i1 %37, label %38, label %40

38:                                               ; preds = %35
  %39 = load ptr, ptr %6, align 8
  br label %46

40:                                               ; preds = %35, %32, %29, %23
  %41 = load ptr, ptr %6, align 8
  %42 = load i64, ptr %7, align 8
  %43 = load ptr, ptr %8, align 8
  %44 = getelementptr inbounds [4 x ptr], ptr %10, i64 0, i64 0
  %45 = call ptr @_PyArg_UnpackKeywords(ptr noundef %41, i64 noundef %42, ptr noundef null, ptr noundef %43, ptr noundef @builtin_exec._parser, i32 noundef 1, i32 noundef 3, i32 noundef 0, i32 noundef 0, ptr noundef %44)
  br label %46

46:                                               ; preds = %40, %38
  %47 = phi ptr [ %39, %38 ], [ %45, %40 ]
  store ptr %47, ptr %6, align 8
  %48 = load ptr, ptr %6, align 8
  %49 = icmp ne ptr %48, null
  br i1 %49, label %51, label %50

50:                                               ; preds = %46
  br label %102

51:                                               ; preds = %46
  %52 = load ptr, ptr %6, align 8
  %53 = getelementptr inbounds ptr, ptr %52, i64 0
  %54 = load ptr, ptr %53, align 8
  store ptr %54, ptr %12, align 8
  %55 = load i64, ptr %11, align 8
  %56 = icmp ne i64 %55, 0
  br i1 %56, label %58, label %57

57:                                               ; preds = %51
  br label %87

58:                                               ; preds = %51
  %59 = load ptr, ptr %6, align 8
  %60 = getelementptr inbounds ptr, ptr %59, i64 1
  %61 = load ptr, ptr %60, align 8
  %62 = icmp ne ptr %61, null
  br i1 %62, label %63, label %72

63:                                               ; preds = %58
  %64 = load ptr, ptr %6, align 8
  %65 = getelementptr inbounds ptr, ptr %64, i64 1
  %66 = load ptr, ptr %65, align 8
  store ptr %66, ptr %13, align 8
  %67 = load i64, ptr %11, align 8
  %68 = add nsw i64 %67, -1
  store i64 %68, ptr %11, align 8
  %69 = icmp ne i64 %68, 0
  br i1 %69, label %71, label %70

70:                                               ; preds = %63
  br label %87

71:                                               ; preds = %63
  br label %72

72:                                               ; preds = %71, %58
  %73 = load ptr, ptr %6, align 8
  %74 = getelementptr inbounds ptr, ptr %73, i64 2
  %75 = load ptr, ptr %74, align 8
  %76 = icmp ne ptr %75, null
  br i1 %76, label %77, label %86

77:                                               ; preds = %72
  %78 = load ptr, ptr %6, align 8
  %79 = getelementptr inbounds ptr, ptr %78, i64 2
  %80 = load ptr, ptr %79, align 8
  store ptr %80, ptr %14, align 8
  %81 = load i64, ptr %11, align 8
  %82 = add nsw i64 %81, -1
  store i64 %82, ptr %11, align 8
  %83 = icmp ne i64 %82, 0
  br i1 %83, label %85, label %84

84:                                               ; preds = %77
  br label %87

85:                                               ; preds = %77
  br label %86

86:                                               ; preds = %85, %72
  br label %87

87:                                               ; preds = %86, %84, %70, %57
  %88 = load i64, ptr %11, align 8
  %89 = icmp ne i64 %88, 0
  br i1 %89, label %91, label %90

90:                                               ; preds = %87
  br label %95

91:                                               ; preds = %87
  %92 = load ptr, ptr %6, align 8
  %93 = getelementptr inbounds ptr, ptr %92, i64 3
  %94 = load ptr, ptr %93, align 8
  store ptr %94, ptr %15, align 8
  br label %95

95:                                               ; preds = %91, %90
  %96 = load ptr, ptr %5, align 8
  %97 = load ptr, ptr %12, align 8
  %98 = load ptr, ptr %13, align 8
  %99 = load ptr, ptr %14, align 8
  %100 = load ptr, ptr %15, align 8
  %101 = call ptr @builtin_exec_impl(ptr noundef %96, ptr noundef %97, ptr noundef %98, ptr noundef %99, ptr noundef %100)
  store ptr %101, ptr %9, align 8
  br label %102

102:                                              ; preds = %95, %50
  %103 = load ptr, ptr %9, align 8
  ret ptr %103
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_format(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  store ptr null, ptr %9, align 8
  %10 = load i64, ptr %6, align 8
  %11 = icmp sle i64 1, %10
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = load i64, ptr %6, align 8
  %14 = icmp sle i64 %13, 2
  br i1 %14, label %20, label %15

15:                                               ; preds = %12, %3
  %16 = load i64, ptr %6, align 8
  %17 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.91, i64 noundef %16, i64 noundef 1, i64 noundef 2)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %15
  br label %47

20:                                               ; preds = %15, %12
  %21 = load ptr, ptr %5, align 8
  %22 = getelementptr inbounds ptr, ptr %21, i64 0
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %8, align 8
  %24 = load i64, ptr %6, align 8
  %25 = icmp slt i64 %24, 2
  br i1 %25, label %26, label %27

26:                                               ; preds = %20
  br label %42

27:                                               ; preds = %20
  %28 = load ptr, ptr %5, align 8
  %29 = getelementptr inbounds ptr, ptr %28, i64 1
  %30 = load ptr, ptr %29, align 8
  %31 = call ptr @_Py_TYPE(ptr noundef %30)
  %32 = call i32 @PyType_HasFeature(ptr noundef %31, i64 noundef 268435456)
  %33 = icmp ne i32 %32, 0
  br i1 %33, label %38, label %34

34:                                               ; preds = %27
  %35 = load ptr, ptr %5, align 8
  %36 = getelementptr inbounds ptr, ptr %35, i64 1
  %37 = load ptr, ptr %36, align 8
  call void @_PyArg_BadArgument(ptr noundef @.str.91, ptr noundef @.str.173, ptr noundef @.str.27, ptr noundef %37)
  br label %47

38:                                               ; preds = %27
  %39 = load ptr, ptr %5, align 8
  %40 = getelementptr inbounds ptr, ptr %39, i64 1
  %41 = load ptr, ptr %40, align 8
  store ptr %41, ptr %9, align 8
  br label %42

42:                                               ; preds = %38, %26
  %43 = load ptr, ptr %4, align 8
  %44 = load ptr, ptr %8, align 8
  %45 = load ptr, ptr %9, align 8
  %46 = call ptr @builtin_format_impl(ptr noundef %43, ptr noundef %44, ptr noundef %45)
  store ptr %46, ptr %7, align 8
  br label %47

47:                                               ; preds = %42, %34, %19
  %48 = load ptr, ptr %7, align 8
  ret ptr %48
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_getattr(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  %12 = load i64, ptr %7, align 8
  %13 = icmp sle i64 2, %12
  br i1 %13, label %14, label %17

14:                                               ; preds = %3
  %15 = load i64, ptr %7, align 8
  %16 = icmp sle i64 %15, 3
  br i1 %16, label %22, label %17

17:                                               ; preds = %14, %3
  %18 = load i64, ptr %7, align 8
  %19 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.92, i64 noundef %18, i64 noundef 2, i64 noundef 3)
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %22, label %21

21:                                               ; preds = %17
  store ptr null, ptr %4, align 8
  br label %49

22:                                               ; preds = %17, %14
  %23 = load ptr, ptr %6, align 8
  %24 = getelementptr inbounds ptr, ptr %23, i64 0
  %25 = load ptr, ptr %24, align 8
  store ptr %25, ptr %8, align 8
  %26 = load ptr, ptr %6, align 8
  %27 = getelementptr inbounds ptr, ptr %26, i64 1
  %28 = load ptr, ptr %27, align 8
  store ptr %28, ptr %9, align 8
  %29 = load i64, ptr %7, align 8
  %30 = icmp sgt i64 %29, 2
  br i1 %30, label %31, label %43

31:                                               ; preds = %22
  %32 = load ptr, ptr %8, align 8
  %33 = load ptr, ptr %9, align 8
  %34 = call i32 @PyObject_GetOptionalAttr(ptr noundef %32, ptr noundef %33, ptr noundef %10)
  %35 = icmp eq i32 %34, 0
  br i1 %35, label %36, label %42

36:                                               ; preds = %31
  %37 = load ptr, ptr %6, align 8
  %38 = getelementptr inbounds ptr, ptr %37, i64 2
  %39 = load ptr, ptr %38, align 8
  store ptr %39, ptr %11, align 8
  %40 = load ptr, ptr %11, align 8
  %41 = call ptr @_Py_NewRef(ptr noundef %40)
  store ptr %41, ptr %4, align 8
  br label %49

42:                                               ; preds = %31
  br label %47

43:                                               ; preds = %22
  %44 = load ptr, ptr %8, align 8
  %45 = load ptr, ptr %9, align 8
  %46 = call ptr @PyObject_GetAttr(ptr noundef %44, ptr noundef %45)
  store ptr %46, ptr %10, align 8
  br label %47

47:                                               ; preds = %43, %42
  %48 = load ptr, ptr %10, align 8
  store ptr %48, ptr %4, align 8
  br label %49

49:                                               ; preds = %47, %36, %21
  %50 = load ptr, ptr %4, align 8
  ret ptr %50
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_globals(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = call ptr @builtin_globals_impl(ptr noundef %5)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_hasattr(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  %10 = load i64, ptr %6, align 8
  %11 = icmp sle i64 2, %10
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = load i64, ptr %6, align 8
  %14 = icmp sle i64 %13, 2
  br i1 %14, label %20, label %15

15:                                               ; preds = %12, %3
  %16 = load i64, ptr %6, align 8
  %17 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.94, i64 noundef %16, i64 noundef 2, i64 noundef 2)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %15
  br label %31

20:                                               ; preds = %15, %12
  %21 = load ptr, ptr %5, align 8
  %22 = getelementptr inbounds ptr, ptr %21, i64 0
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %8, align 8
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr inbounds ptr, ptr %24, i64 1
  %26 = load ptr, ptr %25, align 8
  store ptr %26, ptr %9, align 8
  %27 = load ptr, ptr %4, align 8
  %28 = load ptr, ptr %8, align 8
  %29 = load ptr, ptr %9, align 8
  %30 = call ptr @builtin_hasattr_impl(ptr noundef %27, ptr noundef %28, ptr noundef %29)
  store ptr %30, ptr %7, align 8
  br label %31

31:                                               ; preds = %20, %19
  %32 = load ptr, ptr %7, align 8
  ret ptr %32
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_hash(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = call i64 @PyObject_Hash(ptr noundef %7)
  store i64 %8, ptr %6, align 8
  %9 = load i64, ptr %6, align 8
  %10 = icmp eq i64 %9, -1
  br i1 %10, label %11, label %12

11:                                               ; preds = %2
  store ptr null, ptr %3, align 8
  br label %15

12:                                               ; preds = %2
  %13 = load i64, ptr %6, align 8
  %14 = call ptr @PyLong_FromSsize_t(i64 noundef %13)
  store ptr %14, ptr %3, align 8
  br label %15

15:                                               ; preds = %12, %11
  %16 = load ptr, ptr %3, align 8
  ret ptr %16
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_hex(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call ptr @PyNumber_ToBase(ptr noundef %5, i32 noundef 16)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_id(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store ptr %1, ptr %7, align 8
  %9 = load ptr, ptr %7, align 8
  %10 = call ptr @PyLong_FromVoidPtr(ptr noundef %9)
  store ptr %10, ptr %8, align 8
  %11 = load ptr, ptr %8, align 8
  %12 = icmp ne ptr %11, null
  br i1 %12, label %13, label %34

13:                                               ; preds = %2
  %14 = load ptr, ptr %8, align 8
  %15 = call i32 (ptr, ptr, ...) @PySys_Audit(ptr noundef @.str.174, ptr noundef @.str.138, ptr noundef %14)
  %16 = icmp slt i32 %15, 0
  br i1 %16, label %17, label %34

17:                                               ; preds = %13
  %18 = load ptr, ptr %8, align 8
  store ptr %18, ptr %4, align 8
  %19 = load ptr, ptr %4, align 8
  store ptr %19, ptr %3, align 8
  %20 = load ptr, ptr %3, align 8
  %21 = load i32, ptr %20, align 8
  %22 = icmp slt i32 %21, 0
  %23 = zext i1 %22 to i32
  %24 = icmp ne i32 %23, 0
  br i1 %24, label %25, label %26

25:                                               ; preds = %17
  br label %33

26:                                               ; preds = %17
  %27 = load ptr, ptr %4, align 8
  %28 = load i32, ptr %27, align 8
  %29 = add i32 %28, -1
  store i32 %29, ptr %27, align 8
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %31, label %33

31:                                               ; preds = %26
  %32 = load ptr, ptr %4, align 8
  call void @_Py_Dealloc(ptr noundef %32) #7
  br label %33

33:                                               ; preds = %25, %26, %31
  store ptr null, ptr %5, align 8
  br label %36

34:                                               ; preds = %13, %2
  %35 = load ptr, ptr %8, align 8
  store ptr %35, ptr %5, align 8
  br label %36

36:                                               ; preds = %34, %33
  %37 = load ptr, ptr %5, align 8
  ret ptr %37
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_input(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  store ptr null, ptr %8, align 8
  %9 = load i64, ptr %6, align 8
  %10 = icmp sle i64 0, %9
  br i1 %10, label %11, label %14

11:                                               ; preds = %3
  %12 = load i64, ptr %6, align 8
  %13 = icmp sle i64 %12, 1
  br i1 %13, label %19, label %14

14:                                               ; preds = %11, %3
  %15 = load i64, ptr %6, align 8
  %16 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.98, i64 noundef %15, i64 noundef 0, i64 noundef 1)
  %17 = icmp ne i32 %16, 0
  br i1 %17, label %19, label %18

18:                                               ; preds = %14
  br label %31

19:                                               ; preds = %14, %11
  %20 = load i64, ptr %6, align 8
  %21 = icmp slt i64 %20, 1
  br i1 %21, label %22, label %23

22:                                               ; preds = %19
  br label %27

23:                                               ; preds = %19
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr inbounds ptr, ptr %24, i64 0
  %26 = load ptr, ptr %25, align 8
  store ptr %26, ptr %8, align 8
  br label %27

27:                                               ; preds = %23, %22
  %28 = load ptr, ptr %4, align 8
  %29 = load ptr, ptr %8, align 8
  %30 = call ptr @builtin_input_impl(ptr noundef %28, ptr noundef %29)
  store ptr %30, ptr %7, align 8
  br label %31

31:                                               ; preds = %27, %18
  %32 = load ptr, ptr %7, align 8
  ret ptr %32
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_isinstance(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  %10 = load i64, ptr %6, align 8
  %11 = icmp sle i64 2, %10
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = load i64, ptr %6, align 8
  %14 = icmp sle i64 %13, 2
  br i1 %14, label %20, label %15

15:                                               ; preds = %12, %3
  %16 = load i64, ptr %6, align 8
  %17 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.99, i64 noundef %16, i64 noundef 2, i64 noundef 2)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %15
  br label %31

20:                                               ; preds = %15, %12
  %21 = load ptr, ptr %5, align 8
  %22 = getelementptr inbounds ptr, ptr %21, i64 0
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %8, align 8
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr inbounds ptr, ptr %24, i64 1
  %26 = load ptr, ptr %25, align 8
  store ptr %26, ptr %9, align 8
  %27 = load ptr, ptr %4, align 8
  %28 = load ptr, ptr %8, align 8
  %29 = load ptr, ptr %9, align 8
  %30 = call ptr @builtin_isinstance_impl(ptr noundef %27, ptr noundef %28, ptr noundef %29)
  store ptr %30, ptr %7, align 8
  br label %31

31:                                               ; preds = %20, %19
  %32 = load ptr, ptr %7, align 8
  ret ptr %32
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_issubclass(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  %10 = load i64, ptr %6, align 8
  %11 = icmp sle i64 2, %10
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = load i64, ptr %6, align 8
  %14 = icmp sle i64 %13, 2
  br i1 %14, label %20, label %15

15:                                               ; preds = %12, %3
  %16 = load i64, ptr %6, align 8
  %17 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.100, i64 noundef %16, i64 noundef 2, i64 noundef 2)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %15
  br label %31

20:                                               ; preds = %15, %12
  %21 = load ptr, ptr %5, align 8
  %22 = getelementptr inbounds ptr, ptr %21, i64 0
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %8, align 8
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr inbounds ptr, ptr %24, i64 1
  %26 = load ptr, ptr %25, align 8
  store ptr %26, ptr %9, align 8
  %27 = load ptr, ptr %4, align 8
  %28 = load ptr, ptr %8, align 8
  %29 = load ptr, ptr %9, align 8
  %30 = call ptr @builtin_issubclass_impl(ptr noundef %27, ptr noundef %28, ptr noundef %29)
  store ptr %30, ptr %7, align 8
  br label %31

31:                                               ; preds = %20, %19
  %32 = load ptr, ptr %7, align 8
  ret ptr %32
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_iter(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  %10 = load i64, ptr %7, align 8
  %11 = icmp sle i64 1, %10
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = load i64, ptr %7, align 8
  %14 = icmp sle i64 %13, 2
  br i1 %14, label %20, label %15

15:                                               ; preds = %12, %3
  %16 = load i64, ptr %7, align 8
  %17 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.101, i64 noundef %16, i64 noundef 1, i64 noundef 2)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %15
  store ptr null, ptr %4, align 8
  br label %42

20:                                               ; preds = %15, %12
  %21 = load ptr, ptr %6, align 8
  %22 = getelementptr inbounds ptr, ptr %21, i64 0
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %8, align 8
  %24 = load i64, ptr %7, align 8
  %25 = icmp eq i64 %24, 1
  br i1 %25, label %26, label %29

26:                                               ; preds = %20
  %27 = load ptr, ptr %8, align 8
  %28 = call ptr @PyObject_GetIter(ptr noundef %27)
  store ptr %28, ptr %4, align 8
  br label %42

29:                                               ; preds = %20
  %30 = load ptr, ptr %8, align 8
  %31 = call i32 @PyCallable_Check(ptr noundef %30)
  %32 = icmp ne i32 %31, 0
  br i1 %32, label %35, label %33

33:                                               ; preds = %29
  %34 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %34, ptr noundef @.str.185)
  store ptr null, ptr %4, align 8
  br label %42

35:                                               ; preds = %29
  %36 = load ptr, ptr %6, align 8
  %37 = getelementptr inbounds ptr, ptr %36, i64 1
  %38 = load ptr, ptr %37, align 8
  store ptr %38, ptr %9, align 8
  %39 = load ptr, ptr %8, align 8
  %40 = load ptr, ptr %9, align 8
  %41 = call ptr @PyCallIter_New(ptr noundef %39, ptr noundef %40)
  store ptr %41, ptr %4, align 8
  br label %42

42:                                               ; preds = %35, %33, %26, %19
  %43 = load ptr, ptr %4, align 8
  ret ptr %43
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_aiter(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call ptr @PyObject_GetAIter(ptr noundef %5)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_len(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = call i64 @PyObject_Size(ptr noundef %7)
  store i64 %8, ptr %6, align 8
  %9 = load i64, ptr %6, align 8
  %10 = icmp slt i64 %9, 0
  br i1 %10, label %11, label %22

11:                                               ; preds = %2
  %12 = call ptr @PyErr_Occurred()
  %13 = icmp ne ptr %12, null
  %14 = xor i1 %13, true
  %15 = zext i1 %14 to i32
  %16 = sext i32 %15 to i64
  %17 = icmp ne i64 %16, 0
  br i1 %17, label %18, label %20

18:                                               ; preds = %11
  call void @__assert_rtn(ptr noundef @__func__.builtin_len, ptr noundef @.str.34, i32 noundef 1861, ptr noundef @.str.186) #8
  unreachable

19:                                               ; No predecessors!
  br label %21

20:                                               ; preds = %11
  br label %21

21:                                               ; preds = %20, %19
  store ptr null, ptr %3, align 8
  br label %25

22:                                               ; preds = %2
  %23 = load i64, ptr %6, align 8
  %24 = call ptr @PyLong_FromSsize_t(i64 noundef %23)
  store ptr %24, ptr %3, align 8
  br label %25

25:                                               ; preds = %22, %21
  %26 = load ptr, ptr %3, align 8
  ret ptr %26
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_locals(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = call ptr @builtin_locals_impl(ptr noundef %5)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_max(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = load i64, ptr %7, align 8
  %11 = load ptr, ptr %8, align 8
  %12 = call ptr @min_max(ptr noundef %9, i64 noundef %10, ptr noundef %11, i32 noundef 4)
  ret ptr %12
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_min(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = load i64, ptr %7, align 8
  %11 = load ptr, ptr %8, align 8
  %12 = call ptr @min_max(ptr noundef %9, i64 noundef %10, ptr noundef %11, i32 noundef 0)
  ret ptr %12
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_next(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  %11 = load i64, ptr %7, align 8
  %12 = icmp sle i64 1, %11
  br i1 %12, label %13, label %16

13:                                               ; preds = %3
  %14 = load i64, ptr %7, align 8
  %15 = icmp sle i64 %14, 2
  br i1 %15, label %21, label %16

16:                                               ; preds = %13, %3
  %17 = load i64, ptr %7, align 8
  %18 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.107, i64 noundef %17, i64 noundef 1, i64 noundef 2)
  %19 = icmp ne i32 %18, 0
  br i1 %19, label %21, label %20

20:                                               ; preds = %16
  store ptr null, ptr %4, align 8
  br label %70

21:                                               ; preds = %16, %13
  %22 = load ptr, ptr %6, align 8
  %23 = getelementptr inbounds ptr, ptr %22, i64 0
  %24 = load ptr, ptr %23, align 8
  store ptr %24, ptr %8, align 8
  %25 = load ptr, ptr %8, align 8
  %26 = call i32 @PyIter_Check(ptr noundef %25)
  %27 = icmp ne i32 %26, 0
  br i1 %27, label %35, label %28

28:                                               ; preds = %21
  %29 = load ptr, ptr @PyExc_TypeError, align 8
  %30 = load ptr, ptr %8, align 8
  %31 = call ptr @_Py_TYPE(ptr noundef %30)
  %32 = getelementptr inbounds %struct._typeobject, ptr %31, i32 0, i32 1
  %33 = load ptr, ptr %32, align 8
  %34 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %29, ptr noundef @.str.195, ptr noundef %33)
  store ptr null, ptr %4, align 8
  br label %70

35:                                               ; preds = %21
  %36 = load ptr, ptr %8, align 8
  %37 = call ptr @_Py_TYPE(ptr noundef %36)
  %38 = getelementptr inbounds %struct._typeobject, ptr %37, i32 0, i32 26
  %39 = load ptr, ptr %38, align 8
  %40 = load ptr, ptr %8, align 8
  %41 = call ptr %39(ptr noundef %40)
  store ptr %41, ptr %9, align 8
  %42 = load ptr, ptr %9, align 8
  %43 = icmp ne ptr %42, null
  br i1 %43, label %44, label %46

44:                                               ; preds = %35
  %45 = load ptr, ptr %9, align 8
  store ptr %45, ptr %4, align 8
  br label %70

46:                                               ; preds = %35
  %47 = load i64, ptr %7, align 8
  %48 = icmp sgt i64 %47, 1
  br i1 %48, label %49, label %64

49:                                               ; preds = %46
  %50 = load ptr, ptr %6, align 8
  %51 = getelementptr inbounds ptr, ptr %50, i64 1
  %52 = load ptr, ptr %51, align 8
  store ptr %52, ptr %10, align 8
  %53 = call ptr @PyErr_Occurred()
  %54 = icmp ne ptr %53, null
  br i1 %54, label %55, label %61

55:                                               ; preds = %49
  %56 = load ptr, ptr @PyExc_StopIteration, align 8
  %57 = call i32 @PyErr_ExceptionMatches(ptr noundef %56)
  %58 = icmp ne i32 %57, 0
  br i1 %58, label %60, label %59

59:                                               ; preds = %55
  store ptr null, ptr %4, align 8
  br label %70

60:                                               ; preds = %55
  call void @PyErr_Clear()
  br label %61

61:                                               ; preds = %60, %49
  %62 = load ptr, ptr %10, align 8
  %63 = call ptr @_Py_NewRef(ptr noundef %62)
  store ptr %63, ptr %4, align 8
  br label %70

64:                                               ; preds = %46
  %65 = call ptr @PyErr_Occurred()
  %66 = icmp ne ptr %65, null
  br i1 %66, label %67, label %68

67:                                               ; preds = %64
  store ptr null, ptr %4, align 8
  br label %70

68:                                               ; preds = %64
  %69 = load ptr, ptr @PyExc_StopIteration, align 8
  call void @PyErr_SetNone(ptr noundef %69)
  store ptr null, ptr %4, align 8
  br label %70

70:                                               ; preds = %68, %67, %61, %59, %44, %28, %20
  %71 = load ptr, ptr %4, align 8
  ret ptr %71
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_anext(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  store ptr null, ptr %9, align 8
  %10 = load i64, ptr %6, align 8
  %11 = icmp sle i64 1, %10
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = load i64, ptr %6, align 8
  %14 = icmp sle i64 %13, 2
  br i1 %14, label %20, label %15

15:                                               ; preds = %12, %3
  %16 = load i64, ptr %6, align 8
  %17 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.108, i64 noundef %16, i64 noundef 1, i64 noundef 2)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %15
  br label %36

20:                                               ; preds = %15, %12
  %21 = load ptr, ptr %5, align 8
  %22 = getelementptr inbounds ptr, ptr %21, i64 0
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %8, align 8
  %24 = load i64, ptr %6, align 8
  %25 = icmp slt i64 %24, 2
  br i1 %25, label %26, label %27

26:                                               ; preds = %20
  br label %31

27:                                               ; preds = %20
  %28 = load ptr, ptr %5, align 8
  %29 = getelementptr inbounds ptr, ptr %28, i64 1
  %30 = load ptr, ptr %29, align 8
  store ptr %30, ptr %9, align 8
  br label %31

31:                                               ; preds = %27, %26
  %32 = load ptr, ptr %4, align 8
  %33 = load ptr, ptr %8, align 8
  %34 = load ptr, ptr %9, align 8
  %35 = call ptr @builtin_anext_impl(ptr noundef %32, ptr noundef %33, ptr noundef %34)
  store ptr %35, ptr %7, align 8
  br label %36

36:                                               ; preds = %31, %19
  %37 = load ptr, ptr %7, align 8
  ret ptr %37
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_oct(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call ptr @PyNumber_ToBase(ptr noundef %5, i32 noundef 8)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_ord(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = call ptr @_Py_TYPE(ptr noundef %8)
  %10 = call i32 @PyType_HasFeature(ptr noundef %9, i64 noundef 134217728)
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %25

12:                                               ; preds = %2
  %13 = load ptr, ptr %5, align 8
  %14 = call i64 @PyBytes_GET_SIZE(ptr noundef %13)
  store i64 %14, ptr %7, align 8
  %15 = load i64, ptr %7, align 8
  %16 = icmp eq i64 %15, 1
  br i1 %16, label %17, label %24

17:                                               ; preds = %12
  %18 = load ptr, ptr %5, align 8
  %19 = call ptr @PyBytes_AS_STRING(ptr noundef %18)
  %20 = load i8, ptr %19, align 1
  %21 = zext i8 %20 to i64
  store i64 %21, ptr %6, align 8
  %22 = load i64, ptr %6, align 8
  %23 = call ptr @PyLong_FromLong(i64 noundef %22)
  store ptr %23, ptr %3, align 8
  br label %72

24:                                               ; preds = %12
  br label %68

25:                                               ; preds = %2
  %26 = load ptr, ptr %5, align 8
  %27 = call ptr @_Py_TYPE(ptr noundef %26)
  %28 = call i32 @PyType_HasFeature(ptr noundef %27, i64 noundef 268435456)
  %29 = icmp ne i32 %28, 0
  br i1 %29, label %30, label %42

30:                                               ; preds = %25
  %31 = load ptr, ptr %5, align 8
  %32 = call i64 @PyUnicode_GET_LENGTH(ptr noundef %31)
  store i64 %32, ptr %7, align 8
  %33 = load i64, ptr %7, align 8
  %34 = icmp eq i64 %33, 1
  br i1 %34, label %35, label %41

35:                                               ; preds = %30
  %36 = load ptr, ptr %5, align 8
  %37 = call i32 @PyUnicode_READ_CHAR(ptr noundef %36, i64 noundef 0)
  %38 = zext i32 %37 to i64
  store i64 %38, ptr %6, align 8
  %39 = load i64, ptr %6, align 8
  %40 = call ptr @PyLong_FromLong(i64 noundef %39)
  store ptr %40, ptr %3, align 8
  br label %72

41:                                               ; preds = %30
  br label %67

42:                                               ; preds = %25
  %43 = load ptr, ptr %5, align 8
  %44 = call i32 @PyObject_TypeCheck(ptr noundef %43, ptr noundef @PyByteArray_Type)
  %45 = icmp ne i32 %44, 0
  br i1 %45, label %46, label %59

46:                                               ; preds = %42
  %47 = load ptr, ptr %5, align 8
  %48 = call i64 @PyByteArray_GET_SIZE(ptr noundef %47)
  store i64 %48, ptr %7, align 8
  %49 = load i64, ptr %7, align 8
  %50 = icmp eq i64 %49, 1
  br i1 %50, label %51, label %58

51:                                               ; preds = %46
  %52 = load ptr, ptr %5, align 8
  %53 = call ptr @PyByteArray_AS_STRING(ptr noundef %52)
  %54 = load i8, ptr %53, align 1
  %55 = zext i8 %54 to i64
  store i64 %55, ptr %6, align 8
  %56 = load i64, ptr %6, align 8
  %57 = call ptr @PyLong_FromLong(i64 noundef %56)
  store ptr %57, ptr %3, align 8
  br label %72

58:                                               ; preds = %46
  br label %66

59:                                               ; preds = %42
  %60 = load ptr, ptr @PyExc_TypeError, align 8
  %61 = load ptr, ptr %5, align 8
  %62 = call ptr @_Py_TYPE(ptr noundef %61)
  %63 = getelementptr inbounds %struct._typeobject, ptr %62, i32 0, i32 1
  %64 = load ptr, ptr %63, align 8
  %65 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %60, ptr noundef @.str.197, ptr noundef %64)
  store ptr null, ptr %3, align 8
  br label %72

66:                                               ; preds = %58
  br label %67

67:                                               ; preds = %66, %41
  br label %68

68:                                               ; preds = %67, %24
  %69 = load ptr, ptr @PyExc_TypeError, align 8
  %70 = load i64, ptr %7, align 8
  %71 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %69, ptr noundef @.str.198, i64 noundef %70)
  store ptr null, ptr %3, align 8
  br label %72

72:                                               ; preds = %68, %59, %51, %35, %17
  %73 = load ptr, ptr %3, align 8
  ret ptr %73
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_pow(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [3 x ptr], align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store ptr null, ptr %9, align 8
  %15 = load i64, ptr %7, align 8
  %16 = load ptr, ptr %8, align 8
  %17 = icmp ne ptr %16, null
  br i1 %17, label %18, label %21

18:                                               ; preds = %4
  %19 = load ptr, ptr %8, align 8
  %20 = call i64 @PyTuple_GET_SIZE(ptr noundef %19)
  br label %22

21:                                               ; preds = %4
  br label %22

22:                                               ; preds = %21, %18
  %23 = phi i64 [ %20, %18 ], [ 0, %21 ]
  %24 = add nsw i64 %15, %23
  %25 = sub nsw i64 %24, 2
  store i64 %25, ptr %11, align 8
  store ptr @_Py_NoneStruct, ptr %14, align 8
  %26 = load ptr, ptr %8, align 8
  %27 = icmp eq ptr %26, null
  br i1 %27, label %28, label %39

28:                                               ; preds = %22
  %29 = load i64, ptr %7, align 8
  %30 = icmp sle i64 2, %29
  br i1 %30, label %31, label %39

31:                                               ; preds = %28
  %32 = load i64, ptr %7, align 8
  %33 = icmp sle i64 %32, 3
  br i1 %33, label %34, label %39

34:                                               ; preds = %31
  %35 = load ptr, ptr %6, align 8
  %36 = icmp ne ptr %35, null
  br i1 %36, label %37, label %39

37:                                               ; preds = %34
  %38 = load ptr, ptr %6, align 8
  br label %45

39:                                               ; preds = %34, %31, %28, %22
  %40 = load ptr, ptr %6, align 8
  %41 = load i64, ptr %7, align 8
  %42 = load ptr, ptr %8, align 8
  %43 = getelementptr inbounds [3 x ptr], ptr %10, i64 0, i64 0
  %44 = call ptr @_PyArg_UnpackKeywords(ptr noundef %40, i64 noundef %41, ptr noundef null, ptr noundef %42, ptr noundef @builtin_pow._parser, i32 noundef 2, i32 noundef 3, i32 noundef 0, i32 noundef 0, ptr noundef %43)
  br label %45

45:                                               ; preds = %39, %37
  %46 = phi ptr [ %38, %37 ], [ %44, %39 ]
  store ptr %46, ptr %6, align 8
  %47 = load ptr, ptr %6, align 8
  %48 = icmp ne ptr %47, null
  br i1 %48, label %50, label %49

49:                                               ; preds = %45
  br label %70

50:                                               ; preds = %45
  %51 = load ptr, ptr %6, align 8
  %52 = getelementptr inbounds ptr, ptr %51, i64 0
  %53 = load ptr, ptr %52, align 8
  store ptr %53, ptr %12, align 8
  %54 = load ptr, ptr %6, align 8
  %55 = getelementptr inbounds ptr, ptr %54, i64 1
  %56 = load ptr, ptr %55, align 8
  store ptr %56, ptr %13, align 8
  %57 = load i64, ptr %11, align 8
  %58 = icmp ne i64 %57, 0
  br i1 %58, label %60, label %59

59:                                               ; preds = %50
  br label %64

60:                                               ; preds = %50
  %61 = load ptr, ptr %6, align 8
  %62 = getelementptr inbounds ptr, ptr %61, i64 2
  %63 = load ptr, ptr %62, align 8
  store ptr %63, ptr %14, align 8
  br label %64

64:                                               ; preds = %60, %59
  %65 = load ptr, ptr %5, align 8
  %66 = load ptr, ptr %12, align 8
  %67 = load ptr, ptr %13, align 8
  %68 = load ptr, ptr %14, align 8
  %69 = call ptr @builtin_pow_impl(ptr noundef %65, ptr noundef %66, ptr noundef %67, ptr noundef %68)
  store ptr %69, ptr %9, align 8
  br label %70

70:                                               ; preds = %64, %49
  %71 = load ptr, ptr %9, align 8
  ret ptr %71
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_print(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [4 x ptr], align 8
  %11 = alloca ptr, align 8
  %12 = alloca i64, align 8
  %13 = alloca ptr, align 8
  %14 = alloca i64, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store ptr null, ptr %9, align 8
  %19 = load ptr, ptr %8, align 8
  %20 = icmp ne ptr %19, null
  br i1 %20, label %21, label %24

21:                                               ; preds = %4
  %22 = load ptr, ptr %8, align 8
  %23 = call i64 @PyTuple_GET_SIZE(ptr noundef %22)
  br label %25

24:                                               ; preds = %4
  br label %25

25:                                               ; preds = %24, %21
  %26 = phi i64 [ %23, %21 ], [ 0, %24 ]
  %27 = add nsw i64 0, %26
  %28 = sub nsw i64 %27, 0
  store i64 %28, ptr %12, align 8
  store ptr @_Py_NoneStruct, ptr %15, align 8
  store ptr @_Py_NoneStruct, ptr %16, align 8
  store ptr @_Py_NoneStruct, ptr %17, align 8
  store i32 0, ptr %18, align 4
  %29 = load ptr, ptr %8, align 8
  %30 = icmp eq ptr %29, null
  br i1 %30, label %31, label %39

31:                                               ; preds = %25
  %32 = load i64, ptr %7, align 8
  %33 = icmp sle i64 0, %32
  br i1 %33, label %34, label %39

34:                                               ; preds = %31
  %35 = load ptr, ptr %6, align 8
  %36 = icmp ne ptr %35, null
  br i1 %36, label %37, label %39

37:                                               ; preds = %34
  %38 = load ptr, ptr %6, align 8
  br label %45

39:                                               ; preds = %34, %31, %25
  %40 = load ptr, ptr %6, align 8
  %41 = load i64, ptr %7, align 8
  %42 = load ptr, ptr %8, align 8
  %43 = getelementptr inbounds [4 x ptr], ptr %10, i64 0, i64 0
  %44 = call ptr @_PyArg_UnpackKeywords(ptr noundef %40, i64 noundef %41, ptr noundef null, ptr noundef %42, ptr noundef @builtin_print._parser, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 1, ptr noundef %43)
  br label %45

45:                                               ; preds = %39, %37
  %46 = phi ptr [ %38, %37 ], [ %44, %39 ]
  store ptr %46, ptr %11, align 8
  %47 = load ptr, ptr %11, align 8
  %48 = icmp ne ptr %47, null
  br i1 %48, label %50, label %49

49:                                               ; preds = %45
  br label %116

50:                                               ; preds = %45
  %51 = load i64, ptr %12, align 8
  %52 = icmp ne i64 %51, 0
  br i1 %52, label %54, label %53

53:                                               ; preds = %50
  br label %105

54:                                               ; preds = %50
  %55 = load ptr, ptr %11, align 8
  %56 = getelementptr inbounds ptr, ptr %55, i64 0
  %57 = load ptr, ptr %56, align 8
  %58 = icmp ne ptr %57, null
  br i1 %58, label %59, label %68

59:                                               ; preds = %54
  %60 = load ptr, ptr %11, align 8
  %61 = getelementptr inbounds ptr, ptr %60, i64 0
  %62 = load ptr, ptr %61, align 8
  store ptr %62, ptr %15, align 8
  %63 = load i64, ptr %12, align 8
  %64 = add nsw i64 %63, -1
  store i64 %64, ptr %12, align 8
  %65 = icmp ne i64 %64, 0
  br i1 %65, label %67, label %66

66:                                               ; preds = %59
  br label %105

67:                                               ; preds = %59
  br label %68

68:                                               ; preds = %67, %54
  %69 = load ptr, ptr %11, align 8
  %70 = getelementptr inbounds ptr, ptr %69, i64 1
  %71 = load ptr, ptr %70, align 8
  %72 = icmp ne ptr %71, null
  br i1 %72, label %73, label %82

73:                                               ; preds = %68
  %74 = load ptr, ptr %11, align 8
  %75 = getelementptr inbounds ptr, ptr %74, i64 1
  %76 = load ptr, ptr %75, align 8
  store ptr %76, ptr %16, align 8
  %77 = load i64, ptr %12, align 8
  %78 = add nsw i64 %77, -1
  store i64 %78, ptr %12, align 8
  %79 = icmp ne i64 %78, 0
  br i1 %79, label %81, label %80

80:                                               ; preds = %73
  br label %105

81:                                               ; preds = %73
  br label %82

82:                                               ; preds = %81, %68
  %83 = load ptr, ptr %11, align 8
  %84 = getelementptr inbounds ptr, ptr %83, i64 2
  %85 = load ptr, ptr %84, align 8
  %86 = icmp ne ptr %85, null
  br i1 %86, label %87, label %96

87:                                               ; preds = %82
  %88 = load ptr, ptr %11, align 8
  %89 = getelementptr inbounds ptr, ptr %88, i64 2
  %90 = load ptr, ptr %89, align 8
  store ptr %90, ptr %17, align 8
  %91 = load i64, ptr %12, align 8
  %92 = add nsw i64 %91, -1
  store i64 %92, ptr %12, align 8
  %93 = icmp ne i64 %92, 0
  br i1 %93, label %95, label %94

94:                                               ; preds = %87
  br label %105

95:                                               ; preds = %87
  br label %96

96:                                               ; preds = %95, %82
  %97 = load ptr, ptr %11, align 8
  %98 = getelementptr inbounds ptr, ptr %97, i64 3
  %99 = load ptr, ptr %98, align 8
  %100 = call i32 @PyObject_IsTrue(ptr noundef %99)
  store i32 %100, ptr %18, align 4
  %101 = load i32, ptr %18, align 4
  %102 = icmp slt i32 %101, 0
  br i1 %102, label %103, label %104

103:                                              ; preds = %96
  br label %116

104:                                              ; preds = %96
  br label %105

105:                                              ; preds = %104, %94, %80, %66, %53
  %106 = load ptr, ptr %6, align 8
  store ptr %106, ptr %13, align 8
  %107 = load i64, ptr %7, align 8
  store i64 %107, ptr %14, align 8
  %108 = load ptr, ptr %5, align 8
  %109 = load ptr, ptr %13, align 8
  %110 = load i64, ptr %14, align 8
  %111 = load ptr, ptr %15, align 8
  %112 = load ptr, ptr %16, align 8
  %113 = load ptr, ptr %17, align 8
  %114 = load i32, ptr %18, align 4
  %115 = call ptr @builtin_print_impl(ptr noundef %108, ptr noundef %109, i64 noundef %110, ptr noundef %111, ptr noundef %112, ptr noundef %113, i32 noundef %114)
  store ptr %115, ptr %9, align 8
  br label %116

116:                                              ; preds = %105, %103, %49
  %117 = load ptr, ptr %9, align 8
  ret ptr %117
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_repr(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = call ptr @PyObject_Repr(ptr noundef %5)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_round(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [2 x ptr], align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store ptr null, ptr %9, align 8
  %14 = load i64, ptr %7, align 8
  %15 = load ptr, ptr %8, align 8
  %16 = icmp ne ptr %15, null
  br i1 %16, label %17, label %20

17:                                               ; preds = %4
  %18 = load ptr, ptr %8, align 8
  %19 = call i64 @PyTuple_GET_SIZE(ptr noundef %18)
  br label %21

20:                                               ; preds = %4
  br label %21

21:                                               ; preds = %20, %17
  %22 = phi i64 [ %19, %17 ], [ 0, %20 ]
  %23 = add nsw i64 %14, %22
  %24 = sub nsw i64 %23, 1
  store i64 %24, ptr %11, align 8
  store ptr @_Py_NoneStruct, ptr %13, align 8
  %25 = load ptr, ptr %8, align 8
  %26 = icmp eq ptr %25, null
  br i1 %26, label %27, label %38

27:                                               ; preds = %21
  %28 = load i64, ptr %7, align 8
  %29 = icmp sle i64 1, %28
  br i1 %29, label %30, label %38

30:                                               ; preds = %27
  %31 = load i64, ptr %7, align 8
  %32 = icmp sle i64 %31, 2
  br i1 %32, label %33, label %38

33:                                               ; preds = %30
  %34 = load ptr, ptr %6, align 8
  %35 = icmp ne ptr %34, null
  br i1 %35, label %36, label %38

36:                                               ; preds = %33
  %37 = load ptr, ptr %6, align 8
  br label %44

38:                                               ; preds = %33, %30, %27, %21
  %39 = load ptr, ptr %6, align 8
  %40 = load i64, ptr %7, align 8
  %41 = load ptr, ptr %8, align 8
  %42 = getelementptr inbounds [2 x ptr], ptr %10, i64 0, i64 0
  %43 = call ptr @_PyArg_UnpackKeywords(ptr noundef %39, i64 noundef %40, ptr noundef null, ptr noundef %41, ptr noundef @builtin_round._parser, i32 noundef 1, i32 noundef 2, i32 noundef 0, i32 noundef 0, ptr noundef %42)
  br label %44

44:                                               ; preds = %38, %36
  %45 = phi ptr [ %37, %36 ], [ %43, %38 ]
  store ptr %45, ptr %6, align 8
  %46 = load ptr, ptr %6, align 8
  %47 = icmp ne ptr %46, null
  br i1 %47, label %49, label %48

48:                                               ; preds = %44
  br label %65

49:                                               ; preds = %44
  %50 = load ptr, ptr %6, align 8
  %51 = getelementptr inbounds ptr, ptr %50, i64 0
  %52 = load ptr, ptr %51, align 8
  store ptr %52, ptr %12, align 8
  %53 = load i64, ptr %11, align 8
  %54 = icmp ne i64 %53, 0
  br i1 %54, label %56, label %55

55:                                               ; preds = %49
  br label %60

56:                                               ; preds = %49
  %57 = load ptr, ptr %6, align 8
  %58 = getelementptr inbounds ptr, ptr %57, i64 1
  %59 = load ptr, ptr %58, align 8
  store ptr %59, ptr %13, align 8
  br label %60

60:                                               ; preds = %56, %55
  %61 = load ptr, ptr %5, align 8
  %62 = load ptr, ptr %12, align 8
  %63 = load ptr, ptr %13, align 8
  %64 = call ptr @builtin_round_impl(ptr noundef %61, ptr noundef %62, ptr noundef %63)
  store ptr %64, ptr %9, align 8
  br label %65

65:                                               ; preds = %60, %48
  %66 = load ptr, ptr %9, align 8
  ret ptr %66
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_setattr(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  store ptr null, ptr %7, align 8
  %11 = load i64, ptr %6, align 8
  %12 = icmp sle i64 3, %11
  br i1 %12, label %13, label %16

13:                                               ; preds = %3
  %14 = load i64, ptr %6, align 8
  %15 = icmp sle i64 %14, 3
  br i1 %15, label %21, label %16

16:                                               ; preds = %13, %3
  %17 = load i64, ptr %6, align 8
  %18 = call i32 @_PyArg_CheckPositional(ptr noundef @.str.115, i64 noundef %17, i64 noundef 3, i64 noundef 3)
  %19 = icmp ne i32 %18, 0
  br i1 %19, label %21, label %20

20:                                               ; preds = %16
  br label %36

21:                                               ; preds = %16, %13
  %22 = load ptr, ptr %5, align 8
  %23 = getelementptr inbounds ptr, ptr %22, i64 0
  %24 = load ptr, ptr %23, align 8
  store ptr %24, ptr %8, align 8
  %25 = load ptr, ptr %5, align 8
  %26 = getelementptr inbounds ptr, ptr %25, i64 1
  %27 = load ptr, ptr %26, align 8
  store ptr %27, ptr %9, align 8
  %28 = load ptr, ptr %5, align 8
  %29 = getelementptr inbounds ptr, ptr %28, i64 2
  %30 = load ptr, ptr %29, align 8
  store ptr %30, ptr %10, align 8
  %31 = load ptr, ptr %4, align 8
  %32 = load ptr, ptr %8, align 8
  %33 = load ptr, ptr %9, align 8
  %34 = load ptr, ptr %10, align 8
  %35 = call ptr @builtin_setattr_impl(ptr noundef %31, ptr noundef %32, ptr noundef %33, ptr noundef %34)
  store ptr %35, ptr %7, align 8
  br label %36

36:                                               ; preds = %21, %20
  %37 = load ptr, ptr %7, align 8
  ret ptr %37
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_sorted(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca i64, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  store ptr %0, ptr %14, align 8
  store ptr %1, ptr %15, align 8
  store i64 %2, ptr %16, align 8
  store ptr %3, ptr %17, align 8
  %22 = load ptr, ptr %15, align 8
  %23 = load i64, ptr %16, align 8
  %24 = call i32 (ptr, i64, ptr, i64, i64, ...) @_PyArg_UnpackStack(ptr noundef %22, i64 noundef %23, ptr noundef @.str.116, i64 noundef 1, i64 noundef 1, ptr noundef %20)
  %25 = icmp ne i32 %24, 0
  br i1 %25, label %27, label %26

26:                                               ; preds = %4
  store ptr null, ptr %13, align 8
  br label %126

27:                                               ; preds = %4
  %28 = load ptr, ptr %20, align 8
  %29 = call ptr @PySequence_List(ptr noundef %28)
  store ptr %29, ptr %18, align 8
  %30 = load ptr, ptr %18, align 8
  %31 = icmp eq ptr %30, null
  br i1 %31, label %32, label %33

32:                                               ; preds = %27
  store ptr null, ptr %13, align 8
  br label %126

33:                                               ; preds = %27
  %34 = load ptr, ptr %18, align 8
  %35 = call ptr @PyObject_GetAttr(ptr noundef %34, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 646))
  store ptr %35, ptr %21, align 8
  %36 = load ptr, ptr %21, align 8
  %37 = icmp eq ptr %36, null
  br i1 %37, label %38, label %55

38:                                               ; preds = %33
  %39 = load ptr, ptr %18, align 8
  store ptr %39, ptr %9, align 8
  %40 = load ptr, ptr %9, align 8
  store ptr %40, ptr %8, align 8
  %41 = load ptr, ptr %8, align 8
  %42 = load i32, ptr %41, align 8
  %43 = icmp slt i32 %42, 0
  %44 = zext i1 %43 to i32
  %45 = icmp ne i32 %44, 0
  br i1 %45, label %46, label %47

46:                                               ; preds = %38
  br label %54

47:                                               ; preds = %38
  %48 = load ptr, ptr %9, align 8
  %49 = load i32, ptr %48, align 8
  %50 = add i32 %49, -1
  store i32 %50, ptr %48, align 8
  %51 = icmp eq i32 %50, 0
  br i1 %51, label %52, label %54

52:                                               ; preds = %47
  %53 = load ptr, ptr %9, align 8
  call void @_Py_Dealloc(ptr noundef %53) #7
  br label %54

54:                                               ; preds = %46, %47, %52
  store ptr null, ptr %13, align 8
  br label %126

55:                                               ; preds = %33
  %56 = load i64, ptr %16, align 8
  %57 = icmp sge i64 %56, 1
  %58 = xor i1 %57, true
  %59 = zext i1 %58 to i32
  %60 = sext i32 %59 to i64
  %61 = icmp ne i64 %60, 0
  br i1 %61, label %62, label %64

62:                                               ; preds = %55
  call void @__assert_rtn(ptr noundef @__func__.builtin_sorted, ptr noundef @.str.34, i32 noundef 2571, ptr noundef @.str.223) #8
  unreachable

63:                                               ; No predecessors!
  br label %65

64:                                               ; preds = %55
  br label %65

65:                                               ; preds = %64, %63
  %66 = load ptr, ptr %21, align 8
  %67 = load ptr, ptr %15, align 8
  %68 = getelementptr inbounds ptr, ptr %67, i64 1
  %69 = load i64, ptr %16, align 8
  %70 = sub nsw i64 %69, 1
  %71 = load ptr, ptr %17, align 8
  %72 = call ptr @PyObject_Vectorcall(ptr noundef %66, ptr noundef %68, i64 noundef %70, ptr noundef %71)
  store ptr %72, ptr %19, align 8
  %73 = load ptr, ptr %21, align 8
  store ptr %73, ptr %10, align 8
  %74 = load ptr, ptr %10, align 8
  store ptr %74, ptr %7, align 8
  %75 = load ptr, ptr %7, align 8
  %76 = load i32, ptr %75, align 8
  %77 = icmp slt i32 %76, 0
  %78 = zext i1 %77 to i32
  %79 = icmp ne i32 %78, 0
  br i1 %79, label %80, label %81

80:                                               ; preds = %65
  br label %88

81:                                               ; preds = %65
  %82 = load ptr, ptr %10, align 8
  %83 = load i32, ptr %82, align 8
  %84 = add i32 %83, -1
  store i32 %84, ptr %82, align 8
  %85 = icmp eq i32 %84, 0
  br i1 %85, label %86, label %88

86:                                               ; preds = %81
  %87 = load ptr, ptr %10, align 8
  call void @_Py_Dealloc(ptr noundef %87) #7
  br label %88

88:                                               ; preds = %80, %81, %86
  %89 = load ptr, ptr %19, align 8
  %90 = icmp eq ptr %89, null
  br i1 %90, label %91, label %108

91:                                               ; preds = %88
  %92 = load ptr, ptr %18, align 8
  store ptr %92, ptr %11, align 8
  %93 = load ptr, ptr %11, align 8
  store ptr %93, ptr %6, align 8
  %94 = load ptr, ptr %6, align 8
  %95 = load i32, ptr %94, align 8
  %96 = icmp slt i32 %95, 0
  %97 = zext i1 %96 to i32
  %98 = icmp ne i32 %97, 0
  br i1 %98, label %99, label %100

99:                                               ; preds = %91
  br label %107

100:                                              ; preds = %91
  %101 = load ptr, ptr %11, align 8
  %102 = load i32, ptr %101, align 8
  %103 = add i32 %102, -1
  store i32 %103, ptr %101, align 8
  %104 = icmp eq i32 %103, 0
  br i1 %104, label %105, label %107

105:                                              ; preds = %100
  %106 = load ptr, ptr %11, align 8
  call void @_Py_Dealloc(ptr noundef %106) #7
  br label %107

107:                                              ; preds = %99, %100, %105
  store ptr null, ptr %13, align 8
  br label %126

108:                                              ; preds = %88
  %109 = load ptr, ptr %19, align 8
  store ptr %109, ptr %12, align 8
  %110 = load ptr, ptr %12, align 8
  store ptr %110, ptr %5, align 8
  %111 = load ptr, ptr %5, align 8
  %112 = load i32, ptr %111, align 8
  %113 = icmp slt i32 %112, 0
  %114 = zext i1 %113 to i32
  %115 = icmp ne i32 %114, 0
  br i1 %115, label %116, label %117

116:                                              ; preds = %108
  br label %124

117:                                              ; preds = %108
  %118 = load ptr, ptr %12, align 8
  %119 = load i32, ptr %118, align 8
  %120 = add i32 %119, -1
  store i32 %120, ptr %118, align 8
  %121 = icmp eq i32 %120, 0
  br i1 %121, label %122, label %124

122:                                              ; preds = %117
  %123 = load ptr, ptr %12, align 8
  call void @_Py_Dealloc(ptr noundef %123) #7
  br label %124

124:                                              ; preds = %116, %117, %122
  %125 = load ptr, ptr %18, align 8
  store ptr %125, ptr %13, align 8
  br label %126

126:                                              ; preds = %124, %107, %54, %32, %26
  %127 = load ptr, ptr %13, align 8
  ret ptr %127
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_sum(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [2 x ptr], align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store ptr null, ptr %9, align 8
  %14 = load i64, ptr %7, align 8
  %15 = load ptr, ptr %8, align 8
  %16 = icmp ne ptr %15, null
  br i1 %16, label %17, label %20

17:                                               ; preds = %4
  %18 = load ptr, ptr %8, align 8
  %19 = call i64 @PyTuple_GET_SIZE(ptr noundef %18)
  br label %21

20:                                               ; preds = %4
  br label %21

21:                                               ; preds = %20, %17
  %22 = phi i64 [ %19, %17 ], [ 0, %20 ]
  %23 = add nsw i64 %14, %22
  %24 = sub nsw i64 %23, 1
  store i64 %24, ptr %11, align 8
  store ptr null, ptr %13, align 8
  %25 = load ptr, ptr %8, align 8
  %26 = icmp eq ptr %25, null
  br i1 %26, label %27, label %38

27:                                               ; preds = %21
  %28 = load i64, ptr %7, align 8
  %29 = icmp sle i64 1, %28
  br i1 %29, label %30, label %38

30:                                               ; preds = %27
  %31 = load i64, ptr %7, align 8
  %32 = icmp sle i64 %31, 2
  br i1 %32, label %33, label %38

33:                                               ; preds = %30
  %34 = load ptr, ptr %6, align 8
  %35 = icmp ne ptr %34, null
  br i1 %35, label %36, label %38

36:                                               ; preds = %33
  %37 = load ptr, ptr %6, align 8
  br label %44

38:                                               ; preds = %33, %30, %27, %21
  %39 = load ptr, ptr %6, align 8
  %40 = load i64, ptr %7, align 8
  %41 = load ptr, ptr %8, align 8
  %42 = getelementptr inbounds [2 x ptr], ptr %10, i64 0, i64 0
  %43 = call ptr @_PyArg_UnpackKeywords(ptr noundef %39, i64 noundef %40, ptr noundef null, ptr noundef %41, ptr noundef @builtin_sum._parser, i32 noundef 1, i32 noundef 2, i32 noundef 0, i32 noundef 0, ptr noundef %42)
  br label %44

44:                                               ; preds = %38, %36
  %45 = phi ptr [ %37, %36 ], [ %43, %38 ]
  store ptr %45, ptr %6, align 8
  %46 = load ptr, ptr %6, align 8
  %47 = icmp ne ptr %46, null
  br i1 %47, label %49, label %48

48:                                               ; preds = %44
  br label %65

49:                                               ; preds = %44
  %50 = load ptr, ptr %6, align 8
  %51 = getelementptr inbounds ptr, ptr %50, i64 0
  %52 = load ptr, ptr %51, align 8
  store ptr %52, ptr %12, align 8
  %53 = load i64, ptr %11, align 8
  %54 = icmp ne i64 %53, 0
  br i1 %54, label %56, label %55

55:                                               ; preds = %49
  br label %60

56:                                               ; preds = %49
  %57 = load ptr, ptr %6, align 8
  %58 = getelementptr inbounds ptr, ptr %57, i64 1
  %59 = load ptr, ptr %58, align 8
  store ptr %59, ptr %13, align 8
  br label %60

60:                                               ; preds = %56, %55
  %61 = load ptr, ptr %5, align 8
  %62 = load ptr, ptr %12, align 8
  %63 = load ptr, ptr %13, align 8
  %64 = call ptr @builtin_sum_impl(ptr noundef %61, ptr noundef %62, ptr noundef %63)
  store ptr %64, ptr %9, align 8
  br label %65

65:                                               ; preds = %60, %48
  %66 = load ptr, ptr %9, align 8
  ret ptr %66
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_vars(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr null, ptr %6, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr noundef %8, ptr noundef @.str.118, i64 noundef 0, i64 noundef 1, ptr noundef %6)
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %2
  store ptr null, ptr %3, align 8
  br label %26

12:                                               ; preds = %2
  %13 = load ptr, ptr %6, align 8
  %14 = icmp eq ptr %13, null
  br i1 %14, label %15, label %17

15:                                               ; preds = %12
  %16 = call ptr @_PyEval_GetFrameLocals()
  store ptr %16, ptr %7, align 8
  br label %24

17:                                               ; preds = %12
  %18 = load ptr, ptr %6, align 8
  %19 = call i32 @PyObject_GetOptionalAttr(ptr noundef %18, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 45), ptr noundef %7)
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %21, label %23

21:                                               ; preds = %17
  %22 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %22, ptr noundef @.str.235)
  br label %23

23:                                               ; preds = %21, %17
  br label %24

24:                                               ; preds = %23, %15
  %25 = load ptr, ptr %7, align 8
  store ptr %25, ptr %3, align 8
  br label %26

26:                                               ; preds = %24, %11
  %27 = load ptr, ptr %3, align 8
  ret ptr %27
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @Py_IS_TYPE(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = call ptr @_Py_TYPE(ptr noundef %5)
  %7 = load ptr, ptr %4, align 8
  %8 = icmp eq ptr %6, %7
  %9 = zext i1 %8 to i32
  ret i32 %9
}

declare ptr @_PyTuple_FromArray(ptr noundef, i64 noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @update_bases(ptr noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca i64, align 8
  %20 = alloca i64, align 8
  %21 = alloca i64, align 8
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca ptr, align 8
  %25 = alloca ptr, align 8
  %26 = alloca ptr, align 8
  store ptr %0, ptr %17, align 8
  store ptr %1, ptr %18, align 8
  store i64 %2, ptr %19, align 8
  store ptr null, ptr %26, align 8
  %27 = load ptr, ptr %17, align 8
  %28 = call ptr @_Py_TYPE(ptr noundef %27)
  %29 = call i32 @PyType_HasFeature(ptr noundef %28, i64 noundef 67108864)
  %30 = icmp ne i32 %29, 0
  %31 = xor i1 %30, true
  %32 = zext i1 %31 to i32
  %33 = sext i32 %32 to i64
  %34 = icmp ne i64 %33, 0
  br i1 %34, label %35, label %37

35:                                               ; preds = %3
  call void @__assert_rtn(ptr noundef @__func__.update_bases, ptr noundef @.str.34, i32 noundef 30, ptr noundef @.str.122) #8
  unreachable

36:                                               ; No predecessors!
  br label %38

37:                                               ; preds = %3
  br label %38

38:                                               ; preds = %37, %36
  store i64 0, ptr %20, align 8
  br label %39

39:                                               ; preds = %212, %38
  %40 = load i64, ptr %20, align 8
  %41 = load i64, ptr %19, align 8
  %42 = icmp slt i64 %40, %41
  br i1 %42, label %43, label %215

43:                                               ; preds = %39
  %44 = load ptr, ptr %18, align 8
  %45 = load i64, ptr %20, align 8
  %46 = getelementptr inbounds ptr, ptr %44, i64 %45
  %47 = load ptr, ptr %46, align 8
  store ptr %47, ptr %22, align 8
  %48 = load ptr, ptr %22, align 8
  %49 = call i32 @PyType_Check(ptr noundef %48)
  %50 = icmp ne i32 %49, 0
  br i1 %50, label %51, label %62

51:                                               ; preds = %43
  %52 = load ptr, ptr %26, align 8
  %53 = icmp ne ptr %52, null
  br i1 %53, label %54, label %61

54:                                               ; preds = %51
  %55 = load ptr, ptr %26, align 8
  %56 = load ptr, ptr %22, align 8
  %57 = call i32 @PyList_Append(ptr noundef %55, ptr noundef %56)
  %58 = icmp slt i32 %57, 0
  br i1 %58, label %59, label %60

59:                                               ; preds = %54
  br label %240

60:                                               ; preds = %54
  br label %61

61:                                               ; preds = %60, %51
  br label %212

62:                                               ; preds = %43
  %63 = load ptr, ptr %22, align 8
  %64 = call i32 @PyObject_GetOptionalAttr(ptr noundef %63, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 105), ptr noundef %23)
  %65 = icmp slt i32 %64, 0
  br i1 %65, label %66, label %67

66:                                               ; preds = %62
  br label %240

67:                                               ; preds = %62
  %68 = load ptr, ptr %23, align 8
  %69 = icmp ne ptr %68, null
  br i1 %69, label %81, label %70

70:                                               ; preds = %67
  %71 = load ptr, ptr %26, align 8
  %72 = icmp ne ptr %71, null
  br i1 %72, label %73, label %80

73:                                               ; preds = %70
  %74 = load ptr, ptr %26, align 8
  %75 = load ptr, ptr %22, align 8
  %76 = call i32 @PyList_Append(ptr noundef %74, ptr noundef %75)
  %77 = icmp slt i32 %76, 0
  br i1 %77, label %78, label %79

78:                                               ; preds = %73
  br label %240

79:                                               ; preds = %73
  br label %80

80:                                               ; preds = %79, %70
  br label %212

81:                                               ; preds = %67
  %82 = load ptr, ptr %23, align 8
  %83 = load ptr, ptr %17, align 8
  %84 = call ptr @PyObject_CallOneArg(ptr noundef %82, ptr noundef %83)
  store ptr %84, ptr %24, align 8
  %85 = load ptr, ptr %23, align 8
  store ptr %85, ptr %10, align 8
  %86 = load ptr, ptr %10, align 8
  store ptr %86, ptr %9, align 8
  %87 = load ptr, ptr %9, align 8
  %88 = load i32, ptr %87, align 8
  %89 = icmp slt i32 %88, 0
  %90 = zext i1 %89 to i32
  %91 = icmp ne i32 %90, 0
  br i1 %91, label %92, label %93

92:                                               ; preds = %81
  br label %100

93:                                               ; preds = %81
  %94 = load ptr, ptr %10, align 8
  %95 = load i32, ptr %94, align 8
  %96 = add i32 %95, -1
  store i32 %96, ptr %94, align 8
  %97 = icmp eq i32 %96, 0
  br i1 %97, label %98, label %100

98:                                               ; preds = %93
  %99 = load ptr, ptr %10, align 8
  call void @_Py_Dealloc(ptr noundef %99) #7
  br label %100

100:                                              ; preds = %92, %93, %98
  %101 = load ptr, ptr %24, align 8
  %102 = icmp ne ptr %101, null
  br i1 %102, label %104, label %103

103:                                              ; preds = %100
  br label %240

104:                                              ; preds = %100
  %105 = load ptr, ptr %24, align 8
  %106 = call ptr @_Py_TYPE(ptr noundef %105)
  %107 = call i32 @PyType_HasFeature(ptr noundef %106, i64 noundef 67108864)
  %108 = icmp ne i32 %107, 0
  br i1 %108, label %127, label %109

109:                                              ; preds = %104
  %110 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %110, ptr noundef @.str.128)
  %111 = load ptr, ptr %24, align 8
  store ptr %111, ptr %11, align 8
  %112 = load ptr, ptr %11, align 8
  store ptr %112, ptr %8, align 8
  %113 = load ptr, ptr %8, align 8
  %114 = load i32, ptr %113, align 8
  %115 = icmp slt i32 %114, 0
  %116 = zext i1 %115 to i32
  %117 = icmp ne i32 %116, 0
  br i1 %117, label %118, label %119

118:                                              ; preds = %109
  br label %126

119:                                              ; preds = %109
  %120 = load ptr, ptr %11, align 8
  %121 = load i32, ptr %120, align 8
  %122 = add i32 %121, -1
  store i32 %122, ptr %120, align 8
  %123 = icmp eq i32 %122, 0
  br i1 %123, label %124, label %126

124:                                              ; preds = %119
  %125 = load ptr, ptr %11, align 8
  call void @_Py_Dealloc(ptr noundef %125) #7
  br label %126

126:                                              ; preds = %118, %119, %124
  br label %240

127:                                              ; preds = %104
  %128 = load ptr, ptr %26, align 8
  %129 = icmp ne ptr %128, null
  br i1 %129, label %169, label %130

130:                                              ; preds = %127
  %131 = load i64, ptr %20, align 8
  %132 = call ptr @PyList_New(i64 noundef %131)
  store ptr %132, ptr %26, align 8
  %133 = icmp ne ptr %132, null
  br i1 %133, label %151, label %134

134:                                              ; preds = %130
  %135 = load ptr, ptr %24, align 8
  store ptr %135, ptr %12, align 8
  %136 = load ptr, ptr %12, align 8
  store ptr %136, ptr %7, align 8
  %137 = load ptr, ptr %7, align 8
  %138 = load i32, ptr %137, align 8
  %139 = icmp slt i32 %138, 0
  %140 = zext i1 %139 to i32
  %141 = icmp ne i32 %140, 0
  br i1 %141, label %142, label %143

142:                                              ; preds = %134
  br label %150

143:                                              ; preds = %134
  %144 = load ptr, ptr %12, align 8
  %145 = load i32, ptr %144, align 8
  %146 = add i32 %145, -1
  store i32 %146, ptr %144, align 8
  %147 = icmp eq i32 %146, 0
  br i1 %147, label %148, label %150

148:                                              ; preds = %143
  %149 = load ptr, ptr %12, align 8
  call void @_Py_Dealloc(ptr noundef %149) #7
  br label %150

150:                                              ; preds = %142, %143, %148
  br label %240

151:                                              ; preds = %130
  store i64 0, ptr %21, align 8
  br label %152

152:                                              ; preds = %165, %151
  %153 = load i64, ptr %21, align 8
  %154 = load i64, ptr %20, align 8
  %155 = icmp slt i64 %153, %154
  br i1 %155, label %156, label %168

156:                                              ; preds = %152
  %157 = load ptr, ptr %18, align 8
  %158 = load i64, ptr %21, align 8
  %159 = getelementptr inbounds ptr, ptr %157, i64 %158
  %160 = load ptr, ptr %159, align 8
  store ptr %160, ptr %22, align 8
  %161 = load ptr, ptr %26, align 8
  %162 = load i64, ptr %21, align 8
  %163 = load ptr, ptr %22, align 8
  %164 = call ptr @_Py_NewRef(ptr noundef %163)
  call void @PyList_SET_ITEM(ptr noundef %161, i64 noundef %162, ptr noundef %164)
  br label %165

165:                                              ; preds = %156
  %166 = load i64, ptr %21, align 8
  %167 = add nsw i64 %166, 1
  store i64 %167, ptr %21, align 8
  br label %152, !llvm.loop !18

168:                                              ; preds = %152
  br label %169

169:                                              ; preds = %168, %127
  %170 = load ptr, ptr %26, align 8
  %171 = call i64 @PyList_GET_SIZE(ptr noundef %170)
  store i64 %171, ptr %21, align 8
  %172 = load ptr, ptr %26, align 8
  %173 = load i64, ptr %21, align 8
  %174 = load i64, ptr %21, align 8
  %175 = load ptr, ptr %24, align 8
  %176 = call i32 @PyList_SetSlice(ptr noundef %172, i64 noundef %173, i64 noundef %174, ptr noundef %175)
  %177 = icmp slt i32 %176, 0
  br i1 %177, label %178, label %195

178:                                              ; preds = %169
  %179 = load ptr, ptr %24, align 8
  store ptr %179, ptr %13, align 8
  %180 = load ptr, ptr %13, align 8
  store ptr %180, ptr %6, align 8
  %181 = load ptr, ptr %6, align 8
  %182 = load i32, ptr %181, align 8
  %183 = icmp slt i32 %182, 0
  %184 = zext i1 %183 to i32
  %185 = icmp ne i32 %184, 0
  br i1 %185, label %186, label %187

186:                                              ; preds = %178
  br label %194

187:                                              ; preds = %178
  %188 = load ptr, ptr %13, align 8
  %189 = load i32, ptr %188, align 8
  %190 = add i32 %189, -1
  store i32 %190, ptr %188, align 8
  %191 = icmp eq i32 %190, 0
  br i1 %191, label %192, label %194

192:                                              ; preds = %187
  %193 = load ptr, ptr %13, align 8
  call void @_Py_Dealloc(ptr noundef %193) #7
  br label %194

194:                                              ; preds = %186, %187, %192
  br label %240

195:                                              ; preds = %169
  %196 = load ptr, ptr %24, align 8
  store ptr %196, ptr %14, align 8
  %197 = load ptr, ptr %14, align 8
  store ptr %197, ptr %5, align 8
  %198 = load ptr, ptr %5, align 8
  %199 = load i32, ptr %198, align 8
  %200 = icmp slt i32 %199, 0
  %201 = zext i1 %200 to i32
  %202 = icmp ne i32 %201, 0
  br i1 %202, label %203, label %204

203:                                              ; preds = %195
  br label %211

204:                                              ; preds = %195
  %205 = load ptr, ptr %14, align 8
  %206 = load i32, ptr %205, align 8
  %207 = add i32 %206, -1
  store i32 %207, ptr %205, align 8
  %208 = icmp eq i32 %207, 0
  br i1 %208, label %209, label %211

209:                                              ; preds = %204
  %210 = load ptr, ptr %14, align 8
  call void @_Py_Dealloc(ptr noundef %210) #7
  br label %211

211:                                              ; preds = %203, %204, %209
  br label %212

212:                                              ; preds = %211, %80, %61
  %213 = load i64, ptr %20, align 8
  %214 = add nsw i64 %213, 1
  store i64 %214, ptr %20, align 8
  br label %39, !llvm.loop !19

215:                                              ; preds = %39
  %216 = load ptr, ptr %26, align 8
  %217 = icmp ne ptr %216, null
  br i1 %217, label %220, label %218

218:                                              ; preds = %215
  %219 = load ptr, ptr %17, align 8
  store ptr %219, ptr %16, align 8
  br label %242

220:                                              ; preds = %215
  %221 = load ptr, ptr %26, align 8
  %222 = call ptr @PyList_AsTuple(ptr noundef %221)
  store ptr %222, ptr %25, align 8
  %223 = load ptr, ptr %26, align 8
  store ptr %223, ptr %15, align 8
  %224 = load ptr, ptr %15, align 8
  store ptr %224, ptr %4, align 8
  %225 = load ptr, ptr %4, align 8
  %226 = load i32, ptr %225, align 8
  %227 = icmp slt i32 %226, 0
  %228 = zext i1 %227 to i32
  %229 = icmp ne i32 %228, 0
  br i1 %229, label %230, label %231

230:                                              ; preds = %220
  br label %238

231:                                              ; preds = %220
  %232 = load ptr, ptr %15, align 8
  %233 = load i32, ptr %232, align 8
  %234 = add i32 %233, -1
  store i32 %234, ptr %232, align 8
  %235 = icmp eq i32 %234, 0
  br i1 %235, label %236, label %238

236:                                              ; preds = %231
  %237 = load ptr, ptr %15, align 8
  call void @_Py_Dealloc(ptr noundef %237) #7
  br label %238

238:                                              ; preds = %230, %231, %236
  %239 = load ptr, ptr %25, align 8
  store ptr %239, ptr %16, align 8
  br label %242

240:                                              ; preds = %194, %150, %126, %103, %78, %66, %59
  %241 = load ptr, ptr %26, align 8
  call void @Py_XDECREF(ptr noundef %241)
  store ptr null, ptr %16, align 8
  br label %242

242:                                              ; preds = %240, %238, %218
  %243 = load ptr, ptr %16, align 8
  ret ptr %243
}

declare ptr @_PyStack_AsDict(ptr noundef, ptr noundef) #1

declare i32 @PyDict_Pop(ptr noundef, ptr noundef, ptr noundef) #1

declare ptr @_PyType_CalculateMetaclass(ptr noundef, ptr noundef) #1

declare i32 @PyObject_GetOptionalAttr(ptr noundef, ptr noundef, ptr noundef) #1

declare ptr @PyDict_New() #1

declare ptr @PyObject_VectorcallDict(ptr noundef, ptr noundef, i64 noundef, ptr noundef) #1

declare i32 @PyMapping_Check(ptr noundef) #1

declare ptr @_PyEval_Vector(ptr noundef, ptr noundef, ptr noundef, ptr noundef, i64 noundef, ptr noundef) #1

declare i32 @PyMapping_SetItemString(ptr noundef, ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @PyCell_GetRef(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds %struct.PyCellObject, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = call ptr @_Py_XNewRef(ptr noundef %6)
  store ptr %7, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  ret ptr %8
}

declare i32 @PyList_Append(ptr noundef, ptr noundef) #1

declare ptr @PyList_New(i64 noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @PyList_SET_ITEM(ptr noundef %0, i64 noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store i64 %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = call ptr @_Py_TYPE(ptr noundef %8)
  %10 = call i32 @PyType_HasFeature(ptr noundef %9, i64 noundef 33554432)
  %11 = icmp ne i32 %10, 0
  %12 = xor i1 %11, true
  %13 = zext i1 %12 to i32
  %14 = sext i32 %13 to i64
  %15 = icmp ne i64 %14, 0
  br i1 %15, label %16, label %18

16:                                               ; preds = %3
  call void @__assert_rtn(ptr noundef @__func__.PyList_SET_ITEM, ptr noundef @.str.129, i32 noundef 44, ptr noundef @.str.130) #8
  unreachable

17:                                               ; No predecessors!
  br label %19

18:                                               ; preds = %3
  br label %19

19:                                               ; preds = %18, %17
  %20 = load ptr, ptr %4, align 8
  store ptr %20, ptr %7, align 8
  %21 = load i64, ptr %5, align 8
  %22 = icmp sle i64 0, %21
  %23 = xor i1 %22, true
  %24 = zext i1 %23 to i32
  %25 = sext i32 %24 to i64
  %26 = icmp ne i64 %25, 0
  br i1 %26, label %27, label %29

27:                                               ; preds = %19
  call void @__assert_rtn(ptr noundef @__func__.PyList_SET_ITEM, ptr noundef @.str.129, i32 noundef 45, ptr noundef @.str.55) #8
  unreachable

28:                                               ; No predecessors!
  br label %30

29:                                               ; preds = %19
  br label %30

30:                                               ; preds = %29, %28
  %31 = load i64, ptr %5, align 8
  %32 = load ptr, ptr %7, align 8
  %33 = getelementptr inbounds %struct.PyListObject, ptr %32, i32 0, i32 2
  %34 = load i64, ptr %33, align 8
  %35 = icmp slt i64 %31, %34
  %36 = xor i1 %35, true
  %37 = zext i1 %36 to i32
  %38 = sext i32 %37 to i64
  %39 = icmp ne i64 %38, 0
  br i1 %39, label %40, label %42

40:                                               ; preds = %30
  call void @__assert_rtn(ptr noundef @__func__.PyList_SET_ITEM, ptr noundef @.str.129, i32 noundef 46, ptr noundef @.str.131) #8
  unreachable

41:                                               ; No predecessors!
  br label %43

42:                                               ; preds = %30
  br label %43

43:                                               ; preds = %42, %41
  %44 = load ptr, ptr %6, align 8
  %45 = load ptr, ptr %7, align 8
  %46 = getelementptr inbounds %struct.PyListObject, ptr %45, i32 0, i32 1
  %47 = load ptr, ptr %46, align 8
  %48 = load i64, ptr %5, align 8
  %49 = getelementptr inbounds ptr, ptr %47, i64 %48
  store ptr %44, ptr %49, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @PyList_GET_SIZE(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call ptr @_Py_TYPE(ptr noundef %4)
  %6 = call i32 @PyType_HasFeature(ptr noundef %5, i64 noundef 33554432)
  %7 = icmp ne i32 %6, 0
  %8 = xor i1 %7, true
  %9 = zext i1 %8 to i32
  %10 = sext i32 %9 to i64
  %11 = icmp ne i64 %10, 0
  br i1 %11, label %12, label %14

12:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyList_GET_SIZE, ptr noundef @.str.129, i32 noundef 31, ptr noundef @.str.130) #8
  unreachable

13:                                               ; No predecessors!
  br label %15

14:                                               ; preds = %1
  br label %15

15:                                               ; preds = %14, %13
  %16 = load ptr, ptr %2, align 8
  store ptr %16, ptr %3, align 8
  %17 = load ptr, ptr %3, align 8
  %18 = call i64 @Py_SIZE(ptr noundef %17)
  ret i64 %18
}

declare i32 @PyList_SetSlice(ptr noundef, i64 noundef, i64 noundef, ptr noundef) #1

declare ptr @PyList_AsTuple(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_Py_XNewRef(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @Py_XINCREF(ptr noundef %3)
  %4 = load ptr, ptr %2, align 8
  ret ptr %4
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @Py_XINCREF(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = icmp ne ptr %5, null
  br i1 %6, label %7, label %19

7:                                                ; preds = %1
  %8 = load ptr, ptr %4, align 8
  store ptr %8, ptr %2, align 8
  %9 = load ptr, ptr %2, align 8
  %10 = load i32, ptr %9, align 8
  store i32 %10, ptr %3, align 4
  %11 = load i32, ptr %3, align 4
  %12 = icmp slt i32 %11, 0
  br i1 %12, label %13, label %14

13:                                               ; preds = %7
  br label %18

14:                                               ; preds = %7
  %15 = load i32, ptr %3, align 4
  %16 = add i32 %15, 1
  %17 = load ptr, ptr %2, align 8
  store i32 %16, ptr %17, align 8
  br label %18

18:                                               ; preds = %13, %14
  br label %19

19:                                               ; preds = %18, %1
  ret void
}

declare ptr @_PyArg_UnpackKeywords(ptr noundef, i64 noundef, ptr noundef, ptr noundef, ptr noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, ptr noundef) #1

declare i32 @PyLong_AsInt(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin___import___impl(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4, i32 noundef %5) #0 {
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i32, align 4
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  store ptr %3, ptr %10, align 8
  store ptr %4, ptr %11, align 8
  store i32 %5, ptr %12, align 4
  %13 = load ptr, ptr %8, align 8
  %14 = load ptr, ptr %9, align 8
  %15 = load ptr, ptr %10, align 8
  %16 = load ptr, ptr %11, align 8
  %17 = load i32, ptr %12, align 4
  %18 = call ptr @PyImport_ImportModuleLevelObject(ptr noundef %13, ptr noundef %14, ptr noundef %15, ptr noundef %16, i32 noundef %17)
  ret ptr %18
}

declare ptr @PyImport_ImportModuleLevelObject(ptr noundef, ptr noundef, ptr noundef, ptr noundef, i32 noundef) #1

declare ptr @PyNumber_Absolute(ptr noundef) #1

declare ptr @PyObject_ASCII(ptr noundef) #1

declare ptr @PyNumber_ToBase(ptr noundef, i32 noundef) #1

declare ptr @PySys_GetObject(ptr noundef) #1

declare i32 @PySys_Audit(ptr noundef, ptr noundef, ...) #1

declare ptr @PyObject_Vectorcall(ptr noundef, ptr noundef, i64 noundef, ptr noundef) #1

declare i64 @PyLong_AsLongAndOverflow(ptr noundef, ptr noundef) #1

declare ptr @PyUnicode_FromOrdinal(i32 noundef) #1

declare i32 @PyUnicode_FSDecoder(ptr noundef, ptr noundef) #1

declare void @_PyArg_BadArgument(ptr noundef, ptr noundef, ptr noundef, ptr noundef) #1

declare ptr @PyUnicode_AsUTF8AndSize(ptr noundef, ptr noundef) #1

; Function Attrs: nounwind
declare i64 @strlen(ptr noundef) #6

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_compile_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef %7) #0 {
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca i32, align 4
  %22 = alloca i32, align 4
  %23 = alloca [4 x i32], align 4
  %24 = alloca ptr, align 8
  %25 = alloca %struct.PyCompilerFlags, align 4
  %26 = alloca ptr, align 8
  %27 = alloca ptr, align 8
  %28 = alloca ptr, align 8
  %29 = alloca ptr, align 8
  store ptr %0, ptr %11, align 8
  store ptr %1, ptr %12, align 8
  store ptr %2, ptr %13, align 8
  store ptr %3, ptr %14, align 8
  store i32 %4, ptr %15, align 4
  store i32 %5, ptr %16, align 4
  store i32 %6, ptr %17, align 4
  store i32 %7, ptr %18, align 4
  store i32 -1, ptr %21, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %23, ptr align 4 @__const.builtin_compile_impl.start, i64 16, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %25, ptr align 4 @__const.builtin_compile_impl.cf, i64 8, i1 false)
  %30 = load i32, ptr %15, align 4
  %31 = or i32 %30, 256
  %32 = getelementptr inbounds %struct.PyCompilerFlags, ptr %25, i32 0, i32 0
  store i32 %31, ptr %32, align 4
  %33 = load i32, ptr %18, align 4
  %34 = icmp sge i32 %33, 0
  br i1 %34, label %35, label %42

35:                                               ; preds = %8
  %36 = load i32, ptr %15, align 4
  %37 = and i32 %36, 1024
  %38 = icmp ne i32 %37, 0
  br i1 %38, label %39, label %42

39:                                               ; preds = %35
  %40 = load i32, ptr %18, align 4
  %41 = getelementptr inbounds %struct.PyCompilerFlags, ptr %25, i32 0, i32 1
  store i32 %40, ptr %41, align 4
  br label %42

42:                                               ; preds = %39, %35, %8
  %43 = load i32, ptr %15, align 4
  %44 = and i32 %43, -33486353
  %45 = icmp ne i32 %44, 0
  br i1 %45, label %46, label %48

46:                                               ; preds = %42
  %47 = load ptr, ptr @PyExc_ValueError, align 8
  call void @PyErr_SetString(ptr noundef %47, ptr noundef @.str.148)
  br label %187

48:                                               ; preds = %42
  %49 = load i32, ptr %17, align 4
  %50 = icmp slt i32 %49, -1
  br i1 %50, label %54, label %51

51:                                               ; preds = %48
  %52 = load i32, ptr %17, align 4
  %53 = icmp sgt i32 %52, 2
  br i1 %53, label %54, label %56

54:                                               ; preds = %51, %48
  %55 = load ptr, ptr @PyExc_ValueError, align 8
  call void @PyErr_SetString(ptr noundef %55, ptr noundef @.str.149)
  br label %187

56:                                               ; preds = %51
  %57 = load i32, ptr %16, align 4
  %58 = icmp ne i32 %57, 0
  br i1 %58, label %61, label %59

59:                                               ; preds = %56
  %60 = call i32 @PyEval_MergeCompilerFlags(ptr noundef %25)
  br label %61

61:                                               ; preds = %59, %56
  %62 = load ptr, ptr %14, align 8
  %63 = call i32 @strcmp(ptr noundef %62, ptr noundef @.str.90) #7
  %64 = icmp eq i32 %63, 0
  br i1 %64, label %65, label %66

65:                                               ; preds = %61
  store i32 0, ptr %21, align 4
  br label %99

66:                                               ; preds = %61
  %67 = load ptr, ptr %14, align 8
  %68 = call i32 @strcmp(ptr noundef %67, ptr noundef @.str.89) #7
  %69 = icmp eq i32 %68, 0
  br i1 %69, label %70, label %71

70:                                               ; preds = %66
  store i32 1, ptr %21, align 4
  br label %98

71:                                               ; preds = %66
  %72 = load ptr, ptr %14, align 8
  %73 = call i32 @strcmp(ptr noundef %72, ptr noundef @.str.150) #7
  %74 = icmp eq i32 %73, 0
  br i1 %74, label %75, label %76

75:                                               ; preds = %71
  store i32 2, ptr %21, align 4
  br label %97

76:                                               ; preds = %71
  %77 = load ptr, ptr %14, align 8
  %78 = call i32 @strcmp(ptr noundef %77, ptr noundef @.str.151) #7
  %79 = icmp eq i32 %78, 0
  br i1 %79, label %80, label %87

80:                                               ; preds = %76
  %81 = load i32, ptr %15, align 4
  %82 = and i32 %81, 1024
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %86, label %84

84:                                               ; preds = %80
  %85 = load ptr, ptr @PyExc_ValueError, align 8
  call void @PyErr_SetString(ptr noundef %85, ptr noundef @.str.152)
  br label %187

86:                                               ; preds = %80
  store i32 3, ptr %21, align 4
  br label %96

87:                                               ; preds = %76
  %88 = load i32, ptr %15, align 4
  %89 = and i32 %88, 1024
  %90 = icmp ne i32 %89, 0
  br i1 %90, label %91, label %92

91:                                               ; preds = %87
  store ptr @.str.153, ptr %26, align 8
  br label %93

92:                                               ; preds = %87
  store ptr @.str.154, ptr %26, align 8
  br label %93

93:                                               ; preds = %92, %91
  %94 = load ptr, ptr @PyExc_ValueError, align 8
  %95 = load ptr, ptr %26, align 8
  call void @PyErr_SetString(ptr noundef %94, ptr noundef %95)
  br label %187

96:                                               ; preds = %86
  br label %97

97:                                               ; preds = %96, %75
  br label %98

98:                                               ; preds = %97, %70
  br label %99

99:                                               ; preds = %98, %65
  %100 = load ptr, ptr %12, align 8
  %101 = call i32 @PyAST_Check(ptr noundef %100)
  store i32 %101, ptr %22, align 4
  %102 = load i32, ptr %22, align 4
  %103 = icmp eq i32 %102, -1
  br i1 %103, label %104, label %105

104:                                              ; preds = %99
  br label %187

105:                                              ; preds = %99
  %106 = load i32, ptr %22, align 4
  %107 = icmp ne i32 %106, 0
  br i1 %107, label %108, label %171

108:                                              ; preds = %105
  %109 = load i32, ptr %15, align 4
  %110 = and i32 %109, 33792
  %111 = icmp eq i32 %110, 1024
  br i1 %111, label %112, label %115

112:                                              ; preds = %108
  %113 = load ptr, ptr %12, align 8
  %114 = call ptr @_Py_NewRef(ptr noundef %113)
  store ptr %114, ptr %24, align 8
  br label %170

115:                                              ; preds = %108
  %116 = call ptr @_PyArena_New()
  store ptr %116, ptr %27, align 8
  %117 = load ptr, ptr %27, align 8
  %118 = icmp eq ptr %117, null
  br i1 %118, label %119, label %120

119:                                              ; preds = %115
  br label %187

120:                                              ; preds = %115
  %121 = load i32, ptr %15, align 4
  %122 = and i32 %121, 1024
  %123 = icmp ne i32 %122, 0
  br i1 %123, label %124, label %149

124:                                              ; preds = %120
  %125 = load ptr, ptr %12, align 8
  %126 = load ptr, ptr %27, align 8
  %127 = load i32, ptr %21, align 4
  %128 = call ptr @PyAST_obj2mod(ptr noundef %125, ptr noundef %126, i32 noundef %127)
  store ptr %128, ptr %28, align 8
  %129 = load ptr, ptr %28, align 8
  %130 = icmp eq ptr %129, null
  br i1 %130, label %135, label %131

131:                                              ; preds = %124
  %132 = load ptr, ptr %28, align 8
  %133 = call i32 @_PyAST_Validate(ptr noundef %132)
  %134 = icmp ne i32 %133, 0
  br i1 %134, label %137, label %135

135:                                              ; preds = %131, %124
  %136 = load ptr, ptr %27, align 8
  call void @_PyArena_Free(ptr noundef %136)
  br label %187

137:                                              ; preds = %131
  %138 = load ptr, ptr %28, align 8
  %139 = load ptr, ptr %13, align 8
  %140 = load i32, ptr %17, align 4
  %141 = load ptr, ptr %27, align 8
  %142 = call i32 @_PyCompile_AstOptimize(ptr noundef %138, ptr noundef %139, ptr noundef %25, i32 noundef %140, ptr noundef %141)
  %143 = icmp slt i32 %142, 0
  br i1 %143, label %144, label %146

144:                                              ; preds = %137
  %145 = load ptr, ptr %27, align 8
  call void @_PyArena_Free(ptr noundef %145)
  br label %187

146:                                              ; preds = %137
  %147 = load ptr, ptr %28, align 8
  %148 = call ptr @PyAST_mod2obj(ptr noundef %147)
  store ptr %148, ptr %24, align 8
  br label %168

149:                                              ; preds = %120
  %150 = load ptr, ptr %12, align 8
  %151 = load ptr, ptr %27, align 8
  %152 = load i32, ptr %21, align 4
  %153 = call ptr @PyAST_obj2mod(ptr noundef %150, ptr noundef %151, i32 noundef %152)
  store ptr %153, ptr %29, align 8
  %154 = load ptr, ptr %29, align 8
  %155 = icmp eq ptr %154, null
  br i1 %155, label %160, label %156

156:                                              ; preds = %149
  %157 = load ptr, ptr %29, align 8
  %158 = call i32 @_PyAST_Validate(ptr noundef %157)
  %159 = icmp ne i32 %158, 0
  br i1 %159, label %162, label %160

160:                                              ; preds = %156, %149
  %161 = load ptr, ptr %27, align 8
  call void @_PyArena_Free(ptr noundef %161)
  br label %187

162:                                              ; preds = %156
  %163 = load ptr, ptr %29, align 8
  %164 = load ptr, ptr %13, align 8
  %165 = load i32, ptr %17, align 4
  %166 = load ptr, ptr %27, align 8
  %167 = call ptr @_PyAST_Compile(ptr noundef %163, ptr noundef %164, ptr noundef %25, i32 noundef %165, ptr noundef %166)
  store ptr %167, ptr %24, align 8
  br label %168

168:                                              ; preds = %162, %146
  %169 = load ptr, ptr %27, align 8
  call void @_PyArena_Free(ptr noundef %169)
  br label %170

170:                                              ; preds = %168, %112
  br label %188

171:                                              ; preds = %105
  %172 = load ptr, ptr %12, align 8
  %173 = call ptr @_Py_SourceAsString(ptr noundef %172, ptr noundef @.str.85, ptr noundef @.str.155, ptr noundef %25, ptr noundef %19)
  store ptr %173, ptr %20, align 8
  %174 = load ptr, ptr %20, align 8
  %175 = icmp eq ptr %174, null
  br i1 %175, label %176, label %177

176:                                              ; preds = %171
  br label %187

177:                                              ; preds = %171
  %178 = load ptr, ptr %20, align 8
  %179 = load ptr, ptr %13, align 8
  %180 = load i32, ptr %21, align 4
  %181 = sext i32 %180 to i64
  %182 = getelementptr inbounds [4 x i32], ptr %23, i64 0, i64 %181
  %183 = load i32, ptr %182, align 4
  %184 = load i32, ptr %17, align 4
  %185 = call ptr @Py_CompileStringObject(ptr noundef %178, ptr noundef %179, i32 noundef %183, ptr noundef %25, i32 noundef %184)
  store ptr %185, ptr %24, align 8
  %186 = load ptr, ptr %19, align 8
  call void @Py_XDECREF(ptr noundef %186)
  br label %188

187:                                              ; preds = %176, %160, %144, %135, %119, %104, %93, %84, %54, %46
  store ptr null, ptr %24, align 8
  br label %188

188:                                              ; preds = %187, %177, %170
  %189 = load ptr, ptr %13, align 8
  store ptr %189, ptr %10, align 8
  %190 = load ptr, ptr %10, align 8
  store ptr %190, ptr %9, align 8
  %191 = load ptr, ptr %9, align 8
  %192 = load i32, ptr %191, align 8
  %193 = icmp slt i32 %192, 0
  %194 = zext i1 %193 to i32
  %195 = icmp ne i32 %194, 0
  br i1 %195, label %196, label %197

196:                                              ; preds = %188
  br label %204

197:                                              ; preds = %188
  %198 = load ptr, ptr %10, align 8
  %199 = load i32, ptr %198, align 8
  %200 = add i32 %199, -1
  store i32 %200, ptr %198, align 8
  %201 = icmp eq i32 %200, 0
  br i1 %201, label %202, label %204

202:                                              ; preds = %197
  %203 = load ptr, ptr %10, align 8
  call void @_Py_Dealloc(ptr noundef %203) #7
  br label %204

204:                                              ; preds = %196, %197, %202
  %205 = load ptr, ptr %24, align 8
  ret ptr %205
}

declare i32 @PyEval_MergeCompilerFlags(ptr noundef) #1

; Function Attrs: nounwind
declare i32 @strcmp(ptr noundef, ptr noundef) #6

declare i32 @PyAST_Check(ptr noundef) #1

declare ptr @_PyArena_New() #1

declare ptr @PyAST_obj2mod(ptr noundef, ptr noundef, i32 noundef) #1

declare i32 @_PyAST_Validate(ptr noundef) #1

declare void @_PyArena_Free(ptr noundef) #1

declare i32 @_PyCompile_AstOptimize(ptr noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef) #1

declare ptr @PyAST_mod2obj(ptr noundef) #1

declare ptr @_PyAST_Compile(ptr noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef) #1

declare ptr @_Py_SourceAsString(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) #1

declare ptr @Py_CompileStringObject(ptr noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_delattr_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = load ptr, ptr %7, align 8
  %10 = call i32 @PyObject_DelAttr(ptr noundef %8, ptr noundef %9)
  %11 = icmp slt i32 %10, 0
  br i1 %11, label %12, label %13

12:                                               ; preds = %3
  store ptr null, ptr %4, align 8
  br label %14

13:                                               ; preds = %3
  store ptr @_Py_NoneStruct, ptr %4, align 8
  br label %14

14:                                               ; preds = %13, %12
  %15 = load ptr, ptr %4, align 8
  ret ptr %15
}

declare i32 @PyObject_DelAttr(ptr noundef, ptr noundef) #1

declare ptr @PyObject_Dir(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_divmod_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = call ptr @PyNumber_Divmod(ptr noundef %7, ptr noundef %8)
  ret ptr %9
}

declare ptr @PyNumber_Divmod(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_eval_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca i32, align 4
  %18 = alloca %struct.PyCompilerFlags, align 4
  store ptr %0, ptr %10, align 8
  store ptr %1, ptr %11, align 8
  store ptr %2, ptr %12, align 8
  store ptr %3, ptr %13, align 8
  store ptr null, ptr %14, align 8
  %19 = load ptr, ptr %13, align 8
  %20 = icmp ne ptr %19, @_Py_NoneStruct
  br i1 %20, label %21, label %27

21:                                               ; preds = %4
  %22 = load ptr, ptr %13, align 8
  %23 = call i32 @PyMapping_Check(ptr noundef %22)
  %24 = icmp ne i32 %23, 0
  br i1 %24, label %27, label %25

25:                                               ; preds = %21
  %26 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %26, ptr noundef @.str.157)
  store ptr null, ptr %9, align 8
  br label %162

27:                                               ; preds = %21, %4
  %28 = load ptr, ptr %12, align 8
  %29 = icmp ne ptr %28, @_Py_NoneStruct
  br i1 %29, label %30, label %42

30:                                               ; preds = %27
  %31 = load ptr, ptr %12, align 8
  %32 = call ptr @_Py_TYPE(ptr noundef %31)
  %33 = call i32 @PyType_HasFeature(ptr noundef %32, i64 noundef 536870912)
  %34 = icmp ne i32 %33, 0
  br i1 %34, label %42, label %35

35:                                               ; preds = %30
  %36 = load ptr, ptr @PyExc_TypeError, align 8
  %37 = load ptr, ptr %12, align 8
  %38 = call i32 @PyMapping_Check(ptr noundef %37)
  %39 = icmp ne i32 %38, 0
  %40 = zext i1 %39 to i64
  %41 = select i1 %39, ptr @.str.158, ptr @.str.159
  call void @PyErr_SetString(ptr noundef %36, ptr noundef %41)
  store ptr null, ptr %9, align 8
  br label %162

42:                                               ; preds = %30, %27
  %43 = load ptr, ptr %12, align 8
  %44 = icmp eq ptr %43, @_Py_NoneStruct
  br i1 %44, label %45, label %68

45:                                               ; preds = %42
  %46 = call ptr @PyEval_GetGlobals()
  store ptr %46, ptr %12, align 8
  %47 = load ptr, ptr %13, align 8
  %48 = icmp eq ptr %47, @_Py_NoneStruct
  br i1 %48, label %49, label %55

49:                                               ; preds = %45
  %50 = call ptr @_PyEval_GetFrameLocals()
  store ptr %50, ptr %13, align 8
  %51 = load ptr, ptr %13, align 8
  %52 = icmp eq ptr %51, null
  br i1 %52, label %53, label %54

53:                                               ; preds = %49
  store ptr null, ptr %9, align 8
  br label %162

54:                                               ; preds = %49
  br label %67

55:                                               ; preds = %45
  %56 = load ptr, ptr %13, align 8
  store ptr %56, ptr %5, align 8
  %57 = load ptr, ptr %5, align 8
  %58 = load i32, ptr %57, align 8
  store i32 %58, ptr %6, align 4
  %59 = load i32, ptr %6, align 4
  %60 = icmp slt i32 %59, 0
  br i1 %60, label %61, label %62

61:                                               ; preds = %55
  br label %66

62:                                               ; preds = %55
  %63 = load i32, ptr %6, align 4
  %64 = add i32 %63, 1
  %65 = load ptr, ptr %5, align 8
  store i32 %64, ptr %65, align 8
  br label %66

66:                                               ; preds = %61, %62
  br label %67

67:                                               ; preds = %66, %54
  br label %87

68:                                               ; preds = %42
  %69 = load ptr, ptr %13, align 8
  %70 = icmp eq ptr %69, @_Py_NoneStruct
  br i1 %70, label %71, label %74

71:                                               ; preds = %68
  %72 = load ptr, ptr %12, align 8
  %73 = call ptr @_Py_NewRef(ptr noundef %72)
  store ptr %73, ptr %13, align 8
  br label %86

74:                                               ; preds = %68
  %75 = load ptr, ptr %13, align 8
  store ptr %75, ptr %7, align 8
  %76 = load ptr, ptr %7, align 8
  %77 = load i32, ptr %76, align 8
  store i32 %77, ptr %8, align 4
  %78 = load i32, ptr %8, align 4
  %79 = icmp slt i32 %78, 0
  br i1 %79, label %80, label %81

80:                                               ; preds = %74
  br label %85

81:                                               ; preds = %74
  %82 = load i32, ptr %8, align 4
  %83 = add i32 %82, 1
  %84 = load ptr, ptr %7, align 8
  store i32 %83, ptr %84, align 8
  br label %85

85:                                               ; preds = %80, %81
  br label %86

86:                                               ; preds = %85, %71
  br label %87

87:                                               ; preds = %86, %67
  %88 = load ptr, ptr %12, align 8
  %89 = icmp eq ptr %88, null
  br i1 %89, label %93, label %90

90:                                               ; preds = %87
  %91 = load ptr, ptr %13, align 8
  %92 = icmp eq ptr %91, null
  br i1 %92, label %93, label %95

93:                                               ; preds = %90, %87
  %94 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %94, ptr noundef @.str.160)
  br label %159

95:                                               ; preds = %90
  %96 = load ptr, ptr %12, align 8
  %97 = call i32 @PyDict_Contains(ptr noundef %96, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 29))
  store i32 %97, ptr %17, align 4
  %98 = load i32, ptr %17, align 4
  %99 = icmp eq i32 %98, 0
  br i1 %99, label %100, label %104

100:                                              ; preds = %95
  %101 = load ptr, ptr %12, align 8
  %102 = call ptr @PyEval_GetBuiltins()
  %103 = call i32 @PyDict_SetItem(ptr noundef %101, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 29), ptr noundef %102)
  store i32 %103, ptr %17, align 4
  br label %104

104:                                              ; preds = %100, %95
  %105 = load i32, ptr %17, align 4
  %106 = icmp slt i32 %105, 0
  br i1 %106, label %107, label %108

107:                                              ; preds = %104
  br label %159

108:                                              ; preds = %104
  %109 = load ptr, ptr %11, align 8
  %110 = call i32 @Py_IS_TYPE(ptr noundef %109, ptr noundef @PyCode_Type)
  %111 = icmp ne i32 %110, 0
  br i1 %111, label %112, label %128

112:                                              ; preds = %108
  %113 = load ptr, ptr %11, align 8
  %114 = call i32 (ptr, ptr, ...) @PySys_Audit(ptr noundef @.str.90, ptr noundef @.str.138, ptr noundef %113)
  %115 = icmp slt i32 %114, 0
  br i1 %115, label %116, label %117

116:                                              ; preds = %112
  br label %159

117:                                              ; preds = %112
  %118 = load ptr, ptr %11, align 8
  %119 = call i64 @PyCode_GetNumFree(ptr noundef %118)
  %120 = icmp sgt i64 %119, 0
  br i1 %120, label %121, label %123

121:                                              ; preds = %117
  %122 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %122, ptr noundef @.str.161)
  br label %159

123:                                              ; preds = %117
  %124 = load ptr, ptr %11, align 8
  %125 = load ptr, ptr %12, align 8
  %126 = load ptr, ptr %13, align 8
  %127 = call ptr @PyEval_EvalCode(ptr noundef %124, ptr noundef %125, ptr noundef %126)
  store ptr %127, ptr %14, align 8
  br label %158

128:                                              ; preds = %108
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %18, ptr align 4 @__const.builtin_eval_impl.cf, i64 8, i1 false)
  %129 = getelementptr inbounds %struct.PyCompilerFlags, ptr %18, i32 0, i32 0
  store i32 256, ptr %129, align 4
  %130 = load ptr, ptr %11, align 8
  %131 = call ptr @_Py_SourceAsString(ptr noundef %130, ptr noundef @.str.89, ptr noundef @.str.162, ptr noundef %18, ptr noundef %15)
  store ptr %131, ptr %16, align 8
  %132 = load ptr, ptr %16, align 8
  %133 = icmp eq ptr %132, null
  br i1 %133, label %134, label %135

134:                                              ; preds = %128
  br label %159

135:                                              ; preds = %128
  br label %136

136:                                              ; preds = %148, %135
  %137 = load ptr, ptr %16, align 8
  %138 = load i8, ptr %137, align 1
  %139 = sext i8 %138 to i32
  %140 = icmp eq i32 %139, 32
  br i1 %140, label %146, label %141

141:                                              ; preds = %136
  %142 = load ptr, ptr %16, align 8
  %143 = load i8, ptr %142, align 1
  %144 = sext i8 %143 to i32
  %145 = icmp eq i32 %144, 9
  br label %146

146:                                              ; preds = %141, %136
  %147 = phi i1 [ true, %136 ], [ %145, %141 ]
  br i1 %147, label %148, label %151

148:                                              ; preds = %146
  %149 = load ptr, ptr %16, align 8
  %150 = getelementptr inbounds i8, ptr %149, i32 1
  store ptr %150, ptr %16, align 8
  br label %136, !llvm.loop !20

151:                                              ; preds = %146
  %152 = call i32 @PyEval_MergeCompilerFlags(ptr noundef %18)
  %153 = load ptr, ptr %16, align 8
  %154 = load ptr, ptr %12, align 8
  %155 = load ptr, ptr %13, align 8
  %156 = call ptr @PyRun_StringFlags(ptr noundef %153, i32 noundef 258, ptr noundef %154, ptr noundef %155, ptr noundef %18)
  store ptr %156, ptr %14, align 8
  %157 = load ptr, ptr %15, align 8
  call void @Py_XDECREF(ptr noundef %157)
  br label %158

158:                                              ; preds = %151, %123
  br label %159

159:                                              ; preds = %158, %134, %121, %116, %107, %93
  %160 = load ptr, ptr %13, align 8
  call void @Py_XDECREF(ptr noundef %160)
  %161 = load ptr, ptr %14, align 8
  store ptr %161, ptr %9, align 8
  br label %162

162:                                              ; preds = %159, %53, %35, %25
  %163 = load ptr, ptr %9, align 8
  ret ptr %163
}

declare ptr @PyEval_GetGlobals() #1

declare ptr @_PyEval_GetFrameLocals() #1

declare i32 @PyDict_Contains(ptr noundef, ptr noundef) #1

declare i32 @PyDict_SetItem(ptr noundef, ptr noundef, ptr noundef) #1

declare ptr @PyEval_GetBuiltins() #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @PyCode_GetNumFree(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call i32 @Py_IS_TYPE(ptr noundef %3, ptr noundef @PyCode_Type)
  %5 = icmp ne i32 %4, 0
  %6 = xor i1 %5, true
  %7 = zext i1 %6 to i32
  %8 = sext i32 %7 to i64
  %9 = icmp ne i64 %8, 0
  br i1 %9, label %10, label %12

10:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyCode_GetNumFree, ptr noundef @.str.163, i32 noundef 217, ptr noundef @.str.164) #8
  unreachable

11:                                               ; No predecessors!
  br label %13

12:                                               ; preds = %1
  br label %13

13:                                               ; preds = %12, %11
  %14 = load ptr, ptr %2, align 8
  %15 = getelementptr inbounds %struct.PyCodeObject, ptr %14, i32 0, i32 14
  %16 = load i32, ptr %15, align 8
  %17 = sext i32 %16 to i64
  ret i64 %17
}

declare ptr @PyEval_EvalCode(ptr noundef, ptr noundef, ptr noundef) #1

declare ptr @PyRun_StringFlags(ptr noundef, i32 noundef, ptr noundef, ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_exec_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4) #0 {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca ptr, align 8
  %11 = alloca i32, align 4
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca i32, align 4
  %22 = alloca i64, align 8
  %23 = alloca i32, align 4
  %24 = alloca i64, align 8
  %25 = alloca ptr, align 8
  %26 = alloca ptr, align 8
  %27 = alloca ptr, align 8
  %28 = alloca %struct.PyCompilerFlags, align 4
  store ptr %0, ptr %15, align 8
  store ptr %1, ptr %16, align 8
  store ptr %2, ptr %17, align 8
  store ptr %3, ptr %18, align 8
  store ptr %4, ptr %19, align 8
  %29 = load ptr, ptr %17, align 8
  %30 = icmp eq ptr %29, @_Py_NoneStruct
  br i1 %30, label %31, label %62

31:                                               ; preds = %5
  %32 = call ptr @PyEval_GetGlobals()
  store ptr %32, ptr %17, align 8
  %33 = load ptr, ptr %18, align 8
  %34 = icmp eq ptr %33, @_Py_NoneStruct
  br i1 %34, label %35, label %41

35:                                               ; preds = %31
  %36 = call ptr @_PyEval_GetFrameLocals()
  store ptr %36, ptr %18, align 8
  %37 = load ptr, ptr %18, align 8
  %38 = icmp eq ptr %37, null
  br i1 %38, label %39, label %40

39:                                               ; preds = %35
  store ptr null, ptr %14, align 8
  br label %279

40:                                               ; preds = %35
  br label %53

41:                                               ; preds = %31
  %42 = load ptr, ptr %18, align 8
  store ptr %42, ptr %8, align 8
  %43 = load ptr, ptr %8, align 8
  %44 = load i32, ptr %43, align 8
  store i32 %44, ptr %9, align 4
  %45 = load i32, ptr %9, align 4
  %46 = icmp slt i32 %45, 0
  br i1 %46, label %47, label %48

47:                                               ; preds = %41
  br label %52

48:                                               ; preds = %41
  %49 = load i32, ptr %9, align 4
  %50 = add i32 %49, 1
  %51 = load ptr, ptr %8, align 8
  store i32 %50, ptr %51, align 8
  br label %52

52:                                               ; preds = %47, %48
  br label %53

53:                                               ; preds = %52, %40
  %54 = load ptr, ptr %17, align 8
  %55 = icmp ne ptr %54, null
  br i1 %55, label %56, label %59

56:                                               ; preds = %53
  %57 = load ptr, ptr %18, align 8
  %58 = icmp ne ptr %57, null
  br i1 %58, label %61, label %59

59:                                               ; preds = %56, %53
  %60 = load ptr, ptr @PyExc_SystemError, align 8
  call void @PyErr_SetString(ptr noundef %60, ptr noundef @.str.166)
  store ptr null, ptr %14, align 8
  br label %279

61:                                               ; preds = %56
  br label %81

62:                                               ; preds = %5
  %63 = load ptr, ptr %18, align 8
  %64 = icmp eq ptr %63, @_Py_NoneStruct
  br i1 %64, label %65, label %68

65:                                               ; preds = %62
  %66 = load ptr, ptr %17, align 8
  %67 = call ptr @_Py_NewRef(ptr noundef %66)
  store ptr %67, ptr %18, align 8
  br label %80

68:                                               ; preds = %62
  %69 = load ptr, ptr %18, align 8
  store ptr %69, ptr %10, align 8
  %70 = load ptr, ptr %10, align 8
  %71 = load i32, ptr %70, align 8
  store i32 %71, ptr %11, align 4
  %72 = load i32, ptr %11, align 4
  %73 = icmp slt i32 %72, 0
  br i1 %73, label %74, label %75

74:                                               ; preds = %68
  br label %79

75:                                               ; preds = %68
  %76 = load i32, ptr %11, align 4
  %77 = add i32 %76, 1
  %78 = load ptr, ptr %10, align 8
  store i32 %77, ptr %78, align 8
  br label %79

79:                                               ; preds = %74, %75
  br label %80

80:                                               ; preds = %79, %65
  br label %81

81:                                               ; preds = %80, %61
  %82 = load ptr, ptr %17, align 8
  %83 = call ptr @_Py_TYPE(ptr noundef %82)
  %84 = call i32 @PyType_HasFeature(ptr noundef %83, i64 noundef 536870912)
  %85 = icmp ne i32 %84, 0
  br i1 %85, label %93, label %86

86:                                               ; preds = %81
  %87 = load ptr, ptr @PyExc_TypeError, align 8
  %88 = load ptr, ptr %17, align 8
  %89 = call ptr @_Py_TYPE(ptr noundef %88)
  %90 = getelementptr inbounds %struct._typeobject, ptr %89, i32 0, i32 1
  %91 = load ptr, ptr %90, align 8
  %92 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %87, ptr noundef @.str.167, ptr noundef %91)
  br label %277

93:                                               ; preds = %81
  %94 = load ptr, ptr %18, align 8
  %95 = call i32 @PyMapping_Check(ptr noundef %94)
  %96 = icmp ne i32 %95, 0
  br i1 %96, label %104, label %97

97:                                               ; preds = %93
  %98 = load ptr, ptr @PyExc_TypeError, align 8
  %99 = load ptr, ptr %18, align 8
  %100 = call ptr @_Py_TYPE(ptr noundef %99)
  %101 = getelementptr inbounds %struct._typeobject, ptr %100, i32 0, i32 1
  %102 = load ptr, ptr %101, align 8
  %103 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %98, ptr noundef @.str.168, ptr noundef %102)
  br label %277

104:                                              ; preds = %93
  %105 = load ptr, ptr %17, align 8
  %106 = call i32 @PyDict_Contains(ptr noundef %105, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 29))
  store i32 %106, ptr %21, align 4
  %107 = load i32, ptr %21, align 4
  %108 = icmp eq i32 %107, 0
  br i1 %108, label %109, label %113

109:                                              ; preds = %104
  %110 = load ptr, ptr %17, align 8
  %111 = call ptr @PyEval_GetBuiltins()
  %112 = call i32 @PyDict_SetItem(ptr noundef %110, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 29), ptr noundef %111)
  store i32 %112, ptr %21, align 4
  br label %113

113:                                              ; preds = %109, %104
  %114 = load i32, ptr %21, align 4
  %115 = icmp slt i32 %114, 0
  br i1 %115, label %116, label %117

116:                                              ; preds = %113
  br label %277

117:                                              ; preds = %113
  %118 = load ptr, ptr %19, align 8
  %119 = icmp eq ptr %118, @_Py_NoneStruct
  br i1 %119, label %120, label %121

120:                                              ; preds = %117
  store ptr null, ptr %19, align 8
  br label %121

121:                                              ; preds = %120, %117
  %122 = load ptr, ptr %16, align 8
  %123 = call i32 @Py_IS_TYPE(ptr noundef %122, ptr noundef @PyCode_Type)
  %124 = icmp ne i32 %123, 0
  br i1 %124, label %125, label %213

125:                                              ; preds = %121
  %126 = load ptr, ptr %16, align 8
  %127 = call i64 @PyCode_GetNumFree(ptr noundef %126)
  store i64 %127, ptr %22, align 8
  %128 = load i64, ptr %22, align 8
  %129 = icmp eq i64 %128, 0
  br i1 %129, label %130, label %136

130:                                              ; preds = %125
  %131 = load ptr, ptr %19, align 8
  %132 = icmp ne ptr %131, null
  br i1 %132, label %133, label %135

133:                                              ; preds = %130
  %134 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %134, ptr noundef @.str.169)
  br label %277

135:                                              ; preds = %130
  br label %193

136:                                              ; preds = %125
  %137 = load ptr, ptr %19, align 8
  %138 = icmp ne ptr %137, null
  br i1 %138, label %139, label %148

139:                                              ; preds = %136
  %140 = load ptr, ptr %19, align 8
  %141 = call i32 @Py_IS_TYPE(ptr noundef %140, ptr noundef @PyTuple_Type)
  %142 = icmp ne i32 %141, 0
  br i1 %142, label %143, label %148

143:                                              ; preds = %139
  %144 = load ptr, ptr %19, align 8
  %145 = call i64 @PyTuple_GET_SIZE(ptr noundef %144)
  %146 = load i64, ptr %22, align 8
  %147 = icmp eq i64 %145, %146
  br label %148

148:                                              ; preds = %143, %139, %136
  %149 = phi i1 [ false, %139 ], [ false, %136 ], [ %147, %143 ]
  %150 = zext i1 %149 to i32
  store i32 %150, ptr %23, align 4
  %151 = load i32, ptr %23, align 4
  %152 = icmp ne i32 %151, 0
  br i1 %152, label %153, label %185

153:                                              ; preds = %148
  store i64 0, ptr %24, align 8
  br label %154

154:                                              ; preds = %181, %153
  %155 = load i64, ptr %24, align 8
  %156 = load i64, ptr %22, align 8
  %157 = icmp slt i64 %155, %156
  br i1 %157, label %158, label %184

158:                                              ; preds = %154
  %159 = load ptr, ptr %19, align 8
  %160 = call ptr @_Py_TYPE(ptr noundef %159)
  %161 = call i32 @PyType_HasFeature(ptr noundef %160, i64 noundef 67108864)
  %162 = icmp ne i32 %161, 0
  %163 = xor i1 %162, true
  %164 = zext i1 %163 to i32
  %165 = sext i32 %164 to i64
  %166 = icmp ne i64 %165, 0
  br i1 %166, label %167, label %169

167:                                              ; preds = %158
  call void @__assert_rtn(ptr noundef @__func__.builtin_exec_impl, ptr noundef @.str.34, i32 noundef 1138, ptr noundef @.str.170) #8
  unreachable

168:                                              ; No predecessors!
  br label %170

169:                                              ; preds = %158
  br label %170

170:                                              ; preds = %169, %168
  %171 = load ptr, ptr %19, align 8
  %172 = getelementptr inbounds %struct.PyTupleObject, ptr %171, i32 0, i32 1
  %173 = load i64, ptr %24, align 8
  %174 = getelementptr inbounds [1 x ptr], ptr %172, i64 0, i64 %173
  %175 = load ptr, ptr %174, align 8
  store ptr %175, ptr %25, align 8
  %176 = load ptr, ptr %25, align 8
  %177 = call i32 @Py_IS_TYPE(ptr noundef %176, ptr noundef @PyCell_Type)
  %178 = icmp ne i32 %177, 0
  br i1 %178, label %180, label %179

179:                                              ; preds = %170
  store i32 0, ptr %23, align 4
  br label %184

180:                                              ; preds = %170
  br label %181

181:                                              ; preds = %180
  %182 = load i64, ptr %24, align 8
  %183 = add nsw i64 %182, 1
  store i64 %183, ptr %24, align 8
  br label %154, !llvm.loop !21

184:                                              ; preds = %179, %154
  br label %185

185:                                              ; preds = %184, %148
  %186 = load i32, ptr %23, align 4
  %187 = icmp ne i32 %186, 0
  br i1 %187, label %192, label %188

188:                                              ; preds = %185
  %189 = load ptr, ptr @PyExc_TypeError, align 8
  %190 = load i64, ptr %22, align 8
  %191 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %189, ptr noundef @.str.171, i64 noundef %190)
  br label %277

192:                                              ; preds = %185
  br label %193

193:                                              ; preds = %192, %135
  %194 = load ptr, ptr %16, align 8
  %195 = call i32 (ptr, ptr, ...) @PySys_Audit(ptr noundef @.str.90, ptr noundef @.str.138, ptr noundef %194)
  %196 = icmp slt i32 %195, 0
  br i1 %196, label %197, label %198

197:                                              ; preds = %193
  br label %277

198:                                              ; preds = %193
  %199 = load ptr, ptr %19, align 8
  %200 = icmp ne ptr %199, null
  br i1 %200, label %206, label %201

201:                                              ; preds = %198
  %202 = load ptr, ptr %16, align 8
  %203 = load ptr, ptr %17, align 8
  %204 = load ptr, ptr %18, align 8
  %205 = call ptr @PyEval_EvalCode(ptr noundef %202, ptr noundef %203, ptr noundef %204)
  store ptr %205, ptr %20, align 8
  br label %212

206:                                              ; preds = %198
  %207 = load ptr, ptr %16, align 8
  %208 = load ptr, ptr %17, align 8
  %209 = load ptr, ptr %18, align 8
  %210 = load ptr, ptr %19, align 8
  %211 = call ptr @PyEval_EvalCodeEx(ptr noundef %207, ptr noundef %208, ptr noundef %209, ptr noundef null, i32 noundef 0, ptr noundef null, i32 noundef 0, ptr noundef null, i32 noundef 0, ptr noundef null, ptr noundef %210)
  store ptr %211, ptr %20, align 8
  br label %212

212:                                              ; preds = %206, %201
  br label %240

213:                                              ; preds = %121
  %214 = load ptr, ptr %19, align 8
  %215 = icmp ne ptr %214, null
  br i1 %215, label %216, label %218

216:                                              ; preds = %213
  %217 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %217, ptr noundef @.str.172)
  br label %218

218:                                              ; preds = %216, %213
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %28, ptr align 4 @__const.builtin_exec_impl.cf, i64 8, i1 false)
  %219 = getelementptr inbounds %struct.PyCompilerFlags, ptr %28, i32 0, i32 0
  store i32 256, ptr %219, align 4
  %220 = load ptr, ptr %16, align 8
  %221 = call ptr @_Py_SourceAsString(ptr noundef %220, ptr noundef @.str.90, ptr noundef @.str.162, ptr noundef %28, ptr noundef %26)
  store ptr %221, ptr %27, align 8
  %222 = load ptr, ptr %27, align 8
  %223 = icmp eq ptr %222, null
  br i1 %223, label %224, label %225

224:                                              ; preds = %218
  br label %277

225:                                              ; preds = %218
  %226 = call i32 @PyEval_MergeCompilerFlags(ptr noundef %28)
  %227 = icmp ne i32 %226, 0
  br i1 %227, label %228, label %233

228:                                              ; preds = %225
  %229 = load ptr, ptr %27, align 8
  %230 = load ptr, ptr %17, align 8
  %231 = load ptr, ptr %18, align 8
  %232 = call ptr @PyRun_StringFlags(ptr noundef %229, i32 noundef 257, ptr noundef %230, ptr noundef %231, ptr noundef %28)
  store ptr %232, ptr %20, align 8
  br label %238

233:                                              ; preds = %225
  %234 = load ptr, ptr %27, align 8
  %235 = load ptr, ptr %17, align 8
  %236 = load ptr, ptr %18, align 8
  %237 = call ptr @PyRun_StringFlags(ptr noundef %234, i32 noundef 257, ptr noundef %235, ptr noundef %236, ptr noundef null)
  store ptr %237, ptr %20, align 8
  br label %238

238:                                              ; preds = %233, %228
  %239 = load ptr, ptr %26, align 8
  call void @Py_XDECREF(ptr noundef %239)
  br label %240

240:                                              ; preds = %238, %212
  %241 = load ptr, ptr %20, align 8
  %242 = icmp eq ptr %241, null
  br i1 %242, label %243, label %244

243:                                              ; preds = %240
  br label %277

244:                                              ; preds = %240
  %245 = load ptr, ptr %18, align 8
  store ptr %245, ptr %12, align 8
  %246 = load ptr, ptr %12, align 8
  store ptr %246, ptr %7, align 8
  %247 = load ptr, ptr %7, align 8
  %248 = load i32, ptr %247, align 8
  %249 = icmp slt i32 %248, 0
  %250 = zext i1 %249 to i32
  %251 = icmp ne i32 %250, 0
  br i1 %251, label %252, label %253

252:                                              ; preds = %244
  br label %260

253:                                              ; preds = %244
  %254 = load ptr, ptr %12, align 8
  %255 = load i32, ptr %254, align 8
  %256 = add i32 %255, -1
  store i32 %256, ptr %254, align 8
  %257 = icmp eq i32 %256, 0
  br i1 %257, label %258, label %260

258:                                              ; preds = %253
  %259 = load ptr, ptr %12, align 8
  call void @_Py_Dealloc(ptr noundef %259) #7
  br label %260

260:                                              ; preds = %252, %253, %258
  %261 = load ptr, ptr %20, align 8
  store ptr %261, ptr %13, align 8
  %262 = load ptr, ptr %13, align 8
  store ptr %262, ptr %6, align 8
  %263 = load ptr, ptr %6, align 8
  %264 = load i32, ptr %263, align 8
  %265 = icmp slt i32 %264, 0
  %266 = zext i1 %265 to i32
  %267 = icmp ne i32 %266, 0
  br i1 %267, label %268, label %269

268:                                              ; preds = %260
  br label %276

269:                                              ; preds = %260
  %270 = load ptr, ptr %13, align 8
  %271 = load i32, ptr %270, align 8
  %272 = add i32 %271, -1
  store i32 %272, ptr %270, align 8
  %273 = icmp eq i32 %272, 0
  br i1 %273, label %274, label %276

274:                                              ; preds = %269
  %275 = load ptr, ptr %13, align 8
  call void @_Py_Dealloc(ptr noundef %275) #7
  br label %276

276:                                              ; preds = %268, %269, %274
  store ptr @_Py_NoneStruct, ptr %14, align 8
  br label %279

277:                                              ; preds = %243, %224, %197, %188, %133, %116, %97, %86
  %278 = load ptr, ptr %18, align 8
  call void @Py_XDECREF(ptr noundef %278)
  store ptr null, ptr %14, align 8
  br label %279

279:                                              ; preds = %277, %276, %59, %39
  %280 = load ptr, ptr %14, align 8
  ret ptr %280
}

declare ptr @PyEval_EvalCodeEx(ptr noundef, ptr noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_format_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = call ptr @PyObject_Format(ptr noundef %7, ptr noundef %8)
  ret ptr %9
}

declare ptr @PyObject_Format(ptr noundef, ptr noundef) #1

declare ptr @PyObject_GetAttr(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_globals_impl(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = call ptr @PyEval_GetGlobals()
  store ptr %4, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = call ptr @_Py_XNewRef(ptr noundef %5)
  ret ptr %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_hasattr_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  %11 = load ptr, ptr %8, align 8
  %12 = load ptr, ptr %9, align 8
  %13 = call i32 @PyObject_GetOptionalAttr(ptr noundef %11, ptr noundef %12, ptr noundef %10)
  %14 = icmp slt i32 %13, 0
  br i1 %14, label %15, label %16

15:                                               ; preds = %3
  store ptr null, ptr %6, align 8
  br label %37

16:                                               ; preds = %3
  %17 = load ptr, ptr %10, align 8
  %18 = icmp eq ptr %17, null
  br i1 %18, label %19, label %20

19:                                               ; preds = %16
  store ptr @_Py_FalseStruct, ptr %6, align 8
  br label %37

20:                                               ; preds = %16
  %21 = load ptr, ptr %10, align 8
  store ptr %21, ptr %5, align 8
  %22 = load ptr, ptr %5, align 8
  store ptr %22, ptr %4, align 8
  %23 = load ptr, ptr %4, align 8
  %24 = load i32, ptr %23, align 8
  %25 = icmp slt i32 %24, 0
  %26 = zext i1 %25 to i32
  %27 = icmp ne i32 %26, 0
  br i1 %27, label %28, label %29

28:                                               ; preds = %20
  br label %36

29:                                               ; preds = %20
  %30 = load ptr, ptr %5, align 8
  %31 = load i32, ptr %30, align 8
  %32 = add i32 %31, -1
  store i32 %32, ptr %30, align 8
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %34, label %36

34:                                               ; preds = %29
  %35 = load ptr, ptr %5, align 8
  call void @_Py_Dealloc(ptr noundef %35) #7
  br label %36

36:                                               ; preds = %28, %29, %34
  store ptr @_Py_TrueStruct, ptr %6, align 8
  br label %37

37:                                               ; preds = %36, %19, %15
  %38 = load ptr, ptr %6, align 8
  ret ptr %38
}

declare i64 @PyObject_Hash(ptr noundef) #1

declare ptr @PyLong_FromSsize_t(i64 noundef) #1

declare ptr @PyLong_FromVoidPtr(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_input_impl(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca ptr, align 8
  %25 = alloca i64, align 8
  %26 = alloca i32, align 4
  %27 = alloca ptr, align 8
  %28 = alloca ptr, align 8
  %29 = alloca ptr, align 8
  %30 = alloca ptr, align 8
  %31 = alloca ptr, align 8
  %32 = alloca ptr, align 8
  %33 = alloca ptr, align 8
  %34 = alloca ptr, align 8
  %35 = alloca ptr, align 8
  %36 = alloca ptr, align 8
  %37 = alloca i64, align 8
  %38 = alloca ptr, align 8
  %39 = alloca ptr, align 8
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca ptr, align 8
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca ptr, align 8
  %46 = alloca ptr, align 8
  store ptr %0, ptr %18, align 8
  store ptr %1, ptr %19, align 8
  %47 = call ptr @_PyThreadState_GET()
  store ptr %47, ptr %20, align 8
  %48 = load ptr, ptr %20, align 8
  %49 = call ptr @_PySys_GetAttr(ptr noundef %48, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 657))
  store ptr %49, ptr %21, align 8
  %50 = load ptr, ptr %20, align 8
  %51 = call ptr @_PySys_GetAttr(ptr noundef %50, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 658))
  store ptr %51, ptr %22, align 8
  %52 = load ptr, ptr %20, align 8
  %53 = call ptr @_PySys_GetAttr(ptr noundef %52, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 656))
  store ptr %53, ptr %23, align 8
  %54 = load ptr, ptr %21, align 8
  %55 = icmp eq ptr %54, null
  br i1 %55, label %59, label %56

56:                                               ; preds = %2
  %57 = load ptr, ptr %21, align 8
  %58 = icmp eq ptr %57, @_Py_NoneStruct
  br i1 %58, label %59, label %61

59:                                               ; preds = %56, %2
  %60 = load ptr, ptr @PyExc_RuntimeError, align 8
  call void @PyErr_SetString(ptr noundef %60, ptr noundef @.str.175)
  store ptr null, ptr %17, align 8
  br label %503

61:                                               ; preds = %56
  %62 = load ptr, ptr %22, align 8
  %63 = icmp eq ptr %62, null
  br i1 %63, label %67, label %64

64:                                               ; preds = %61
  %65 = load ptr, ptr %22, align 8
  %66 = icmp eq ptr %65, @_Py_NoneStruct
  br i1 %66, label %67, label %69

67:                                               ; preds = %64, %61
  %68 = load ptr, ptr @PyExc_RuntimeError, align 8
  call void @PyErr_SetString(ptr noundef %68, ptr noundef @.str.176)
  store ptr null, ptr %17, align 8
  br label %503

69:                                               ; preds = %64
  %70 = load ptr, ptr %23, align 8
  %71 = icmp eq ptr %70, null
  br i1 %71, label %75, label %72

72:                                               ; preds = %69
  %73 = load ptr, ptr %23, align 8
  %74 = icmp eq ptr %73, @_Py_NoneStruct
  br i1 %74, label %75, label %77

75:                                               ; preds = %72, %69
  %76 = load ptr, ptr @PyExc_RuntimeError, align 8
  call void @PyErr_SetString(ptr noundef %76, ptr noundef @.str.177)
  store ptr null, ptr %17, align 8
  br label %503

77:                                               ; preds = %72
  %78 = load ptr, ptr %19, align 8
  %79 = icmp ne ptr %78, null
  br i1 %79, label %80, label %82

80:                                               ; preds = %77
  %81 = load ptr, ptr %19, align 8
  br label %83

82:                                               ; preds = %77
  br label %83

83:                                               ; preds = %82, %80
  %84 = phi ptr [ %81, %80 ], [ @_Py_NoneStruct, %82 ]
  %85 = call i32 (ptr, ptr, ...) @PySys_Audit(ptr noundef @.str.178, ptr noundef @.str.138, ptr noundef %84)
  %86 = icmp slt i32 %85, 0
  br i1 %86, label %87, label %88

87:                                               ; preds = %83
  store ptr null, ptr %17, align 8
  br label %503

88:                                               ; preds = %83
  %89 = load ptr, ptr %23, align 8
  %90 = call i32 @_PyFile_Flush(ptr noundef %89)
  %91 = icmp slt i32 %90, 0
  br i1 %91, label %92, label %93

92:                                               ; preds = %88
  call void @PyErr_Clear()
  br label %93

93:                                               ; preds = %92, %88
  %94 = load ptr, ptr %21, align 8
  %95 = call ptr @PyObject_CallMethodNoArgs(ptr noundef %94, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 374))
  store ptr %95, ptr %24, align 8
  %96 = load ptr, ptr %24, align 8
  %97 = icmp eq ptr %96, null
  br i1 %97, label %98, label %99

98:                                               ; preds = %93
  call void @PyErr_Clear()
  store i32 0, ptr %26, align 4
  br label %138

99:                                               ; preds = %93
  %100 = load ptr, ptr %24, align 8
  %101 = call i64 @PyLong_AsLong(ptr noundef %100)
  store i64 %101, ptr %25, align 8
  %102 = load ptr, ptr %24, align 8
  store ptr %102, ptr %10, align 8
  %103 = load ptr, ptr %10, align 8
  store ptr %103, ptr %9, align 8
  %104 = load ptr, ptr %9, align 8
  %105 = load i32, ptr %104, align 8
  %106 = icmp slt i32 %105, 0
  %107 = zext i1 %106 to i32
  %108 = icmp ne i32 %107, 0
  br i1 %108, label %109, label %110

109:                                              ; preds = %99
  br label %117

110:                                              ; preds = %99
  %111 = load ptr, ptr %10, align 8
  %112 = load i32, ptr %111, align 8
  %113 = add i32 %112, -1
  store i32 %113, ptr %111, align 8
  %114 = icmp eq i32 %113, 0
  br i1 %114, label %115, label %117

115:                                              ; preds = %110
  %116 = load ptr, ptr %10, align 8
  call void @_Py_Dealloc(ptr noundef %116) #7
  br label %117

117:                                              ; preds = %109, %110, %115
  %118 = load i64, ptr %25, align 8
  %119 = icmp slt i64 %118, 0
  br i1 %119, label %120, label %124

120:                                              ; preds = %117
  %121 = call ptr @PyErr_Occurred()
  %122 = icmp ne ptr %121, null
  br i1 %122, label %123, label %124

123:                                              ; preds = %120
  store ptr null, ptr %17, align 8
  br label %503

124:                                              ; preds = %120, %117
  %125 = load i64, ptr %25, align 8
  %126 = load ptr, ptr @__stdinp, align 8
  %127 = call i32 @fileno(ptr noundef %126)
  %128 = sext i32 %127 to i64
  %129 = icmp eq i64 %125, %128
  br i1 %129, label %130, label %135

130:                                              ; preds = %124
  %131 = load i64, ptr %25, align 8
  %132 = trunc i64 %131 to i32
  %133 = call i32 @isatty(i32 noundef %132)
  %134 = icmp ne i32 %133, 0
  br label %135

135:                                              ; preds = %130, %124
  %136 = phi i1 [ false, %124 ], [ %134, %130 ]
  %137 = zext i1 %136 to i32
  store i32 %137, ptr %26, align 4
  br label %138

138:                                              ; preds = %135, %98
  %139 = load i32, ptr %26, align 4
  %140 = icmp ne i32 %139, 0
  br i1 %140, label %141, label %187

141:                                              ; preds = %138
  %142 = load ptr, ptr %22, align 8
  %143 = call ptr @PyObject_CallMethodNoArgs(ptr noundef %142, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 374))
  store ptr %143, ptr %24, align 8
  %144 = load ptr, ptr %24, align 8
  %145 = icmp eq ptr %144, null
  br i1 %145, label %146, label %147

146:                                              ; preds = %141
  call void @PyErr_Clear()
  store i32 0, ptr %26, align 4
  br label %186

147:                                              ; preds = %141
  %148 = load ptr, ptr %24, align 8
  %149 = call i64 @PyLong_AsLong(ptr noundef %148)
  store i64 %149, ptr %25, align 8
  %150 = load ptr, ptr %24, align 8
  store ptr %150, ptr %11, align 8
  %151 = load ptr, ptr %11, align 8
  store ptr %151, ptr %8, align 8
  %152 = load ptr, ptr %8, align 8
  %153 = load i32, ptr %152, align 8
  %154 = icmp slt i32 %153, 0
  %155 = zext i1 %154 to i32
  %156 = icmp ne i32 %155, 0
  br i1 %156, label %157, label %158

157:                                              ; preds = %147
  br label %165

158:                                              ; preds = %147
  %159 = load ptr, ptr %11, align 8
  %160 = load i32, ptr %159, align 8
  %161 = add i32 %160, -1
  store i32 %161, ptr %159, align 8
  %162 = icmp eq i32 %161, 0
  br i1 %162, label %163, label %165

163:                                              ; preds = %158
  %164 = load ptr, ptr %11, align 8
  call void @_Py_Dealloc(ptr noundef %164) #7
  br label %165

165:                                              ; preds = %157, %158, %163
  %166 = load i64, ptr %25, align 8
  %167 = icmp slt i64 %166, 0
  br i1 %167, label %168, label %172

168:                                              ; preds = %165
  %169 = call ptr @PyErr_Occurred()
  %170 = icmp ne ptr %169, null
  br i1 %170, label %171, label %172

171:                                              ; preds = %168
  store ptr null, ptr %17, align 8
  br label %503

172:                                              ; preds = %168, %165
  %173 = load i64, ptr %25, align 8
  %174 = load ptr, ptr @__stdoutp, align 8
  %175 = call i32 @fileno(ptr noundef %174)
  %176 = sext i32 %175 to i64
  %177 = icmp eq i64 %173, %176
  br i1 %177, label %178, label %183

178:                                              ; preds = %172
  %179 = load i64, ptr %25, align 8
  %180 = trunc i64 %179 to i32
  %181 = call i32 @isatty(i32 noundef %180)
  %182 = icmp ne i32 %181, 0
  br label %183

183:                                              ; preds = %178, %172
  %184 = phi i1 [ false, %172 ], [ %182, %178 ]
  %185 = zext i1 %184 to i32
  store i32 %185, ptr %26, align 4
  br label %186

186:                                              ; preds = %183, %146
  br label %187

187:                                              ; preds = %186, %138
  %188 = load i32, ptr %26, align 4
  %189 = icmp ne i32 %188, 0
  br i1 %189, label %190, label %485

190:                                              ; preds = %187
  store ptr null, ptr %27, align 8
  store ptr null, ptr %29, align 8
  store ptr null, ptr %30, align 8
  store ptr null, ptr %31, align 8
  store ptr null, ptr %32, align 8
  store ptr null, ptr %33, align 8
  %191 = load ptr, ptr %21, align 8
  %192 = call ptr @PyObject_GetAttr(ptr noundef %191, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 342))
  store ptr %192, ptr %30, align 8
  %193 = load ptr, ptr %30, align 8
  %194 = icmp eq ptr %193, null
  br i1 %194, label %195, label %196

195:                                              ; preds = %190
  store i32 0, ptr %26, align 4
  br label %475

196:                                              ; preds = %190
  %197 = load ptr, ptr %21, align 8
  %198 = call ptr @PyObject_GetAttr(ptr noundef %197, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 350))
  store ptr %198, ptr %31, align 8
  %199 = load ptr, ptr %31, align 8
  %200 = icmp eq ptr %199, null
  br i1 %200, label %201, label %202

201:                                              ; preds = %196
  store i32 0, ptr %26, align 4
  br label %475

202:                                              ; preds = %196
  %203 = load ptr, ptr %30, align 8
  %204 = call ptr @_Py_TYPE(ptr noundef %203)
  %205 = call i32 @PyType_HasFeature(ptr noundef %204, i64 noundef 268435456)
  %206 = icmp ne i32 %205, 0
  br i1 %206, label %207, label %212

207:                                              ; preds = %202
  %208 = load ptr, ptr %31, align 8
  %209 = call ptr @_Py_TYPE(ptr noundef %208)
  %210 = call i32 @PyType_HasFeature(ptr noundef %209, i64 noundef 268435456)
  %211 = icmp ne i32 %210, 0
  br i1 %211, label %213, label %212

212:                                              ; preds = %207, %202
  store i32 0, ptr %26, align 4
  br label %475

213:                                              ; preds = %207
  %214 = load ptr, ptr %30, align 8
  %215 = call ptr @PyUnicode_AsUTF8(ptr noundef %214)
  store ptr %215, ptr %34, align 8
  %216 = load ptr, ptr %34, align 8
  %217 = icmp eq ptr %216, null
  br i1 %217, label %218, label %219

218:                                              ; preds = %213
  br label %475

219:                                              ; preds = %213
  %220 = load ptr, ptr %31, align 8
  %221 = call ptr @PyUnicode_AsUTF8(ptr noundef %220)
  store ptr %221, ptr %35, align 8
  %222 = load ptr, ptr %35, align 8
  %223 = icmp eq ptr %222, null
  br i1 %223, label %224, label %225

224:                                              ; preds = %219
  br label %475

225:                                              ; preds = %219
  %226 = load ptr, ptr %22, align 8
  %227 = call i32 @_PyFile_Flush(ptr noundef %226)
  %228 = icmp slt i32 %227, 0
  br i1 %228, label %229, label %230

229:                                              ; preds = %225
  call void @PyErr_Clear()
  br label %230

230:                                              ; preds = %229, %225
  %231 = load ptr, ptr %19, align 8
  %232 = icmp ne ptr %231, null
  br i1 %232, label %233, label %380

233:                                              ; preds = %230
  %234 = load ptr, ptr %22, align 8
  %235 = call ptr @PyObject_GetAttr(ptr noundef %234, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 342))
  store ptr %235, ptr %32, align 8
  %236 = load ptr, ptr %32, align 8
  %237 = icmp eq ptr %236, null
  br i1 %237, label %238, label %239

238:                                              ; preds = %233
  store i32 0, ptr %26, align 4
  br label %475

239:                                              ; preds = %233
  %240 = load ptr, ptr %22, align 8
  %241 = call ptr @PyObject_GetAttr(ptr noundef %240, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 350))
  store ptr %241, ptr %33, align 8
  %242 = load ptr, ptr %33, align 8
  %243 = icmp eq ptr %242, null
  br i1 %243, label %244, label %245

244:                                              ; preds = %239
  store i32 0, ptr %26, align 4
  br label %475

245:                                              ; preds = %239
  %246 = load ptr, ptr %32, align 8
  %247 = call ptr @_Py_TYPE(ptr noundef %246)
  %248 = call i32 @PyType_HasFeature(ptr noundef %247, i64 noundef 268435456)
  %249 = icmp ne i32 %248, 0
  br i1 %249, label %250, label %255

250:                                              ; preds = %245
  %251 = load ptr, ptr %33, align 8
  %252 = call ptr @_Py_TYPE(ptr noundef %251)
  %253 = call i32 @PyType_HasFeature(ptr noundef %252, i64 noundef 268435456)
  %254 = icmp ne i32 %253, 0
  br i1 %254, label %256, label %255

255:                                              ; preds = %250, %245
  store i32 0, ptr %26, align 4
  br label %475

256:                                              ; preds = %250
  %257 = load ptr, ptr %32, align 8
  %258 = call ptr @PyUnicode_AsUTF8(ptr noundef %257)
  store ptr %258, ptr %38, align 8
  %259 = load ptr, ptr %38, align 8
  %260 = icmp eq ptr %259, null
  br i1 %260, label %261, label %262

261:                                              ; preds = %256
  br label %475

262:                                              ; preds = %256
  %263 = load ptr, ptr %33, align 8
  %264 = call ptr @PyUnicode_AsUTF8(ptr noundef %263)
  store ptr %264, ptr %39, align 8
  %265 = load ptr, ptr %39, align 8
  %266 = icmp eq ptr %265, null
  br i1 %266, label %267, label %268

267:                                              ; preds = %262
  br label %475

268:                                              ; preds = %262
  %269 = load ptr, ptr %19, align 8
  %270 = call ptr @PyObject_Str(ptr noundef %269)
  store ptr %270, ptr %40, align 8
  %271 = load ptr, ptr %40, align 8
  %272 = icmp eq ptr %271, null
  br i1 %272, label %273, label %274

273:                                              ; preds = %268
  br label %475

274:                                              ; preds = %268
  %275 = load ptr, ptr %40, align 8
  %276 = load ptr, ptr %38, align 8
  %277 = load ptr, ptr %39, align 8
  %278 = call ptr @PyUnicode_AsEncodedString(ptr noundef %275, ptr noundef %276, ptr noundef %277)
  store ptr %278, ptr %27, align 8
  br label %279

279:                                              ; preds = %274
  store ptr %32, ptr %41, align 8
  %280 = load ptr, ptr %41, align 8
  %281 = load ptr, ptr %280, align 8
  store ptr %281, ptr %42, align 8
  %282 = load ptr, ptr %42, align 8
  %283 = icmp ne ptr %282, null
  br i1 %283, label %284, label %302

284:                                              ; preds = %279
  %285 = load ptr, ptr %41, align 8
  store ptr null, ptr %285, align 8
  %286 = load ptr, ptr %42, align 8
  store ptr %286, ptr %12, align 8
  %287 = load ptr, ptr %12, align 8
  store ptr %287, ptr %7, align 8
  %288 = load ptr, ptr %7, align 8
  %289 = load i32, ptr %288, align 8
  %290 = icmp slt i32 %289, 0
  %291 = zext i1 %290 to i32
  %292 = icmp ne i32 %291, 0
  br i1 %292, label %293, label %294

293:                                              ; preds = %284
  br label %301

294:                                              ; preds = %284
  %295 = load ptr, ptr %12, align 8
  %296 = load i32, ptr %295, align 8
  %297 = add i32 %296, -1
  store i32 %297, ptr %295, align 8
  %298 = icmp eq i32 %297, 0
  br i1 %298, label %299, label %301

299:                                              ; preds = %294
  %300 = load ptr, ptr %12, align 8
  call void @_Py_Dealloc(ptr noundef %300) #7
  br label %301

301:                                              ; preds = %293, %294, %299
  br label %302

302:                                              ; preds = %301, %279
  br label %303

303:                                              ; preds = %302
  br label %304

304:                                              ; preds = %303
  store ptr %33, ptr %43, align 8
  %305 = load ptr, ptr %43, align 8
  %306 = load ptr, ptr %305, align 8
  store ptr %306, ptr %44, align 8
  %307 = load ptr, ptr %44, align 8
  %308 = icmp ne ptr %307, null
  br i1 %308, label %309, label %327

309:                                              ; preds = %304
  %310 = load ptr, ptr %43, align 8
  store ptr null, ptr %310, align 8
  %311 = load ptr, ptr %44, align 8
  store ptr %311, ptr %13, align 8
  %312 = load ptr, ptr %13, align 8
  store ptr %312, ptr %6, align 8
  %313 = load ptr, ptr %6, align 8
  %314 = load i32, ptr %313, align 8
  %315 = icmp slt i32 %314, 0
  %316 = zext i1 %315 to i32
  %317 = icmp ne i32 %316, 0
  br i1 %317, label %318, label %319

318:                                              ; preds = %309
  br label %326

319:                                              ; preds = %309
  %320 = load ptr, ptr %13, align 8
  %321 = load i32, ptr %320, align 8
  %322 = add i32 %321, -1
  store i32 %322, ptr %320, align 8
  %323 = icmp eq i32 %322, 0
  br i1 %323, label %324, label %326

324:                                              ; preds = %319
  %325 = load ptr, ptr %13, align 8
  call void @_Py_Dealloc(ptr noundef %325) #7
  br label %326

326:                                              ; preds = %318, %319, %324
  br label %327

327:                                              ; preds = %326, %304
  br label %328

328:                                              ; preds = %327
  br label %329

329:                                              ; preds = %328
  store ptr %40, ptr %45, align 8
  %330 = load ptr, ptr %45, align 8
  %331 = load ptr, ptr %330, align 8
  store ptr %331, ptr %46, align 8
  %332 = load ptr, ptr %46, align 8
  %333 = icmp ne ptr %332, null
  br i1 %333, label %334, label %352

334:                                              ; preds = %329
  %335 = load ptr, ptr %45, align 8
  store ptr null, ptr %335, align 8
  %336 = load ptr, ptr %46, align 8
  store ptr %336, ptr %14, align 8
  %337 = load ptr, ptr %14, align 8
  store ptr %337, ptr %5, align 8
  %338 = load ptr, ptr %5, align 8
  %339 = load i32, ptr %338, align 8
  %340 = icmp slt i32 %339, 0
  %341 = zext i1 %340 to i32
  %342 = icmp ne i32 %341, 0
  br i1 %342, label %343, label %344

343:                                              ; preds = %334
  br label %351

344:                                              ; preds = %334
  %345 = load ptr, ptr %14, align 8
  %346 = load i32, ptr %345, align 8
  %347 = add i32 %346, -1
  store i32 %347, ptr %345, align 8
  %348 = icmp eq i32 %347, 0
  br i1 %348, label %349, label %351

349:                                              ; preds = %344
  %350 = load ptr, ptr %14, align 8
  call void @_Py_Dealloc(ptr noundef %350) #7
  br label %351

351:                                              ; preds = %343, %344, %349
  br label %352

352:                                              ; preds = %351, %329
  br label %353

353:                                              ; preds = %352
  %354 = load ptr, ptr %27, align 8
  %355 = icmp eq ptr %354, null
  br i1 %355, label %356, label %357

356:                                              ; preds = %353
  br label %475

357:                                              ; preds = %353
  %358 = load ptr, ptr %27, align 8
  %359 = call ptr @_Py_TYPE(ptr noundef %358)
  %360 = call i32 @PyType_HasFeature(ptr noundef %359, i64 noundef 134217728)
  %361 = icmp ne i32 %360, 0
  %362 = xor i1 %361, true
  %363 = zext i1 %362 to i32
  %364 = sext i32 %363 to i64
  %365 = icmp ne i64 %364, 0
  br i1 %365, label %366, label %368

366:                                              ; preds = %357
  call void @__assert_rtn(ptr noundef @__func__.builtin_input_impl, ptr noundef @.str.34, i32 noundef 2391, ptr noundef @.str.179) #8
  unreachable

367:                                              ; No predecessors!
  br label %369

368:                                              ; preds = %357
  br label %369

369:                                              ; preds = %368, %367
  %370 = load ptr, ptr %27, align 8
  %371 = call ptr @PyBytes_AS_STRING(ptr noundef %370)
  store ptr %371, ptr %28, align 8
  %372 = load ptr, ptr %28, align 8
  %373 = call i64 @strlen(ptr noundef %372) #7
  %374 = load ptr, ptr %27, align 8
  %375 = call i64 @PyBytes_GET_SIZE(ptr noundef %374)
  %376 = icmp ne i64 %373, %375
  br i1 %376, label %377, label %379

377:                                              ; preds = %369
  %378 = load ptr, ptr @PyExc_ValueError, align 8
  call void @PyErr_SetString(ptr noundef %378, ptr noundef @.str.180)
  br label %475

379:                                              ; preds = %369
  br label %381

380:                                              ; preds = %230
  store ptr null, ptr %27, align 8
  store ptr @.str.156, ptr %28, align 8
  br label %381

381:                                              ; preds = %380, %379
  %382 = load ptr, ptr @__stdinp, align 8
  %383 = load ptr, ptr @__stdoutp, align 8
  %384 = load ptr, ptr %28, align 8
  %385 = call ptr @PyOS_Readline(ptr noundef %382, ptr noundef %383, ptr noundef %384)
  store ptr %385, ptr %29, align 8
  %386 = load ptr, ptr %29, align 8
  %387 = icmp eq ptr %386, null
  br i1 %387, label %388, label %395

388:                                              ; preds = %381
  %389 = call i32 @PyErr_CheckSignals()
  %390 = call ptr @PyErr_Occurred()
  %391 = icmp ne ptr %390, null
  br i1 %391, label %394, label %392

392:                                              ; preds = %388
  %393 = load ptr, ptr @PyExc_KeyboardInterrupt, align 8
  call void @PyErr_SetNone(ptr noundef %393)
  br label %394

394:                                              ; preds = %392, %388
  br label %475

395:                                              ; preds = %381
  %396 = load ptr, ptr %29, align 8
  %397 = call i64 @strlen(ptr noundef %396) #7
  store i64 %397, ptr %37, align 8
  %398 = load i64, ptr %37, align 8
  %399 = icmp eq i64 %398, 0
  br i1 %399, label %400, label %402

400:                                              ; preds = %395
  %401 = load ptr, ptr @PyExc_EOFError, align 8
  call void @PyErr_SetNone(ptr noundef %401)
  store ptr null, ptr %36, align 8
  br label %430

402:                                              ; preds = %395
  %403 = load i64, ptr %37, align 8
  %404 = icmp ugt i64 %403, 9223372036854775807
  br i1 %404, label %405, label %407

405:                                              ; preds = %402
  %406 = load ptr, ptr @PyExc_OverflowError, align 8
  call void @PyErr_SetString(ptr noundef %406, ptr noundef @.str.181)
  store ptr null, ptr %36, align 8
  br label %429

407:                                              ; preds = %402
  %408 = load i64, ptr %37, align 8
  %409 = add i64 %408, -1
  store i64 %409, ptr %37, align 8
  %410 = load i64, ptr %37, align 8
  %411 = icmp ne i64 %410, 0
  br i1 %411, label %412, label %423

412:                                              ; preds = %407
  %413 = load ptr, ptr %29, align 8
  %414 = load i64, ptr %37, align 8
  %415 = sub i64 %414, 1
  %416 = getelementptr inbounds i8, ptr %413, i64 %415
  %417 = load i8, ptr %416, align 1
  %418 = sext i8 %417 to i32
  %419 = icmp eq i32 %418, 13
  br i1 %419, label %420, label %423

420:                                              ; preds = %412
  %421 = load i64, ptr %37, align 8
  %422 = add i64 %421, -1
  store i64 %422, ptr %37, align 8
  br label %423

423:                                              ; preds = %420, %412, %407
  %424 = load ptr, ptr %29, align 8
  %425 = load i64, ptr %37, align 8
  %426 = load ptr, ptr %34, align 8
  %427 = load ptr, ptr %35, align 8
  %428 = call ptr @PyUnicode_Decode(ptr noundef %424, i64 noundef %425, ptr noundef %426, ptr noundef %427)
  store ptr %428, ptr %36, align 8
  br label %429

429:                                              ; preds = %423, %405
  br label %430

430:                                              ; preds = %429, %400
  %431 = load ptr, ptr %30, align 8
  store ptr %431, ptr %15, align 8
  %432 = load ptr, ptr %15, align 8
  store ptr %432, ptr %4, align 8
  %433 = load ptr, ptr %4, align 8
  %434 = load i32, ptr %433, align 8
  %435 = icmp slt i32 %434, 0
  %436 = zext i1 %435 to i32
  %437 = icmp ne i32 %436, 0
  br i1 %437, label %438, label %439

438:                                              ; preds = %430
  br label %446

439:                                              ; preds = %430
  %440 = load ptr, ptr %15, align 8
  %441 = load i32, ptr %440, align 8
  %442 = add i32 %441, -1
  store i32 %442, ptr %440, align 8
  %443 = icmp eq i32 %442, 0
  br i1 %443, label %444, label %446

444:                                              ; preds = %439
  %445 = load ptr, ptr %15, align 8
  call void @_Py_Dealloc(ptr noundef %445) #7
  br label %446

446:                                              ; preds = %438, %439, %444
  %447 = load ptr, ptr %31, align 8
  store ptr %447, ptr %16, align 8
  %448 = load ptr, ptr %16, align 8
  store ptr %448, ptr %3, align 8
  %449 = load ptr, ptr %3, align 8
  %450 = load i32, ptr %449, align 8
  %451 = icmp slt i32 %450, 0
  %452 = zext i1 %451 to i32
  %453 = icmp ne i32 %452, 0
  br i1 %453, label %454, label %455

454:                                              ; preds = %446
  br label %462

455:                                              ; preds = %446
  %456 = load ptr, ptr %16, align 8
  %457 = load i32, ptr %456, align 8
  %458 = add i32 %457, -1
  store i32 %458, ptr %456, align 8
  %459 = icmp eq i32 %458, 0
  br i1 %459, label %460, label %462

460:                                              ; preds = %455
  %461 = load ptr, ptr %16, align 8
  call void @_Py_Dealloc(ptr noundef %461) #7
  br label %462

462:                                              ; preds = %454, %455, %460
  %463 = load ptr, ptr %27, align 8
  call void @Py_XDECREF(ptr noundef %463)
  %464 = load ptr, ptr %29, align 8
  call void @PyMem_Free(ptr noundef %464)
  %465 = load ptr, ptr %36, align 8
  %466 = icmp ne ptr %465, null
  br i1 %466, label %467, label %473

467:                                              ; preds = %462
  %468 = load ptr, ptr %36, align 8
  %469 = call i32 (ptr, ptr, ...) @PySys_Audit(ptr noundef @.str.182, ptr noundef @.str.138, ptr noundef %468)
  %470 = icmp slt i32 %469, 0
  br i1 %470, label %471, label %472

471:                                              ; preds = %467
  store ptr null, ptr %17, align 8
  br label %503

472:                                              ; preds = %467
  br label %473

473:                                              ; preds = %472, %462
  %474 = load ptr, ptr %36, align 8
  store ptr %474, ptr %17, align 8
  br label %503

475:                                              ; preds = %394, %377, %356, %273, %267, %261, %255, %244, %238, %224, %218, %212, %201, %195
  %476 = load ptr, ptr %30, align 8
  call void @Py_XDECREF(ptr noundef %476)
  %477 = load ptr, ptr %32, align 8
  call void @Py_XDECREF(ptr noundef %477)
  %478 = load ptr, ptr %31, align 8
  call void @Py_XDECREF(ptr noundef %478)
  %479 = load ptr, ptr %33, align 8
  call void @Py_XDECREF(ptr noundef %479)
  %480 = load ptr, ptr %27, align 8
  call void @Py_XDECREF(ptr noundef %480)
  %481 = load i32, ptr %26, align 4
  %482 = icmp ne i32 %481, 0
  br i1 %482, label %483, label %484

483:                                              ; preds = %475
  store ptr null, ptr %17, align 8
  br label %503

484:                                              ; preds = %475
  call void @PyErr_Clear()
  br label %485

485:                                              ; preds = %484, %187
  %486 = load ptr, ptr %19, align 8
  %487 = icmp ne ptr %486, null
  br i1 %487, label %488, label %495

488:                                              ; preds = %485
  %489 = load ptr, ptr %19, align 8
  %490 = load ptr, ptr %22, align 8
  %491 = call i32 @PyFile_WriteObject(ptr noundef %489, ptr noundef %490, i32 noundef 1)
  %492 = icmp ne i32 %491, 0
  br i1 %492, label %493, label %494

493:                                              ; preds = %488
  store ptr null, ptr %17, align 8
  br label %503

494:                                              ; preds = %488
  br label %495

495:                                              ; preds = %494, %485
  %496 = load ptr, ptr %22, align 8
  %497 = call i32 @_PyFile_Flush(ptr noundef %496)
  %498 = icmp slt i32 %497, 0
  br i1 %498, label %499, label %500

499:                                              ; preds = %495
  call void @PyErr_Clear()
  br label %500

500:                                              ; preds = %499, %495
  %501 = load ptr, ptr %21, align 8
  %502 = call ptr @PyFile_GetLine(ptr noundef %501, i32 noundef -1)
  store ptr %502, ptr %17, align 8
  br label %503

503:                                              ; preds = %500, %493, %483, %473, %471, %171, %123, %87, %75, %67, %59
  %504 = load ptr, ptr %17, align 8
  ret ptr %504
}

declare ptr @_PySys_GetAttr(ptr noundef, ptr noundef) #1

declare i32 @_PyFile_Flush(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @PyObject_CallMethodNoArgs(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  store i64 -9223372036854775807, ptr %5, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load i64, ptr %5, align 8
  %8 = call ptr @PyObject_VectorcallMethod(ptr noundef %6, ptr noundef %3, i64 noundef %7, ptr noundef null)
  ret ptr %8
}

declare i64 @PyLong_AsLong(ptr noundef) #1

declare i32 @fileno(ptr noundef) #1

declare i32 @isatty(i32 noundef) #1

declare ptr @PyUnicode_AsUTF8(ptr noundef) #1

declare ptr @PyObject_Str(ptr noundef) #1

declare ptr @PyUnicode_AsEncodedString(ptr noundef, ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @PyBytes_AS_STRING(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_Py_TYPE(ptr noundef %3)
  %5 = call i32 @PyType_HasFeature(ptr noundef %4, i64 noundef 134217728)
  %6 = icmp ne i32 %5, 0
  %7 = xor i1 %6, true
  %8 = zext i1 %7 to i32
  %9 = sext i32 %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyBytes_AS_STRING, ptr noundef @.str.183, i32 noundef 25, ptr noundef @.str.184) #8
  unreachable

12:                                               ; No predecessors!
  br label %14

13:                                               ; preds = %1
  br label %14

14:                                               ; preds = %13, %12
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds %struct.PyBytesObject, ptr %15, i32 0, i32 2
  %17 = getelementptr inbounds [1 x i8], ptr %16, i64 0, i64 0
  ret ptr %17
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @PyBytes_GET_SIZE(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call ptr @_Py_TYPE(ptr noundef %4)
  %6 = call i32 @PyType_HasFeature(ptr noundef %5, i64 noundef 134217728)
  %7 = icmp ne i32 %6, 0
  %8 = xor i1 %7, true
  %9 = zext i1 %8 to i32
  %10 = sext i32 %9 to i64
  %11 = icmp ne i64 %10, 0
  br i1 %11, label %12, label %14

12:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyBytes_GET_SIZE, ptr noundef @.str.183, i32 noundef 30, ptr noundef @.str.184) #8
  unreachable

13:                                               ; No predecessors!
  br label %15

14:                                               ; preds = %1
  br label %15

15:                                               ; preds = %14, %13
  %16 = load ptr, ptr %2, align 8
  store ptr %16, ptr %3, align 8
  %17 = load ptr, ptr %3, align 8
  %18 = call i64 @Py_SIZE(ptr noundef %17)
  ret i64 %18
}

declare ptr @PyOS_Readline(ptr noundef, ptr noundef, ptr noundef) #1

declare i32 @PyErr_CheckSignals() #1

declare void @PyErr_SetNone(ptr noundef) #1

declare ptr @PyUnicode_Decode(ptr noundef, i64 noundef, ptr noundef, ptr noundef) #1

declare i32 @PyFile_WriteObject(ptr noundef, ptr noundef, i32 noundef) #1

declare ptr @PyFile_GetLine(ptr noundef, i32 noundef) #1

declare ptr @PyObject_VectorcallMethod(ptr noundef, ptr noundef, i64 noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_isinstance_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = load ptr, ptr %7, align 8
  %11 = call i32 @PyObject_IsInstance(ptr noundef %9, ptr noundef %10)
  store i32 %11, ptr %8, align 4
  %12 = load i32, ptr %8, align 4
  %13 = icmp slt i32 %12, 0
  br i1 %13, label %14, label %15

14:                                               ; preds = %3
  store ptr null, ptr %4, align 8
  br label %19

15:                                               ; preds = %3
  %16 = load i32, ptr %8, align 4
  %17 = sext i32 %16 to i64
  %18 = call ptr @PyBool_FromLong(i64 noundef %17)
  store ptr %18, ptr %4, align 8
  br label %19

19:                                               ; preds = %15, %14
  %20 = load ptr, ptr %4, align 8
  ret ptr %20
}

declare i32 @PyObject_IsInstance(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_issubclass_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = load ptr, ptr %7, align 8
  %11 = call i32 @PyObject_IsSubclass(ptr noundef %9, ptr noundef %10)
  store i32 %11, ptr %8, align 4
  %12 = load i32, ptr %8, align 4
  %13 = icmp slt i32 %12, 0
  br i1 %13, label %14, label %15

14:                                               ; preds = %3
  store ptr null, ptr %4, align 8
  br label %19

15:                                               ; preds = %3
  %16 = load i32, ptr %8, align 4
  %17 = sext i32 %16 to i64
  %18 = call ptr @PyBool_FromLong(i64 noundef %17)
  store ptr %18, ptr %4, align 8
  br label %19

19:                                               ; preds = %15, %14
  %20 = load ptr, ptr %4, align 8
  ret ptr %20
}

declare i32 @PyObject_IsSubclass(ptr noundef, ptr noundef) #1

declare ptr @PyCallIter_New(ptr noundef, ptr noundef) #1

declare ptr @PyObject_GetAIter(ptr noundef) #1

declare i64 @PyObject_Size(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_locals_impl(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = call ptr @_PyEval_GetFrameLocals()
  ret ptr %3
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @min_max(ptr noundef %0, i64 noundef %1, ptr noundef %2, i32 noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i32, align 4
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca i64, align 8
  %24 = alloca ptr, align 8
  %25 = alloca i32, align 4
  %26 = alloca ptr, align 8
  %27 = alloca ptr, align 8
  %28 = alloca ptr, align 8
  %29 = alloca ptr, align 8
  %30 = alloca ptr, align 8
  %31 = alloca ptr, align 8
  %32 = alloca ptr, align 8
  %33 = alloca ptr, align 8
  %34 = alloca ptr, align 8
  %35 = alloca i32, align 4
  %36 = alloca i32, align 4
  store ptr %0, ptr %22, align 8
  store i64 %1, ptr %23, align 8
  store ptr %2, ptr %24, align 8
  store i32 %3, ptr %25, align 4
  store ptr null, ptr %26, align 8
  store ptr null, ptr %31, align 8
  store ptr null, ptr %32, align 8
  %37 = load i32, ptr %25, align 4
  %38 = icmp eq i32 %37, 0
  %39 = zext i1 %38 to i64
  %40 = select i1 %38, ptr @.str.106, ptr @.str.105
  store ptr %40, ptr %33, align 8
  %41 = load i32, ptr %25, align 4
  %42 = icmp eq i32 %41, 0
  %43 = zext i1 %42 to i64
  %44 = select i1 %42, ptr @min_max._parser_min, ptr @min_max._parser_max
  store ptr %44, ptr %34, align 8
  %45 = load i64, ptr %23, align 8
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %47, label %51

47:                                               ; preds = %4
  %48 = load ptr, ptr @PyExc_TypeError, align 8
  %49 = load ptr, ptr %33, align 8
  %50 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %48, ptr noundef @.str.191, ptr noundef %49)
  store ptr null, ptr %21, align 8
  br label %311

51:                                               ; preds = %4
  %52 = load ptr, ptr %24, align 8
  %53 = icmp ne ptr %52, null
  br i1 %53, label %54, label %63

54:                                               ; preds = %51
  %55 = load ptr, ptr %22, align 8
  %56 = load i64, ptr %23, align 8
  %57 = getelementptr inbounds ptr, ptr %55, i64 %56
  %58 = load ptr, ptr %24, align 8
  %59 = load ptr, ptr %34, align 8
  %60 = call i32 (ptr, i64, ptr, ptr, ...) @_PyArg_ParseStackAndKeywords(ptr noundef %57, i64 noundef 0, ptr noundef %58, ptr noundef %59, ptr noundef %31, ptr noundef %32)
  %61 = icmp ne i32 %60, 0
  br i1 %61, label %63, label %62

62:                                               ; preds = %54
  store ptr null, ptr %21, align 8
  br label %311

63:                                               ; preds = %54, %51
  %64 = load i64, ptr %23, align 8
  %65 = icmp sgt i64 %64, 1
  %66 = zext i1 %65 to i32
  store i32 %66, ptr %35, align 4
  %67 = load i32, ptr %35, align 4
  %68 = icmp ne i32 %67, 0
  br i1 %68, label %69, label %76

69:                                               ; preds = %63
  %70 = load ptr, ptr %32, align 8
  %71 = icmp ne ptr %70, null
  br i1 %71, label %72, label %76

72:                                               ; preds = %69
  %73 = load ptr, ptr @PyExc_TypeError, align 8
  %74 = load ptr, ptr %33, align 8
  %75 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %73, ptr noundef @.str.192, ptr noundef %74)
  store ptr null, ptr %21, align 8
  br label %311

76:                                               ; preds = %69, %63
  %77 = load i32, ptr %35, align 4
  %78 = icmp ne i32 %77, 0
  br i1 %78, label %88, label %79

79:                                               ; preds = %76
  %80 = load ptr, ptr %22, align 8
  %81 = getelementptr inbounds ptr, ptr %80, i64 0
  %82 = load ptr, ptr %81, align 8
  %83 = call ptr @PyObject_GetIter(ptr noundef %82)
  store ptr %83, ptr %26, align 8
  %84 = load ptr, ptr %26, align 8
  %85 = icmp eq ptr %84, null
  br i1 %85, label %86, label %87

86:                                               ; preds = %79
  store ptr null, ptr %21, align 8
  br label %311

87:                                               ; preds = %79
  br label %88

88:                                               ; preds = %87, %76
  %89 = load ptr, ptr %31, align 8
  %90 = icmp eq ptr %89, @_Py_NoneStruct
  br i1 %90, label %91, label %92

91:                                               ; preds = %88
  store ptr null, ptr %31, align 8
  br label %92

92:                                               ; preds = %91, %88
  store ptr null, ptr %29, align 8
  store ptr null, ptr %30, align 8
  br label %93

93:                                               ; preds = %92, %228
  %94 = load ptr, ptr %26, align 8
  %95 = icmp eq ptr %94, null
  br i1 %95, label %96, label %116

96:                                               ; preds = %93
  %97 = load i64, ptr %23, align 8
  %98 = add nsw i64 %97, -1
  store i64 %98, ptr %23, align 8
  %99 = icmp sle i64 %97, 0
  br i1 %99, label %100, label %101

100:                                              ; preds = %96
  br label %229

101:                                              ; preds = %96
  %102 = load ptr, ptr %22, align 8
  %103 = getelementptr inbounds ptr, ptr %102, i32 1
  store ptr %103, ptr %22, align 8
  %104 = load ptr, ptr %102, align 8
  store ptr %104, ptr %27, align 8
  %105 = load ptr, ptr %27, align 8
  store ptr %105, ptr %12, align 8
  %106 = load ptr, ptr %12, align 8
  %107 = load i32, ptr %106, align 8
  store i32 %107, ptr %13, align 4
  %108 = load i32, ptr %13, align 4
  %109 = icmp slt i32 %108, 0
  br i1 %109, label %110, label %111

110:                                              ; preds = %101
  br label %115

111:                                              ; preds = %101
  %112 = load i32, ptr %13, align 4
  %113 = add i32 %112, 1
  %114 = load ptr, ptr %12, align 8
  store i32 %113, ptr %114, align 8
  br label %115

115:                                              ; preds = %110, %111
  br label %127

116:                                              ; preds = %93
  %117 = load ptr, ptr %26, align 8
  %118 = call ptr @PyIter_Next(ptr noundef %117)
  store ptr %118, ptr %27, align 8
  %119 = load ptr, ptr %27, align 8
  %120 = icmp eq ptr %119, null
  br i1 %120, label %121, label %126

121:                                              ; preds = %116
  %122 = call ptr @PyErr_Occurred()
  %123 = icmp ne ptr %122, null
  br i1 %123, label %124, label %125

124:                                              ; preds = %121
  br label %307

125:                                              ; preds = %121
  br label %229

126:                                              ; preds = %116
  br label %127

127:                                              ; preds = %126, %115
  %128 = load ptr, ptr %31, align 8
  %129 = icmp ne ptr %128, null
  br i1 %129, label %130, label %138

130:                                              ; preds = %127
  %131 = load ptr, ptr %31, align 8
  %132 = load ptr, ptr %27, align 8
  %133 = call ptr @PyObject_CallOneArg(ptr noundef %131, ptr noundef %132)
  store ptr %133, ptr %28, align 8
  %134 = load ptr, ptr %28, align 8
  %135 = icmp eq ptr %134, null
  br i1 %135, label %136, label %137

136:                                              ; preds = %130
  br label %290

137:                                              ; preds = %130
  br label %141

138:                                              ; preds = %127
  %139 = load ptr, ptr %27, align 8
  %140 = call ptr @_Py_NewRef(ptr noundef %139)
  store ptr %140, ptr %28, align 8
  br label %141

141:                                              ; preds = %138, %137
  %142 = load ptr, ptr %30, align 8
  %143 = icmp eq ptr %142, null
  br i1 %143, label %144, label %147

144:                                              ; preds = %141
  %145 = load ptr, ptr %27, align 8
  store ptr %145, ptr %29, align 8
  %146 = load ptr, ptr %28, align 8
  store ptr %146, ptr %30, align 8
  br label %228

147:                                              ; preds = %141
  %148 = load ptr, ptr %28, align 8
  %149 = load ptr, ptr %30, align 8
  %150 = load i32, ptr %25, align 4
  %151 = call i32 @PyObject_RichCompareBool(ptr noundef %148, ptr noundef %149, i32 noundef %150)
  store i32 %151, ptr %36, align 4
  %152 = load i32, ptr %36, align 4
  %153 = icmp slt i32 %152, 0
  br i1 %153, label %154, label %155

154:                                              ; preds = %147
  br label %273

155:                                              ; preds = %147
  %156 = load i32, ptr %36, align 4
  %157 = icmp sgt i32 %156, 0
  br i1 %157, label %158, label %193

158:                                              ; preds = %155
  %159 = load ptr, ptr %30, align 8
  store ptr %159, ptr %14, align 8
  %160 = load ptr, ptr %14, align 8
  store ptr %160, ptr %11, align 8
  %161 = load ptr, ptr %11, align 8
  %162 = load i32, ptr %161, align 8
  %163 = icmp slt i32 %162, 0
  %164 = zext i1 %163 to i32
  %165 = icmp ne i32 %164, 0
  br i1 %165, label %166, label %167

166:                                              ; preds = %158
  br label %174

167:                                              ; preds = %158
  %168 = load ptr, ptr %14, align 8
  %169 = load i32, ptr %168, align 8
  %170 = add i32 %169, -1
  store i32 %170, ptr %168, align 8
  %171 = icmp eq i32 %170, 0
  br i1 %171, label %172, label %174

172:                                              ; preds = %167
  %173 = load ptr, ptr %14, align 8
  call void @_Py_Dealloc(ptr noundef %173) #7
  br label %174

174:                                              ; preds = %166, %167, %172
  %175 = load ptr, ptr %29, align 8
  store ptr %175, ptr %15, align 8
  %176 = load ptr, ptr %15, align 8
  store ptr %176, ptr %10, align 8
  %177 = load ptr, ptr %10, align 8
  %178 = load i32, ptr %177, align 8
  %179 = icmp slt i32 %178, 0
  %180 = zext i1 %179 to i32
  %181 = icmp ne i32 %180, 0
  br i1 %181, label %182, label %183

182:                                              ; preds = %174
  br label %190

183:                                              ; preds = %174
  %184 = load ptr, ptr %15, align 8
  %185 = load i32, ptr %184, align 8
  %186 = add i32 %185, -1
  store i32 %186, ptr %184, align 8
  %187 = icmp eq i32 %186, 0
  br i1 %187, label %188, label %190

188:                                              ; preds = %183
  %189 = load ptr, ptr %15, align 8
  call void @_Py_Dealloc(ptr noundef %189) #7
  br label %190

190:                                              ; preds = %182, %183, %188
  %191 = load ptr, ptr %28, align 8
  store ptr %191, ptr %30, align 8
  %192 = load ptr, ptr %27, align 8
  store ptr %192, ptr %29, align 8
  br label %226

193:                                              ; preds = %155
  %194 = load ptr, ptr %27, align 8
  store ptr %194, ptr %16, align 8
  %195 = load ptr, ptr %16, align 8
  store ptr %195, ptr %9, align 8
  %196 = load ptr, ptr %9, align 8
  %197 = load i32, ptr %196, align 8
  %198 = icmp slt i32 %197, 0
  %199 = zext i1 %198 to i32
  %200 = icmp ne i32 %199, 0
  br i1 %200, label %201, label %202

201:                                              ; preds = %193
  br label %209

202:                                              ; preds = %193
  %203 = load ptr, ptr %16, align 8
  %204 = load i32, ptr %203, align 8
  %205 = add i32 %204, -1
  store i32 %205, ptr %203, align 8
  %206 = icmp eq i32 %205, 0
  br i1 %206, label %207, label %209

207:                                              ; preds = %202
  %208 = load ptr, ptr %16, align 8
  call void @_Py_Dealloc(ptr noundef %208) #7
  br label %209

209:                                              ; preds = %201, %202, %207
  %210 = load ptr, ptr %28, align 8
  store ptr %210, ptr %17, align 8
  %211 = load ptr, ptr %17, align 8
  store ptr %211, ptr %8, align 8
  %212 = load ptr, ptr %8, align 8
  %213 = load i32, ptr %212, align 8
  %214 = icmp slt i32 %213, 0
  %215 = zext i1 %214 to i32
  %216 = icmp ne i32 %215, 0
  br i1 %216, label %217, label %218

217:                                              ; preds = %209
  br label %225

218:                                              ; preds = %209
  %219 = load ptr, ptr %17, align 8
  %220 = load i32, ptr %219, align 8
  %221 = add i32 %220, -1
  store i32 %221, ptr %219, align 8
  %222 = icmp eq i32 %221, 0
  br i1 %222, label %223, label %225

223:                                              ; preds = %218
  %224 = load ptr, ptr %17, align 8
  call void @_Py_Dealloc(ptr noundef %224) #7
  br label %225

225:                                              ; preds = %217, %218, %223
  br label %226

226:                                              ; preds = %225, %190
  br label %227

227:                                              ; preds = %226
  br label %228

228:                                              ; preds = %227, %144
  br label %93

229:                                              ; preds = %125, %100
  %230 = load ptr, ptr %30, align 8
  %231 = icmp eq ptr %230, null
  br i1 %231, label %232, label %253

232:                                              ; preds = %229
  %233 = load ptr, ptr %29, align 8
  %234 = icmp eq ptr %233, null
  %235 = xor i1 %234, true
  %236 = zext i1 %235 to i32
  %237 = sext i32 %236 to i64
  %238 = icmp ne i64 %237, 0
  br i1 %238, label %239, label %241

239:                                              ; preds = %232
  call void @__assert_rtn(ptr noundef @__func__.min_max, ptr noundef @.str.34, i32 noundef 1980, ptr noundef @.str.193) #8
  unreachable

240:                                              ; No predecessors!
  br label %242

241:                                              ; preds = %232
  br label %242

242:                                              ; preds = %241, %240
  %243 = load ptr, ptr %32, align 8
  %244 = icmp ne ptr %243, null
  br i1 %244, label %245, label %248

245:                                              ; preds = %242
  %246 = load ptr, ptr %32, align 8
  %247 = call ptr @_Py_NewRef(ptr noundef %246)
  store ptr %247, ptr %29, align 8
  br label %252

248:                                              ; preds = %242
  %249 = load ptr, ptr @PyExc_ValueError, align 8
  %250 = load ptr, ptr %33, align 8
  %251 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %249, ptr noundef @.str.194, ptr noundef %250)
  br label %252

252:                                              ; preds = %248, %245
  br label %270

253:                                              ; preds = %229
  %254 = load ptr, ptr %30, align 8
  store ptr %254, ptr %18, align 8
  %255 = load ptr, ptr %18, align 8
  store ptr %255, ptr %7, align 8
  %256 = load ptr, ptr %7, align 8
  %257 = load i32, ptr %256, align 8
  %258 = icmp slt i32 %257, 0
  %259 = zext i1 %258 to i32
  %260 = icmp ne i32 %259, 0
  br i1 %260, label %261, label %262

261:                                              ; preds = %253
  br label %269

262:                                              ; preds = %253
  %263 = load ptr, ptr %18, align 8
  %264 = load i32, ptr %263, align 8
  %265 = add i32 %264, -1
  store i32 %265, ptr %263, align 8
  %266 = icmp eq i32 %265, 0
  br i1 %266, label %267, label %269

267:                                              ; preds = %262
  %268 = load ptr, ptr %18, align 8
  call void @_Py_Dealloc(ptr noundef %268) #7
  br label %269

269:                                              ; preds = %261, %262, %267
  br label %270

270:                                              ; preds = %269, %252
  %271 = load ptr, ptr %26, align 8
  call void @Py_XDECREF(ptr noundef %271)
  %272 = load ptr, ptr %29, align 8
  store ptr %272, ptr %21, align 8
  br label %311

273:                                              ; preds = %154
  %274 = load ptr, ptr %28, align 8
  store ptr %274, ptr %19, align 8
  %275 = load ptr, ptr %19, align 8
  store ptr %275, ptr %6, align 8
  %276 = load ptr, ptr %6, align 8
  %277 = load i32, ptr %276, align 8
  %278 = icmp slt i32 %277, 0
  %279 = zext i1 %278 to i32
  %280 = icmp ne i32 %279, 0
  br i1 %280, label %281, label %282

281:                                              ; preds = %273
  br label %289

282:                                              ; preds = %273
  %283 = load ptr, ptr %19, align 8
  %284 = load i32, ptr %283, align 8
  %285 = add i32 %284, -1
  store i32 %285, ptr %283, align 8
  %286 = icmp eq i32 %285, 0
  br i1 %286, label %287, label %289

287:                                              ; preds = %282
  %288 = load ptr, ptr %19, align 8
  call void @_Py_Dealloc(ptr noundef %288) #7
  br label %289

289:                                              ; preds = %281, %282, %287
  br label %290

290:                                              ; preds = %289, %136
  %291 = load ptr, ptr %27, align 8
  store ptr %291, ptr %20, align 8
  %292 = load ptr, ptr %20, align 8
  store ptr %292, ptr %5, align 8
  %293 = load ptr, ptr %5, align 8
  %294 = load i32, ptr %293, align 8
  %295 = icmp slt i32 %294, 0
  %296 = zext i1 %295 to i32
  %297 = icmp ne i32 %296, 0
  br i1 %297, label %298, label %299

298:                                              ; preds = %290
  br label %306

299:                                              ; preds = %290
  %300 = load ptr, ptr %20, align 8
  %301 = load i32, ptr %300, align 8
  %302 = add i32 %301, -1
  store i32 %302, ptr %300, align 8
  %303 = icmp eq i32 %302, 0
  br i1 %303, label %304, label %306

304:                                              ; preds = %299
  %305 = load ptr, ptr %20, align 8
  call void @_Py_Dealloc(ptr noundef %305) #7
  br label %306

306:                                              ; preds = %298, %299, %304
  br label %307

307:                                              ; preds = %306, %124
  %308 = load ptr, ptr %30, align 8
  call void @Py_XDECREF(ptr noundef %308)
  %309 = load ptr, ptr %29, align 8
  call void @Py_XDECREF(ptr noundef %309)
  %310 = load ptr, ptr %26, align 8
  call void @Py_XDECREF(ptr noundef %310)
  store ptr null, ptr %21, align 8
  br label %311

311:                                              ; preds = %307, %270, %86, %72, %62, %47
  %312 = load ptr, ptr %21, align 8
  ret ptr %312
}

declare i32 @_PyArg_ParseStackAndKeywords(ptr noundef, i64 noundef, ptr noundef, ptr noundef, ...) #1

declare ptr @PyIter_Next(ptr noundef) #1

declare i32 @PyObject_RichCompareBool(ptr noundef, ptr noundef, i32 noundef) #1

declare i32 @PyIter_Check(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_anext_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  %13 = load ptr, ptr %8, align 8
  %14 = call ptr @_Py_TYPE(ptr noundef %13)
  store ptr %14, ptr %10, align 8
  %15 = load ptr, ptr %10, align 8
  %16 = getelementptr inbounds %struct._typeobject, ptr %15, i32 0, i32 8
  %17 = load ptr, ptr %16, align 8
  %18 = icmp eq ptr %17, null
  br i1 %18, label %26, label %19

19:                                               ; preds = %3
  %20 = load ptr, ptr %10, align 8
  %21 = getelementptr inbounds %struct._typeobject, ptr %20, i32 0, i32 8
  %22 = load ptr, ptr %21, align 8
  %23 = getelementptr inbounds %struct.PyAsyncMethods, ptr %22, i32 0, i32 2
  %24 = load ptr, ptr %23, align 8
  %25 = icmp eq ptr %24, null
  br i1 %25, label %26, label %32

26:                                               ; preds = %19, %3
  %27 = load ptr, ptr @PyExc_TypeError, align 8
  %28 = load ptr, ptr %10, align 8
  %29 = getelementptr inbounds %struct._typeobject, ptr %28, i32 0, i32 1
  %30 = load ptr, ptr %29, align 8
  %31 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %27, ptr noundef @.str.196, ptr noundef %30)
  store ptr null, ptr %6, align 8
  br label %65

32:                                               ; preds = %19
  %33 = load ptr, ptr %10, align 8
  %34 = getelementptr inbounds %struct._typeobject, ptr %33, i32 0, i32 8
  %35 = load ptr, ptr %34, align 8
  %36 = getelementptr inbounds %struct.PyAsyncMethods, ptr %35, i32 0, i32 2
  %37 = load ptr, ptr %36, align 8
  %38 = load ptr, ptr %8, align 8
  %39 = call ptr %37(ptr noundef %38)
  store ptr %39, ptr %11, align 8
  %40 = load ptr, ptr %9, align 8
  %41 = icmp eq ptr %40, null
  br i1 %41, label %42, label %44

42:                                               ; preds = %32
  %43 = load ptr, ptr %11, align 8
  store ptr %43, ptr %6, align 8
  br label %65

44:                                               ; preds = %32
  %45 = load ptr, ptr %11, align 8
  %46 = load ptr, ptr %9, align 8
  %47 = call ptr @PyAnextAwaitable_New(ptr noundef %45, ptr noundef %46)
  store ptr %47, ptr %12, align 8
  %48 = load ptr, ptr %11, align 8
  store ptr %48, ptr %5, align 8
  %49 = load ptr, ptr %5, align 8
  store ptr %49, ptr %4, align 8
  %50 = load ptr, ptr %4, align 8
  %51 = load i32, ptr %50, align 8
  %52 = icmp slt i32 %51, 0
  %53 = zext i1 %52 to i32
  %54 = icmp ne i32 %53, 0
  br i1 %54, label %55, label %56

55:                                               ; preds = %44
  br label %63

56:                                               ; preds = %44
  %57 = load ptr, ptr %5, align 8
  %58 = load i32, ptr %57, align 8
  %59 = add i32 %58, -1
  store i32 %59, ptr %57, align 8
  %60 = icmp eq i32 %59, 0
  br i1 %60, label %61, label %63

61:                                               ; preds = %56
  %62 = load ptr, ptr %5, align 8
  call void @_Py_Dealloc(ptr noundef %62) #7
  br label %63

63:                                               ; preds = %55, %56, %61
  %64 = load ptr, ptr %12, align 8
  store ptr %64, ptr %6, align 8
  br label %65

65:                                               ; preds = %63, %42, %26
  %66 = load ptr, ptr %6, align 8
  ret ptr %66
}

declare ptr @PyAnextAwaitable_New(ptr noundef, ptr noundef) #1

declare ptr @PyLong_FromLong(i64 noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @PyUnicode_GET_LENGTH(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_Py_TYPE(ptr noundef %3)
  %5 = call i32 @PyType_HasFeature(ptr noundef %4, i64 noundef 268435456)
  %6 = icmp ne i32 %5, 0
  %7 = xor i1 %6, true
  %8 = zext i1 %7 to i32
  %9 = sext i32 %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyUnicode_GET_LENGTH, ptr noundef @.str.199, i32 noundef 286, ptr noundef @.str.200) #8
  unreachable

12:                                               ; No predecessors!
  br label %14

13:                                               ; preds = %1
  br label %14

14:                                               ; preds = %13, %12
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds %struct.PyASCIIObject, ptr %15, i32 0, i32 1
  %17 = load i64, ptr %16, align 8
  ret i64 %17
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @PyUnicode_READ_CHAR(ptr noundef %0, i64 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  %6 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store i64 %1, ptr %5, align 8
  %7 = load i64, ptr %5, align 8
  %8 = icmp sge i64 %7, 0
  %9 = xor i1 %8, true
  %10 = zext i1 %9 to i32
  %11 = sext i32 %10 to i64
  %12 = icmp ne i64 %11, 0
  br i1 %12, label %13, label %15

13:                                               ; preds = %2
  call void @__assert_rtn(ptr noundef @__func__.PyUnicode_READ_CHAR, ptr noundef @.str.199, i32 noundef 345, ptr noundef @.str.201) #8
  unreachable

14:                                               ; No predecessors!
  br label %16

15:                                               ; preds = %2
  br label %16

16:                                               ; preds = %15, %14
  %17 = load i64, ptr %5, align 8
  %18 = load ptr, ptr %4, align 8
  %19 = call i64 @PyUnicode_GET_LENGTH(ptr noundef %18)
  %20 = icmp sle i64 %17, %19
  %21 = xor i1 %20, true
  %22 = zext i1 %21 to i32
  %23 = sext i32 %22 to i64
  %24 = icmp ne i64 %23, 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %16
  call void @__assert_rtn(ptr noundef @__func__.PyUnicode_READ_CHAR, ptr noundef @.str.199, i32 noundef 347, ptr noundef @.str.202) #8
  unreachable

26:                                               ; No predecessors!
  br label %28

27:                                               ; preds = %16
  br label %28

28:                                               ; preds = %27, %26
  %29 = load ptr, ptr %4, align 8
  %30 = call ptr @_Py_TYPE(ptr noundef %29)
  %31 = call i32 @PyType_HasFeature(ptr noundef %30, i64 noundef 268435456)
  %32 = icmp ne i32 %31, 0
  %33 = xor i1 %32, true
  %34 = zext i1 %33 to i32
  %35 = sext i32 %34 to i64
  %36 = icmp ne i64 %35, 0
  br i1 %36, label %37, label %39

37:                                               ; preds = %28
  call void @__assert_rtn(ptr noundef @__func__.PyUnicode_READ_CHAR, ptr noundef @.str.199, i32 noundef 349, ptr noundef @.str.203) #8
  unreachable

38:                                               ; No predecessors!
  br label %40

39:                                               ; preds = %28
  br label %40

40:                                               ; preds = %39, %38
  %41 = load ptr, ptr %4, align 8
  %42 = getelementptr inbounds %struct.PyASCIIObject, ptr %41, i32 0, i32 3
  %43 = getelementptr inbounds %struct.anon.51, ptr %42, i32 0, i32 1
  %44 = load i16, ptr %43, align 2
  %45 = and i16 %44, 7
  %46 = zext i16 %45 to i32
  store i32 %46, ptr %6, align 4
  %47 = load i32, ptr %6, align 4
  %48 = icmp eq i32 %47, 1
  br i1 %48, label %49, label %56

49:                                               ; preds = %40
  %50 = load ptr, ptr %4, align 8
  %51 = call ptr @PyUnicode_DATA(ptr noundef %50)
  %52 = load i64, ptr %5, align 8
  %53 = getelementptr inbounds i8, ptr %51, i64 %52
  %54 = load i8, ptr %53, align 1
  %55 = zext i8 %54 to i32
  store i32 %55, ptr %3, align 4
  br label %82

56:                                               ; preds = %40
  %57 = load i32, ptr %6, align 4
  %58 = icmp eq i32 %57, 2
  br i1 %58, label %59, label %66

59:                                               ; preds = %56
  %60 = load ptr, ptr %4, align 8
  %61 = call ptr @PyUnicode_DATA(ptr noundef %60)
  %62 = load i64, ptr %5, align 8
  %63 = getelementptr inbounds i16, ptr %61, i64 %62
  %64 = load i16, ptr %63, align 2
  %65 = zext i16 %64 to i32
  store i32 %65, ptr %3, align 4
  br label %82

66:                                               ; preds = %56
  %67 = load i32, ptr %6, align 4
  %68 = icmp eq i32 %67, 4
  %69 = xor i1 %68, true
  %70 = zext i1 %69 to i32
  %71 = sext i32 %70 to i64
  %72 = icmp ne i64 %71, 0
  br i1 %72, label %73, label %75

73:                                               ; preds = %66
  call void @__assert_rtn(ptr noundef @__func__.PyUnicode_READ_CHAR, ptr noundef @.str.199, i32 noundef 356, ptr noundef @.str.204) #8
  unreachable

74:                                               ; No predecessors!
  br label %76

75:                                               ; preds = %66
  br label %76

76:                                               ; preds = %75, %74
  %77 = load ptr, ptr %4, align 8
  %78 = call ptr @PyUnicode_DATA(ptr noundef %77)
  %79 = load i64, ptr %5, align 8
  %80 = getelementptr inbounds i32, ptr %78, i64 %79
  %81 = load i32, ptr %80, align 4
  store i32 %81, ptr %3, align 4
  br label %82

82:                                               ; preds = %76, %59, %49
  %83 = load i32, ptr %3, align 4
  ret i32 %83
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @PyObject_TypeCheck(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call i32 @Py_IS_TYPE(ptr noundef %5, ptr noundef %6)
  %8 = icmp ne i32 %7, 0
  br i1 %8, label %15, label %9

9:                                                ; preds = %2
  %10 = load ptr, ptr %3, align 8
  %11 = call ptr @_Py_TYPE(ptr noundef %10)
  %12 = load ptr, ptr %4, align 8
  %13 = call i32 @PyType_IsSubtype(ptr noundef %11, ptr noundef %12)
  %14 = icmp ne i32 %13, 0
  br label %15

15:                                               ; preds = %9, %2
  %16 = phi i1 [ true, %2 ], [ %14, %9 ]
  %17 = zext i1 %16 to i32
  ret i32 %17
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @PyByteArray_GET_SIZE(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call i32 @PyObject_TypeCheck(ptr noundef %4, ptr noundef @PyByteArray_Type)
  %6 = icmp ne i32 %5, 0
  %7 = xor i1 %6, true
  %8 = zext i1 %7 to i32
  %9 = sext i32 %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyByteArray_GET_SIZE, ptr noundef @.str.207, i32 noundef 31, ptr noundef @.str.208) #8
  unreachable

12:                                               ; No predecessors!
  br label %14

13:                                               ; preds = %1
  br label %14

14:                                               ; preds = %13, %12
  %15 = load ptr, ptr %2, align 8
  store ptr %15, ptr %3, align 8
  %16 = load ptr, ptr %3, align 8
  %17 = call i64 @Py_SIZE(ptr noundef %16)
  ret i64 %17
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @PyByteArray_AS_STRING(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = call i32 @PyObject_TypeCheck(ptr noundef %5, ptr noundef @PyByteArray_Type)
  %7 = icmp ne i32 %6, 0
  %8 = xor i1 %7, true
  %9 = zext i1 %8 to i32
  %10 = sext i32 %9 to i64
  %11 = icmp ne i64 %10, 0
  br i1 %11, label %12, label %14

12:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyByteArray_AS_STRING, ptr noundef @.str.207, i32 noundef 22, ptr noundef @.str.208) #8
  unreachable

13:                                               ; No predecessors!
  br label %15

14:                                               ; preds = %1
  br label %15

15:                                               ; preds = %14, %13
  %16 = load ptr, ptr %3, align 8
  store ptr %16, ptr %4, align 8
  %17 = load ptr, ptr %4, align 8
  %18 = call i64 @Py_SIZE(ptr noundef %17)
  %19 = icmp ne i64 %18, 0
  br i1 %19, label %20, label %24

20:                                               ; preds = %15
  %21 = load ptr, ptr %4, align 8
  %22 = getelementptr inbounds %struct.PyByteArrayObject, ptr %21, i32 0, i32 3
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %2, align 8
  br label %25

24:                                               ; preds = %15
  store ptr @_PyByteArray_empty_string, ptr %2, align 8
  br label %25

25:                                               ; preds = %24, %20
  %26 = load ptr, ptr %2, align 8
  ret ptr %26
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @PyUnicode_DATA(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call i32 @PyUnicode_IS_COMPACT(ptr noundef %4)
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %1
  %8 = load ptr, ptr %3, align 8
  %9 = call ptr @_PyUnicode_COMPACT_DATA(ptr noundef %8)
  store ptr %9, ptr %2, align 8
  br label %13

10:                                               ; preds = %1
  %11 = load ptr, ptr %3, align 8
  %12 = call ptr @_PyUnicode_NONCOMPACT_DATA(ptr noundef %11)
  store ptr %12, ptr %2, align 8
  br label %13

13:                                               ; preds = %10, %7
  %14 = load ptr, ptr %2, align 8
  ret ptr %14
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @PyUnicode_IS_COMPACT(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_Py_TYPE(ptr noundef %3)
  %5 = call i32 @PyType_HasFeature(ptr noundef %4, i64 noundef 268435456)
  %6 = icmp ne i32 %5, 0
  %7 = xor i1 %6, true
  %8 = zext i1 %7 to i32
  %9 = sext i32 %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyUnicode_IS_COMPACT, ptr noundef @.str.199, i32 noundef 225, ptr noundef @.str.200) #8
  unreachable

12:                                               ; No predecessors!
  br label %14

13:                                               ; preds = %1
  br label %14

14:                                               ; preds = %13, %12
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds %struct.PyASCIIObject, ptr %15, i32 0, i32 3
  %17 = getelementptr inbounds %struct.anon.51, ptr %16, i32 0, i32 1
  %18 = load i16, ptr %17, align 2
  %19 = lshr i16 %18, 3
  %20 = and i16 %19, 1
  %21 = zext i16 %20 to i32
  ret i32 %21
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_PyUnicode_COMPACT_DATA(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call i32 @PyUnicode_IS_ASCII(ptr noundef %4)
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %22

7:                                                ; preds = %1
  %8 = load ptr, ptr %3, align 8
  %9 = call ptr @_Py_TYPE(ptr noundef %8)
  %10 = call i32 @PyType_HasFeature(ptr noundef %9, i64 noundef 268435456)
  %11 = icmp ne i32 %10, 0
  %12 = xor i1 %11, true
  %13 = zext i1 %12 to i32
  %14 = sext i32 %13 to i64
  %15 = icmp ne i64 %14, 0
  br i1 %15, label %16, label %18

16:                                               ; preds = %7
  call void @__assert_rtn(ptr noundef @__func__._PyUnicode_COMPACT_DATA, ptr noundef @.str.199, i32 noundef 254, ptr noundef @.str.200) #8
  unreachable

17:                                               ; No predecessors!
  br label %19

18:                                               ; preds = %7
  br label %19

19:                                               ; preds = %18, %17
  %20 = load ptr, ptr %3, align 8
  %21 = getelementptr inbounds %struct.PyASCIIObject, ptr %20, i64 1
  store ptr %21, ptr %2, align 8
  br label %37

22:                                               ; preds = %1
  %23 = load ptr, ptr %3, align 8
  %24 = call ptr @_Py_TYPE(ptr noundef %23)
  %25 = call i32 @PyType_HasFeature(ptr noundef %24, i64 noundef 268435456)
  %26 = icmp ne i32 %25, 0
  %27 = xor i1 %26, true
  %28 = zext i1 %27 to i32
  %29 = sext i32 %28 to i64
  %30 = icmp ne i64 %29, 0
  br i1 %30, label %31, label %33

31:                                               ; preds = %22
  call void @__assert_rtn(ptr noundef @__func__._PyUnicode_COMPACT_DATA, ptr noundef @.str.199, i32 noundef 256, ptr noundef @.str.200) #8
  unreachable

32:                                               ; No predecessors!
  br label %34

33:                                               ; preds = %22
  br label %34

34:                                               ; preds = %33, %32
  %35 = load ptr, ptr %3, align 8
  %36 = getelementptr inbounds %struct.PyCompactUnicodeObject, ptr %35, i64 1
  store ptr %36, ptr %2, align 8
  br label %37

37:                                               ; preds = %34, %19
  %38 = load ptr, ptr %2, align 8
  ret ptr %38
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_PyUnicode_NONCOMPACT_DATA(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call i32 @PyUnicode_IS_COMPACT(ptr noundef %4)
  %6 = icmp ne i32 %5, 0
  %7 = xor i1 %6, true
  %8 = xor i1 %7, true
  %9 = zext i1 %8 to i32
  %10 = sext i32 %9 to i64
  %11 = icmp ne i64 %10, 0
  br i1 %11, label %12, label %14

12:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__._PyUnicode_NONCOMPACT_DATA, ptr noundef @.str.199, i32 noundef 261, ptr noundef @.str.205) #8
  unreachable

13:                                               ; No predecessors!
  br label %15

14:                                               ; preds = %1
  br label %15

15:                                               ; preds = %14, %13
  %16 = load ptr, ptr %2, align 8
  %17 = call ptr @_Py_TYPE(ptr noundef %16)
  %18 = call i32 @PyType_HasFeature(ptr noundef %17, i64 noundef 268435456)
  %19 = icmp ne i32 %18, 0
  %20 = xor i1 %19, true
  %21 = zext i1 %20 to i32
  %22 = sext i32 %21 to i64
  %23 = icmp ne i64 %22, 0
  br i1 %23, label %24, label %26

24:                                               ; preds = %15
  call void @__assert_rtn(ptr noundef @__func__._PyUnicode_NONCOMPACT_DATA, ptr noundef @.str.199, i32 noundef 262, ptr noundef @.str.200) #8
  unreachable

25:                                               ; No predecessors!
  br label %27

26:                                               ; preds = %15
  br label %27

27:                                               ; preds = %26, %25
  %28 = load ptr, ptr %2, align 8
  %29 = getelementptr inbounds %struct.PyUnicodeObject, ptr %28, i32 0, i32 1
  %30 = load ptr, ptr %29, align 8
  store ptr %30, ptr %3, align 8
  %31 = load ptr, ptr %3, align 8
  %32 = icmp ne ptr %31, null
  %33 = xor i1 %32, true
  %34 = zext i1 %33 to i32
  %35 = sext i32 %34 to i64
  %36 = icmp ne i64 %35, 0
  br i1 %36, label %37, label %39

37:                                               ; preds = %27
  call void @__assert_rtn(ptr noundef @__func__._PyUnicode_NONCOMPACT_DATA, ptr noundef @.str.199, i32 noundef 263, ptr noundef @.str.206) #8
  unreachable

38:                                               ; No predecessors!
  br label %40

39:                                               ; preds = %27
  br label %40

40:                                               ; preds = %39, %38
  %41 = load ptr, ptr %3, align 8
  ret ptr %41
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @PyUnicode_IS_ASCII(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_Py_TYPE(ptr noundef %3)
  %5 = call i32 @PyType_HasFeature(ptr noundef %4, i64 noundef 268435456)
  %6 = icmp ne i32 %5, 0
  %7 = xor i1 %6, true
  %8 = zext i1 %7 to i32
  %9 = sext i32 %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyUnicode_IS_ASCII, ptr noundef @.str.199, i32 noundef 218, ptr noundef @.str.200) #8
  unreachable

12:                                               ; No predecessors!
  br label %14

13:                                               ; preds = %1
  br label %14

14:                                               ; preds = %13, %12
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds %struct.PyASCIIObject, ptr %15, i32 0, i32 3
  %17 = getelementptr inbounds %struct.anon.51, ptr %16, i32 0, i32 1
  %18 = load i16, ptr %17, align 2
  %19 = lshr i16 %18, 4
  %20 = and i16 %19, 1
  %21 = zext i16 %20 to i32
  ret i32 %21
}

declare i32 @PyType_IsSubtype(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_pow_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  %9 = load ptr, ptr %6, align 8
  %10 = load ptr, ptr %7, align 8
  %11 = load ptr, ptr %8, align 8
  %12 = call ptr @PyNumber_Power(ptr noundef %9, ptr noundef %10, ptr noundef %11)
  ret ptr %12
}

declare ptr @PyNumber_Power(ptr noundef, ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_print_impl(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr noundef %3, ptr noundef %4, ptr noundef %5, i32 noundef %6) #0 {
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca ptr, align 8
  store ptr %0, ptr %9, align 8
  store ptr %1, ptr %10, align 8
  store i64 %2, ptr %11, align 8
  store ptr %3, ptr %12, align 8
  store ptr %4, ptr %13, align 8
  store ptr %5, ptr %14, align 8
  store i32 %6, ptr %15, align 4
  %19 = load ptr, ptr %14, align 8
  %20 = icmp eq ptr %19, @_Py_NoneStruct
  br i1 %20, label %21, label %34

21:                                               ; preds = %7
  %22 = call ptr @_PyThreadState_GET()
  store ptr %22, ptr %18, align 8
  %23 = load ptr, ptr %18, align 8
  %24 = call ptr @_PySys_GetAttr(ptr noundef %23, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 658))
  store ptr %24, ptr %14, align 8
  %25 = load ptr, ptr %14, align 8
  %26 = icmp eq ptr %25, null
  br i1 %26, label %27, label %29

27:                                               ; preds = %21
  %28 = load ptr, ptr @PyExc_RuntimeError, align 8
  call void @PyErr_SetString(ptr noundef %28, ptr noundef @.str.216)
  store ptr null, ptr %8, align 8
  br label %137

29:                                               ; preds = %21
  %30 = load ptr, ptr %14, align 8
  %31 = icmp eq ptr %30, @_Py_NoneStruct
  br i1 %31, label %32, label %33

32:                                               ; preds = %29
  store ptr @_Py_NoneStruct, ptr %8, align 8
  br label %137

33:                                               ; preds = %29
  br label %34

34:                                               ; preds = %33, %7
  %35 = load ptr, ptr %12, align 8
  %36 = icmp eq ptr %35, @_Py_NoneStruct
  br i1 %36, label %37, label %38

37:                                               ; preds = %34
  store ptr null, ptr %12, align 8
  br label %54

38:                                               ; preds = %34
  %39 = load ptr, ptr %12, align 8
  %40 = icmp ne ptr %39, null
  br i1 %40, label %41, label %53

41:                                               ; preds = %38
  %42 = load ptr, ptr %12, align 8
  %43 = call ptr @_Py_TYPE(ptr noundef %42)
  %44 = call i32 @PyType_HasFeature(ptr noundef %43, i64 noundef 268435456)
  %45 = icmp ne i32 %44, 0
  br i1 %45, label %53, label %46

46:                                               ; preds = %41
  %47 = load ptr, ptr @PyExc_TypeError, align 8
  %48 = load ptr, ptr %12, align 8
  %49 = call ptr @_Py_TYPE(ptr noundef %48)
  %50 = getelementptr inbounds %struct._typeobject, ptr %49, i32 0, i32 1
  %51 = load ptr, ptr %50, align 8
  %52 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %47, ptr noundef @.str.217, ptr noundef %51)
  store ptr null, ptr %8, align 8
  br label %137

53:                                               ; preds = %41, %38
  br label %54

54:                                               ; preds = %53, %37
  %55 = load ptr, ptr %13, align 8
  %56 = icmp eq ptr %55, @_Py_NoneStruct
  br i1 %56, label %57, label %58

57:                                               ; preds = %54
  store ptr null, ptr %13, align 8
  br label %74

58:                                               ; preds = %54
  %59 = load ptr, ptr %13, align 8
  %60 = icmp ne ptr %59, null
  br i1 %60, label %61, label %73

61:                                               ; preds = %58
  %62 = load ptr, ptr %13, align 8
  %63 = call ptr @_Py_TYPE(ptr noundef %62)
  %64 = call i32 @PyType_HasFeature(ptr noundef %63, i64 noundef 268435456)
  %65 = icmp ne i32 %64, 0
  br i1 %65, label %73, label %66

66:                                               ; preds = %61
  %67 = load ptr, ptr @PyExc_TypeError, align 8
  %68 = load ptr, ptr %13, align 8
  %69 = call ptr @_Py_TYPE(ptr noundef %68)
  %70 = getelementptr inbounds %struct._typeobject, ptr %69, i32 0, i32 1
  %71 = load ptr, ptr %70, align 8
  %72 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %67, ptr noundef @.str.218, ptr noundef %71)
  store ptr null, ptr %8, align 8
  br label %137

73:                                               ; preds = %61, %58
  br label %74

74:                                               ; preds = %73, %57
  store i32 0, ptr %16, align 4
  br label %75

75:                                               ; preds = %110, %74
  %76 = load i32, ptr %16, align 4
  %77 = sext i32 %76 to i64
  %78 = load i64, ptr %11, align 8
  %79 = icmp slt i64 %77, %78
  br i1 %79, label %80, label %113

80:                                               ; preds = %75
  %81 = load i32, ptr %16, align 4
  %82 = icmp sgt i32 %81, 0
  br i1 %82, label %83, label %98

83:                                               ; preds = %80
  %84 = load ptr, ptr %12, align 8
  %85 = icmp eq ptr %84, null
  br i1 %85, label %86, label %89

86:                                               ; preds = %83
  %87 = load ptr, ptr %14, align 8
  %88 = call i32 @PyFile_WriteString(ptr noundef @.str.37, ptr noundef %87)
  store i32 %88, ptr %17, align 4
  br label %93

89:                                               ; preds = %83
  %90 = load ptr, ptr %12, align 8
  %91 = load ptr, ptr %14, align 8
  %92 = call i32 @PyFile_WriteObject(ptr noundef %90, ptr noundef %91, i32 noundef 1)
  store i32 %92, ptr %17, align 4
  br label %93

93:                                               ; preds = %89, %86
  %94 = load i32, ptr %17, align 4
  %95 = icmp ne i32 %94, 0
  br i1 %95, label %96, label %97

96:                                               ; preds = %93
  store ptr null, ptr %8, align 8
  br label %137

97:                                               ; preds = %93
  br label %98

98:                                               ; preds = %97, %80
  %99 = load ptr, ptr %10, align 8
  %100 = load i32, ptr %16, align 4
  %101 = sext i32 %100 to i64
  %102 = getelementptr inbounds ptr, ptr %99, i64 %101
  %103 = load ptr, ptr %102, align 8
  %104 = load ptr, ptr %14, align 8
  %105 = call i32 @PyFile_WriteObject(ptr noundef %103, ptr noundef %104, i32 noundef 1)
  store i32 %105, ptr %17, align 4
  %106 = load i32, ptr %17, align 4
  %107 = icmp ne i32 %106, 0
  br i1 %107, label %108, label %109

108:                                              ; preds = %98
  store ptr null, ptr %8, align 8
  br label %137

109:                                              ; preds = %98
  br label %110

110:                                              ; preds = %109
  %111 = load i32, ptr %16, align 4
  %112 = add nsw i32 %111, 1
  store i32 %112, ptr %16, align 4
  br label %75, !llvm.loop !22

113:                                              ; preds = %75
  %114 = load ptr, ptr %13, align 8
  %115 = icmp eq ptr %114, null
  br i1 %115, label %116, label %119

116:                                              ; preds = %113
  %117 = load ptr, ptr %14, align 8
  %118 = call i32 @PyFile_WriteString(ptr noundef @.str.219, ptr noundef %117)
  store i32 %118, ptr %17, align 4
  br label %123

119:                                              ; preds = %113
  %120 = load ptr, ptr %13, align 8
  %121 = load ptr, ptr %14, align 8
  %122 = call i32 @PyFile_WriteObject(ptr noundef %120, ptr noundef %121, i32 noundef 1)
  store i32 %122, ptr %17, align 4
  br label %123

123:                                              ; preds = %119, %116
  %124 = load i32, ptr %17, align 4
  %125 = icmp ne i32 %124, 0
  br i1 %125, label %126, label %127

126:                                              ; preds = %123
  store ptr null, ptr %8, align 8
  br label %137

127:                                              ; preds = %123
  %128 = load i32, ptr %15, align 4
  %129 = icmp ne i32 %128, 0
  br i1 %129, label %130, label %136

130:                                              ; preds = %127
  %131 = load ptr, ptr %14, align 8
  %132 = call i32 @_PyFile_Flush(ptr noundef %131)
  %133 = icmp slt i32 %132, 0
  br i1 %133, label %134, label %135

134:                                              ; preds = %130
  store ptr null, ptr %8, align 8
  br label %137

135:                                              ; preds = %130
  br label %136

136:                                              ; preds = %135, %127
  store ptr @_Py_NoneStruct, ptr %8, align 8
  br label %137

137:                                              ; preds = %136, %134, %126, %108, %96, %66, %46, %32, %27
  %138 = load ptr, ptr %8, align 8
  ret ptr %138
}

declare i32 @PyFile_WriteString(ptr noundef, ptr noundef) #1

declare ptr @PyObject_Repr(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_round_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  %12 = load ptr, ptr %8, align 8
  %13 = call ptr @_PyObject_LookupSpecial(ptr noundef %12, ptr noundef getelementptr inbounds (%struct.anon.74, ptr getelementptr inbounds (%struct._Py_global_strings, ptr getelementptr inbounds (%struct.anon.47, ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 41), i32 0, i32 3), i32 0, i32 1), i32 0, i32 139))
  store ptr %13, ptr %10, align 8
  %14 = load ptr, ptr %10, align 8
  %15 = icmp eq ptr %14, null
  br i1 %15, label %16, label %27

16:                                               ; preds = %3
  %17 = call ptr @PyErr_Occurred()
  %18 = icmp ne ptr %17, null
  br i1 %18, label %26, label %19

19:                                               ; preds = %16
  %20 = load ptr, ptr @PyExc_TypeError, align 8
  %21 = load ptr, ptr %8, align 8
  %22 = call ptr @_Py_TYPE(ptr noundef %21)
  %23 = getelementptr inbounds %struct._typeobject, ptr %22, i32 0, i32 1
  %24 = load ptr, ptr %23, align 8
  %25 = call ptr (ptr, ptr, ...) @PyErr_Format(ptr noundef %20, ptr noundef @.str.222, ptr noundef %24)
  br label %26

26:                                               ; preds = %19, %16
  store ptr null, ptr %6, align 8
  br label %55

27:                                               ; preds = %3
  %28 = load ptr, ptr %9, align 8
  %29 = icmp eq ptr %28, @_Py_NoneStruct
  br i1 %29, label %30, label %33

30:                                               ; preds = %27
  %31 = load ptr, ptr %10, align 8
  %32 = call ptr @_PyObject_CallNoArgs(ptr noundef %31)
  store ptr %32, ptr %11, align 8
  br label %37

33:                                               ; preds = %27
  %34 = load ptr, ptr %10, align 8
  %35 = load ptr, ptr %9, align 8
  %36 = call ptr @PyObject_CallOneArg(ptr noundef %34, ptr noundef %35)
  store ptr %36, ptr %11, align 8
  br label %37

37:                                               ; preds = %33, %30
  %38 = load ptr, ptr %10, align 8
  store ptr %38, ptr %5, align 8
  %39 = load ptr, ptr %5, align 8
  store ptr %39, ptr %4, align 8
  %40 = load ptr, ptr %4, align 8
  %41 = load i32, ptr %40, align 8
  %42 = icmp slt i32 %41, 0
  %43 = zext i1 %42 to i32
  %44 = icmp ne i32 %43, 0
  br i1 %44, label %45, label %46

45:                                               ; preds = %37
  br label %53

46:                                               ; preds = %37
  %47 = load ptr, ptr %5, align 8
  %48 = load i32, ptr %47, align 8
  %49 = add i32 %48, -1
  store i32 %49, ptr %47, align 8
  %50 = icmp eq i32 %49, 0
  br i1 %50, label %51, label %53

51:                                               ; preds = %46
  %52 = load ptr, ptr %5, align 8
  call void @_Py_Dealloc(ptr noundef %52) #7
  br label %53

53:                                               ; preds = %45, %46, %51
  %54 = load ptr, ptr %11, align 8
  store ptr %54, ptr %6, align 8
  br label %55

55:                                               ; preds = %53, %26
  %56 = load ptr, ptr %6, align 8
  ret ptr %56
}

declare ptr @_PyObject_LookupSpecial(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @_PyObject_CallNoArgs(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = call ptr @_PyThreadState_GET()
  store ptr %4, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %2, align 8
  %7 = call ptr @_PyObject_VectorcallTstate(ptr noundef %5, ptr noundef %6, ptr noundef null, i64 noundef 0, ptr noundef null)
  ret ptr %7
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_setattr_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store ptr %1, ptr %7, align 8
  store ptr %2, ptr %8, align 8
  store ptr %3, ptr %9, align 8
  %10 = load ptr, ptr %7, align 8
  %11 = load ptr, ptr %8, align 8
  %12 = load ptr, ptr %9, align 8
  %13 = call i32 @PyObject_SetAttr(ptr noundef %10, ptr noundef %11, ptr noundef %12)
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %16

15:                                               ; preds = %4
  store ptr null, ptr %5, align 8
  br label %17

16:                                               ; preds = %4
  store ptr @_Py_NoneStruct, ptr %5, align 8
  br label %17

17:                                               ; preds = %16, %15
  %18 = load ptr, ptr %5, align 8
  ret ptr %18
}

declare i32 @PyObject_SetAttr(ptr noundef, ptr noundef, ptr noundef) #1

declare i32 @_PyArg_UnpackStack(ptr noundef, i64 noundef, ptr noundef, i64 noundef, i64 noundef, ...) #1

declare ptr @PySequence_List(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal ptr @builtin_sum_impl(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca ptr, align 8
  %25 = alloca ptr, align 8
  %26 = alloca ptr, align 8
  %27 = alloca ptr, align 8
  %28 = alloca ptr, align 8
  %29 = alloca ptr, align 8
  %30 = alloca ptr, align 8
  %31 = alloca ptr, align 8
  %32 = alloca ptr, align 8
  %33 = alloca ptr, align 8
  %34 = alloca ptr, align 8
  %35 = alloca ptr, align 8
  %36 = alloca ptr, align 8
  %37 = alloca ptr, align 8
  %38 = alloca ptr, align 8
  %39 = alloca ptr, align 8
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca i32, align 4
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca ptr, align 8
  %46 = alloca ptr, align 8
  %47 = alloca ptr, align 8
  %48 = alloca ptr, align 8
  %49 = alloca ptr, align 8
  %50 = alloca ptr, align 8
  %51 = alloca ptr, align 8
  %52 = alloca ptr, align 8
  %53 = alloca ptr, align 8
  %54 = alloca ptr, align 8
  %55 = alloca ptr, align 8
  %56 = alloca ptr, align 8
  %57 = alloca ptr, align 8
  %58 = alloca ptr, align 8
  %59 = alloca ptr, align 8
  %60 = alloca ptr, align 8
  %61 = alloca ptr, align 8
  %62 = alloca ptr, align 8
  %63 = alloca ptr, align 8
  %64 = alloca ptr, align 8
  %65 = alloca ptr, align 8
  %66 = alloca ptr, align 8
  %67 = alloca ptr, align 8
  %68 = alloca ptr, align 8
  %69 = alloca ptr, align 8
  %70 = alloca ptr, align 8
  %71 = alloca ptr, align 8
  %72 = alloca ptr, align 8
  %73 = alloca ptr, align 8
  %74 = alloca ptr, align 8
  %75 = alloca ptr, align 8
  %76 = alloca ptr, align 8
  %77 = alloca ptr, align 8
  %78 = alloca ptr, align 8
  %79 = alloca ptr, align 8
  %80 = alloca ptr, align 8
  %81 = alloca ptr, align 8
  %82 = alloca ptr, align 8
  %83 = alloca ptr, align 8
  %84 = alloca ptr, align 8
  %85 = alloca ptr, align 8
  %86 = alloca ptr, align 8
  %87 = alloca ptr, align 8
  %88 = alloca i32, align 4
  %89 = alloca i64, align 8
  %90 = alloca ptr, align 8
  %91 = alloca ptr, align 8
  %92 = alloca i64, align 8
  %93 = alloca %struct.CompensatedSum, align 8
  %94 = alloca ptr, align 8
  %95 = alloca ptr, align 8
  %96 = alloca %struct.CompensatedSum, align 8
  %97 = alloca double, align 8
  %98 = alloca %struct.CompensatedSum, align 8
  %99 = alloca %struct.Py_complex, align 8
  %100 = alloca %struct.CompensatedSum, align 8
  %101 = alloca %struct.CompensatedSum, align 8
  %102 = alloca ptr, align 8
  %103 = alloca ptr, align 8
  %104 = alloca %struct.Py_complex, align 8
  %105 = alloca %struct.CompensatedSum, align 8
  %106 = alloca %struct.CompensatedSum, align 8
  %107 = alloca double, align 8
  %108 = alloca %struct.CompensatedSum, align 8
  %109 = alloca double, align 8
  %110 = alloca %struct.CompensatedSum, align 8
  %111 = alloca ptr, align 8
  %112 = alloca ptr, align 8
  store ptr %0, ptr %81, align 8
  store ptr %1, ptr %82, align 8
  store ptr %2, ptr %83, align 8
  %113 = load ptr, ptr %83, align 8
  store ptr %113, ptr %84, align 8
  %114 = load ptr, ptr %82, align 8
  %115 = call ptr @PyObject_GetIter(ptr noundef %114)
  store ptr %115, ptr %87, align 8
  %116 = load ptr, ptr %87, align 8
  %117 = icmp eq ptr %116, null
  br i1 %117, label %118, label %119

118:                                              ; preds = %3
  store ptr null, ptr %80, align 8
  br label %1082

119:                                              ; preds = %3
  %120 = load ptr, ptr %84, align 8
  %121 = icmp eq ptr %120, null
  br i1 %121, label %122, label %144

122:                                              ; preds = %119
  %123 = call ptr @PyLong_FromLong(i64 noundef 0)
  store ptr %123, ptr %84, align 8
  %124 = load ptr, ptr %84, align 8
  %125 = icmp eq ptr %124, null
  br i1 %125, label %126, label %143

126:                                              ; preds = %122
  %127 = load ptr, ptr %87, align 8
  store ptr %127, ptr %43, align 8
  %128 = load ptr, ptr %43, align 8
  store ptr %128, ptr %40, align 8
  %129 = load ptr, ptr %40, align 8
  %130 = load i32, ptr %129, align 8
  %131 = icmp slt i32 %130, 0
  %132 = zext i1 %131 to i32
  %133 = icmp ne i32 %132, 0
  br i1 %133, label %134, label %135

134:                                              ; preds = %126
  br label %142

135:                                              ; preds = %126
  %136 = load ptr, ptr %43, align 8
  %137 = load i32, ptr %136, align 8
  %138 = add i32 %137, -1
  store i32 %138, ptr %136, align 8
  %139 = icmp eq i32 %138, 0
  br i1 %139, label %140, label %142

140:                                              ; preds = %135
  %141 = load ptr, ptr %43, align 8
  call void @_Py_Dealloc(ptr noundef %141) #7
  br label %142

142:                                              ; preds = %134, %135, %140
  store ptr null, ptr %80, align 8
  br label %1082

143:                                              ; preds = %122
  br label %224

144:                                              ; preds = %119
  %145 = load ptr, ptr %84, align 8
  %146 = call ptr @_Py_TYPE(ptr noundef %145)
  %147 = call i32 @PyType_HasFeature(ptr noundef %146, i64 noundef 268435456)
  %148 = icmp ne i32 %147, 0
  br i1 %148, label %149, label %167

149:                                              ; preds = %144
  %150 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %150, ptr noundef @.str.225)
  %151 = load ptr, ptr %87, align 8
  store ptr %151, ptr %44, align 8
  %152 = load ptr, ptr %44, align 8
  store ptr %152, ptr %39, align 8
  %153 = load ptr, ptr %39, align 8
  %154 = load i32, ptr %153, align 8
  %155 = icmp slt i32 %154, 0
  %156 = zext i1 %155 to i32
  %157 = icmp ne i32 %156, 0
  br i1 %157, label %158, label %159

158:                                              ; preds = %149
  br label %166

159:                                              ; preds = %149
  %160 = load ptr, ptr %44, align 8
  %161 = load i32, ptr %160, align 8
  %162 = add i32 %161, -1
  store i32 %162, ptr %160, align 8
  %163 = icmp eq i32 %162, 0
  br i1 %163, label %164, label %166

164:                                              ; preds = %159
  %165 = load ptr, ptr %44, align 8
  call void @_Py_Dealloc(ptr noundef %165) #7
  br label %166

166:                                              ; preds = %158, %159, %164
  store ptr null, ptr %80, align 8
  br label %1082

167:                                              ; preds = %144
  %168 = load ptr, ptr %84, align 8
  %169 = call ptr @_Py_TYPE(ptr noundef %168)
  %170 = call i32 @PyType_HasFeature(ptr noundef %169, i64 noundef 134217728)
  %171 = icmp ne i32 %170, 0
  br i1 %171, label %172, label %190

172:                                              ; preds = %167
  %173 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %173, ptr noundef @.str.226)
  %174 = load ptr, ptr %87, align 8
  store ptr %174, ptr %45, align 8
  %175 = load ptr, ptr %45, align 8
  store ptr %175, ptr %38, align 8
  %176 = load ptr, ptr %38, align 8
  %177 = load i32, ptr %176, align 8
  %178 = icmp slt i32 %177, 0
  %179 = zext i1 %178 to i32
  %180 = icmp ne i32 %179, 0
  br i1 %180, label %181, label %182

181:                                              ; preds = %172
  br label %189

182:                                              ; preds = %172
  %183 = load ptr, ptr %45, align 8
  %184 = load i32, ptr %183, align 8
  %185 = add i32 %184, -1
  store i32 %185, ptr %183, align 8
  %186 = icmp eq i32 %185, 0
  br i1 %186, label %187, label %189

187:                                              ; preds = %182
  %188 = load ptr, ptr %45, align 8
  call void @_Py_Dealloc(ptr noundef %188) #7
  br label %189

189:                                              ; preds = %181, %182, %187
  store ptr null, ptr %80, align 8
  br label %1082

190:                                              ; preds = %167
  %191 = load ptr, ptr %84, align 8
  %192 = call i32 @PyObject_TypeCheck(ptr noundef %191, ptr noundef @PyByteArray_Type)
  %193 = icmp ne i32 %192, 0
  br i1 %193, label %194, label %212

194:                                              ; preds = %190
  %195 = load ptr, ptr @PyExc_TypeError, align 8
  call void @PyErr_SetString(ptr noundef %195, ptr noundef @.str.227)
  %196 = load ptr, ptr %87, align 8
  store ptr %196, ptr %46, align 8
  %197 = load ptr, ptr %46, align 8
  store ptr %197, ptr %37, align 8
  %198 = load ptr, ptr %37, align 8
  %199 = load i32, ptr %198, align 8
  %200 = icmp slt i32 %199, 0
  %201 = zext i1 %200 to i32
  %202 = icmp ne i32 %201, 0
  br i1 %202, label %203, label %204

203:                                              ; preds = %194
  br label %211

204:                                              ; preds = %194
  %205 = load ptr, ptr %46, align 8
  %206 = load i32, ptr %205, align 8
  %207 = add i32 %206, -1
  store i32 %207, ptr %205, align 8
  %208 = icmp eq i32 %207, 0
  br i1 %208, label %209, label %211

209:                                              ; preds = %204
  %210 = load ptr, ptr %46, align 8
  call void @_Py_Dealloc(ptr noundef %210) #7
  br label %211

211:                                              ; preds = %203, %204, %209
  store ptr null, ptr %80, align 8
  br label %1082

212:                                              ; preds = %190
  %213 = load ptr, ptr %84, align 8
  store ptr %213, ptr %41, align 8
  %214 = load ptr, ptr %41, align 8
  %215 = load i32, ptr %214, align 8
  store i32 %215, ptr %42, align 4
  %216 = load i32, ptr %42, align 4
  %217 = icmp slt i32 %216, 0
  br i1 %217, label %218, label %219

218:                                              ; preds = %212
  br label %223

219:                                              ; preds = %212
  %220 = load i32, ptr %42, align 4
  %221 = add i32 %220, 1
  %222 = load ptr, ptr %41, align 8
  store i32 %221, ptr %222, align 8
  br label %223

223:                                              ; preds = %218, %219
  br label %224

224:                                              ; preds = %223, %143
  %225 = load ptr, ptr %84, align 8
  %226 = call i32 @Py_IS_TYPE(ptr noundef %225, ptr noundef @PyLong_Type)
  %227 = icmp ne i32 %226, 0
  br i1 %227, label %228, label %438

228:                                              ; preds = %224
  %229 = load ptr, ptr %84, align 8
  %230 = call i64 @PyLong_AsLongAndOverflow(ptr noundef %229, ptr noundef %88)
  store i64 %230, ptr %89, align 8
  %231 = load i32, ptr %88, align 4
  %232 = icmp eq i32 %231, 0
  br i1 %232, label %233, label %255

233:                                              ; preds = %228
  br label %234

234:                                              ; preds = %233
  store ptr %84, ptr %90, align 8
  %235 = load ptr, ptr %90, align 8
  %236 = load ptr, ptr %235, align 8
  store ptr %236, ptr %91, align 8
  %237 = load ptr, ptr %90, align 8
  store ptr null, ptr %237, align 8
  %238 = load ptr, ptr %91, align 8
  store ptr %238, ptr %47, align 8
  %239 = load ptr, ptr %47, align 8
  store ptr %239, ptr %36, align 8
  %240 = load ptr, ptr %36, align 8
  %241 = load i32, ptr %240, align 8
  %242 = icmp slt i32 %241, 0
  %243 = zext i1 %242 to i32
  %244 = icmp ne i32 %243, 0
  br i1 %244, label %245, label %246

245:                                              ; preds = %234
  br label %253

246:                                              ; preds = %234
  %247 = load ptr, ptr %47, align 8
  %248 = load i32, ptr %247, align 8
  %249 = add i32 %248, -1
  store i32 %249, ptr %247, align 8
  %250 = icmp eq i32 %249, 0
  br i1 %250, label %251, label %253

251:                                              ; preds = %246
  %252 = load ptr, ptr %47, align 8
  call void @_Py_Dealloc(ptr noundef %252) #7
  br label %253

253:                                              ; preds = %245, %246, %251
  br label %254

254:                                              ; preds = %253
  br label %255

255:                                              ; preds = %254, %228
  br label %256

256:                                              ; preds = %436, %340, %255
  %257 = load ptr, ptr %84, align 8
  %258 = icmp eq ptr %257, null
  br i1 %258, label %259, label %437

259:                                              ; preds = %256
  %260 = load ptr, ptr %87, align 8
  %261 = call ptr @PyIter_Next(ptr noundef %260)
  store ptr %261, ptr %86, align 8
  %262 = load ptr, ptr %86, align 8
  %263 = icmp eq ptr %262, null
  br i1 %263, label %264, label %287

264:                                              ; preds = %259
  %265 = load ptr, ptr %87, align 8
  store ptr %265, ptr %48, align 8
  %266 = load ptr, ptr %48, align 8
  store ptr %266, ptr %35, align 8
  %267 = load ptr, ptr %35, align 8
  %268 = load i32, ptr %267, align 8
  %269 = icmp slt i32 %268, 0
  %270 = zext i1 %269 to i32
  %271 = icmp ne i32 %270, 0
  br i1 %271, label %272, label %273

272:                                              ; preds = %264
  br label %280

273:                                              ; preds = %264
  %274 = load ptr, ptr %48, align 8
  %275 = load i32, ptr %274, align 8
  %276 = add i32 %275, -1
  store i32 %276, ptr %274, align 8
  %277 = icmp eq i32 %276, 0
  br i1 %277, label %278, label %280

278:                                              ; preds = %273
  %279 = load ptr, ptr %48, align 8
  call void @_Py_Dealloc(ptr noundef %279) #7
  br label %280

280:                                              ; preds = %272, %273, %278
  %281 = call ptr @PyErr_Occurred()
  %282 = icmp ne ptr %281, null
  br i1 %282, label %283, label %284

283:                                              ; preds = %280
  store ptr null, ptr %80, align 8
  br label %1082

284:                                              ; preds = %280
  %285 = load i64, ptr %89, align 8
  %286 = call ptr @PyLong_FromSsize_t(i64 noundef %285)
  store ptr %286, ptr %80, align 8
  br label %1082

287:                                              ; preds = %259
  %288 = load ptr, ptr %86, align 8
  %289 = call i32 @Py_IS_TYPE(ptr noundef %288, ptr noundef @PyLong_Type)
  %290 = icmp ne i32 %289, 0
  br i1 %290, label %295, label %291

291:                                              ; preds = %287
  %292 = load ptr, ptr %86, align 8
  %293 = call i32 @Py_IS_TYPE(ptr noundef %292, ptr noundef @PyBool_Type)
  %294 = icmp ne i32 %293, 0
  br i1 %294, label %295, label %342

295:                                              ; preds = %291, %287
  store i32 0, ptr %88, align 4
  %296 = load ptr, ptr %86, align 8
  %297 = call i32 @_PyLong_IsCompact(ptr noundef %296)
  %298 = icmp ne i32 %297, 0
  br i1 %298, label %299, label %302

299:                                              ; preds = %295
  %300 = load ptr, ptr %86, align 8
  %301 = call i64 @_PyLong_CompactValue(ptr noundef %300)
  store i64 %301, ptr %92, align 8
  br label %305

302:                                              ; preds = %295
  %303 = load ptr, ptr %86, align 8
  %304 = call i64 @PyLong_AsLongAndOverflow(ptr noundef %303, ptr noundef %88)
  store i64 %304, ptr %92, align 8
  br label %305

305:                                              ; preds = %302, %299
  %306 = load i32, ptr %88, align 4
  %307 = icmp eq i32 %306, 0
  br i1 %307, label %308, label %341

308:                                              ; preds = %305
  %309 = load i64, ptr %89, align 8
  %310 = icmp sge i64 %309, 0
  br i1 %310, label %311, label %316

311:                                              ; preds = %308
  %312 = load i64, ptr %92, align 8
  %313 = load i64, ptr %89, align 8
  %314 = sub nsw i64 9223372036854775807, %313
  %315 = icmp sle i64 %312, %314
  br i1 %315, label %321, label %341

316:                                              ; preds = %308
  %317 = load i64, ptr %92, align 8
  %318 = load i64, ptr %89, align 8
  %319 = sub nsw i64 -9223372036854775808, %318
  %320 = icmp sge i64 %317, %319
  br i1 %320, label %321, label %341

321:                                              ; preds = %316, %311
  %322 = load i64, ptr %92, align 8
  %323 = load i64, ptr %89, align 8
  %324 = add nsw i64 %323, %322
  store i64 %324, ptr %89, align 8
  %325 = load ptr, ptr %86, align 8
  store ptr %325, ptr %49, align 8
  %326 = load ptr, ptr %49, align 8
  store ptr %326, ptr %34, align 8
  %327 = load ptr, ptr %34, align 8
  %328 = load i32, ptr %327, align 8
  %329 = icmp slt i32 %328, 0
  %330 = zext i1 %329 to i32
  %331 = icmp ne i32 %330, 0
  br i1 %331, label %332, label %333

332:                                              ; preds = %321
  br label %340

333:                                              ; preds = %321
  %334 = load ptr, ptr %49, align 8
  %335 = load i32, ptr %334, align 8
  %336 = add i32 %335, -1
  store i32 %336, ptr %334, align 8
  %337 = icmp eq i32 %336, 0
  br i1 %337, label %338, label %340

338:                                              ; preds = %333
  %339 = load ptr, ptr %49, align 8
  call void @_Py_Dealloc(ptr noundef %339) #7
  br label %340

340:                                              ; preds = %332, %333, %338
  br label %256, !llvm.loop !23

341:                                              ; preds = %316, %311, %305
  br label %342

342:                                              ; preds = %341, %291
  %343 = load i64, ptr %89, align 8
  %344 = call ptr @PyLong_FromSsize_t(i64 noundef %343)
  store ptr %344, ptr %84, align 8
  %345 = load ptr, ptr %84, align 8
  %346 = icmp eq ptr %345, null
  br i1 %346, label %347, label %380

347:                                              ; preds = %342
  %348 = load ptr, ptr %86, align 8
  store ptr %348, ptr %50, align 8
  %349 = load ptr, ptr %50, align 8
  store ptr %349, ptr %33, align 8
  %350 = load ptr, ptr %33, align 8
  %351 = load i32, ptr %350, align 8
  %352 = icmp slt i32 %351, 0
  %353 = zext i1 %352 to i32
  %354 = icmp ne i32 %353, 0
  br i1 %354, label %355, label %356

355:                                              ; preds = %347
  br label %363

356:                                              ; preds = %347
  %357 = load ptr, ptr %50, align 8
  %358 = load i32, ptr %357, align 8
  %359 = add i32 %358, -1
  store i32 %359, ptr %357, align 8
  %360 = icmp eq i32 %359, 0
  br i1 %360, label %361, label %363

361:                                              ; preds = %356
  %362 = load ptr, ptr %50, align 8
  call void @_Py_Dealloc(ptr noundef %362) #7
  br label %363

363:                                              ; preds = %355, %356, %361
  %364 = load ptr, ptr %87, align 8
  store ptr %364, ptr %51, align 8
  %365 = load ptr, ptr %51, align 8
  store ptr %365, ptr %32, align 8
  %366 = load ptr, ptr %32, align 8
  %367 = load i32, ptr %366, align 8
  %368 = icmp slt i32 %367, 0
  %369 = zext i1 %368 to i32
  %370 = icmp ne i32 %369, 0
  br i1 %370, label %371, label %372

371:                                              ; preds = %363
  br label %379

372:                                              ; preds = %363
  %373 = load ptr, ptr %51, align 8
  %374 = load i32, ptr %373, align 8
  %375 = add i32 %374, -1
  store i32 %375, ptr %373, align 8
  %376 = icmp eq i32 %375, 0
  br i1 %376, label %377, label %379

377:                                              ; preds = %372
  %378 = load ptr, ptr %51, align 8
  call void @_Py_Dealloc(ptr noundef %378) #7
  br label %379

379:                                              ; preds = %371, %372, %377
  store ptr null, ptr %80, align 8
  br label %1082

380:                                              ; preds = %342
  %381 = load ptr, ptr %84, align 8
  %382 = load ptr, ptr %86, align 8
  %383 = call ptr @PyNumber_Add(ptr noundef %381, ptr noundef %382)
  store ptr %383, ptr %85, align 8
  %384 = load ptr, ptr %84, align 8
  store ptr %384, ptr %52, align 8
  %385 = load ptr, ptr %52, align 8
  store ptr %385, ptr %31, align 8
  %386 = load ptr, ptr %31, align 8
  %387 = load i32, ptr %386, align 8
  %388 = icmp slt i32 %387, 0
  %389 = zext i1 %388 to i32
  %390 = icmp ne i32 %389, 0
  br i1 %390, label %391, label %392

391:                                              ; preds = %380
  br label %399

392:                                              ; preds = %380
  %393 = load ptr, ptr %52, align 8
  %394 = load i32, ptr %393, align 8
  %395 = add i32 %394, -1
  store i32 %395, ptr %393, align 8
  %396 = icmp eq i32 %395, 0
  br i1 %396, label %397, label %399

397:                                              ; preds = %392
  %398 = load ptr, ptr %52, align 8
  call void @_Py_Dealloc(ptr noundef %398) #7
  br label %399

399:                                              ; preds = %391, %392, %397
  %400 = load ptr, ptr %86, align 8
  store ptr %400, ptr %53, align 8
  %401 = load ptr, ptr %53, align 8
  store ptr %401, ptr %30, align 8
  %402 = load ptr, ptr %30, align 8
  %403 = load i32, ptr %402, align 8
  %404 = icmp slt i32 %403, 0
  %405 = zext i1 %404 to i32
  %406 = icmp ne i32 %405, 0
  br i1 %406, label %407, label %408

407:                                              ; preds = %399
  br label %415

408:                                              ; preds = %399
  %409 = load ptr, ptr %53, align 8
  %410 = load i32, ptr %409, align 8
  %411 = add i32 %410, -1
  store i32 %411, ptr %409, align 8
  %412 = icmp eq i32 %411, 0
  br i1 %412, label %413, label %415

413:                                              ; preds = %408
  %414 = load ptr, ptr %53, align 8
  call void @_Py_Dealloc(ptr noundef %414) #7
  br label %415

415:                                              ; preds = %407, %408, %413
  %416 = load ptr, ptr %85, align 8
  store ptr %416, ptr %84, align 8
  %417 = load ptr, ptr %84, align 8
  %418 = icmp eq ptr %417, null
  br i1 %418, label %419, label %436

419:                                              ; preds = %415
  %420 = load ptr, ptr %87, align 8
  store ptr %420, ptr %54, align 8
  %421 = load ptr, ptr %54, align 8
  store ptr %421, ptr %29, align 8
  %422 = load ptr, ptr %29, align 8
  %423 = load i32, ptr %422, align 8
  %424 = icmp slt i32 %423, 0
  %425 = zext i1 %424 to i32
  %426 = icmp ne i32 %425, 0
  br i1 %426, label %427, label %428

427:                                              ; preds = %419
  br label %435

428:                                              ; preds = %419
  %429 = load ptr, ptr %54, align 8
  %430 = load i32, ptr %429, align 8
  %431 = add i32 %430, -1
  store i32 %431, ptr %429, align 8
  %432 = icmp eq i32 %431, 0
  br i1 %432, label %433, label %435

433:                                              ; preds = %428
  %434 = load ptr, ptr %54, align 8
  call void @_Py_Dealloc(ptr noundef %434) #7
  br label %435

435:                                              ; preds = %427, %428, %433
  store ptr null, ptr %80, align 8
  br label %1082

436:                                              ; preds = %415
  br label %256, !llvm.loop !23

437:                                              ; preds = %256
  br label %438

438:                                              ; preds = %437, %224
  %439 = load ptr, ptr %84, align 8
  %440 = call i32 @Py_IS_TYPE(ptr noundef %439, ptr noundef @PyFloat_Type)
  %441 = icmp ne i32 %440, 0
  br i1 %441, label %442, label %684

442:                                              ; preds = %438
  %443 = load ptr, ptr %84, align 8
  %444 = call double @PyFloat_AS_DOUBLE(ptr noundef %443)
  %445 = call %struct.CompensatedSum @cs_from_double(double noundef %444)
  %446 = getelementptr inbounds %struct.CompensatedSum, ptr %93, i32 0, i32 0
  %447 = extractvalue %struct.CompensatedSum %445, 0
  store double %447, ptr %446, align 8
  %448 = getelementptr inbounds %struct.CompensatedSum, ptr %93, i32 0, i32 1
  %449 = extractvalue %struct.CompensatedSum %445, 1
  store double %449, ptr %448, align 8
  br label %450

450:                                              ; preds = %442
  store ptr %84, ptr %94, align 8
  %451 = load ptr, ptr %94, align 8
  %452 = load ptr, ptr %451, align 8
  store ptr %452, ptr %95, align 8
  %453 = load ptr, ptr %94, align 8
  store ptr null, ptr %453, align 8
  %454 = load ptr, ptr %95, align 8
  store ptr %454, ptr %55, align 8
  %455 = load ptr, ptr %55, align 8
  store ptr %455, ptr %28, align 8
  %456 = load ptr, ptr %28, align 8
  %457 = load i32, ptr %456, align 8
  %458 = icmp slt i32 %457, 0
  %459 = zext i1 %458 to i32
  %460 = icmp ne i32 %459, 0
  br i1 %460, label %461, label %462

461:                                              ; preds = %450
  br label %469

462:                                              ; preds = %450
  %463 = load ptr, ptr %55, align 8
  %464 = load i32, ptr %463, align 8
  %465 = add i32 %464, -1
  store i32 %465, ptr %463, align 8
  %466 = icmp eq i32 %465, 0
  br i1 %466, label %467, label %469

467:                                              ; preds = %462
  %468 = load ptr, ptr %55, align 8
  call void @_Py_Dealloc(ptr noundef %468) #7
  br label %469

469:                                              ; preds = %461, %462, %467
  br label %470

470:                                              ; preds = %469
  br label %471

471:                                              ; preds = %682, %553, %507, %470
  %472 = load ptr, ptr %84, align 8
  %473 = icmp eq ptr %472, null
  br i1 %473, label %474, label %683

474:                                              ; preds = %471
  %475 = load ptr, ptr %87, align 8
  %476 = call ptr @PyIter_Next(ptr noundef %475)
  store ptr %476, ptr %86, align 8
  %477 = load ptr, ptr %86, align 8
  %478 = icmp eq ptr %477, null
  br i1 %478, label %479, label %503

479:                                              ; preds = %474
  %480 = load ptr, ptr %87, align 8
  store ptr %480, ptr %56, align 8
  %481 = load ptr, ptr %56, align 8
  store ptr %481, ptr %27, align 8
  %482 = load ptr, ptr %27, align 8
  %483 = load i32, ptr %482, align 8
  %484 = icmp slt i32 %483, 0
  %485 = zext i1 %484 to i32
  %486 = icmp ne i32 %485, 0
  br i1 %486, label %487, label %488

487:                                              ; preds = %479
  br label %495

488:                                              ; preds = %479
  %489 = load ptr, ptr %56, align 8
  %490 = load i32, ptr %489, align 8
  %491 = add i32 %490, -1
  store i32 %491, ptr %489, align 8
  %492 = icmp eq i32 %491, 0
  br i1 %492, label %493, label %495

493:                                              ; preds = %488
  %494 = load ptr, ptr %56, align 8
  call void @_Py_Dealloc(ptr noundef %494) #7
  br label %495

495:                                              ; preds = %487, %488, %493
  %496 = call ptr @PyErr_Occurred()
  %497 = icmp ne ptr %496, null
  br i1 %497, label %498, label %499

498:                                              ; preds = %495
  store ptr null, ptr %80, align 8
  br label %1082

499:                                              ; preds = %495
  %500 = load [2 x double], ptr %93, align 8
  %501 = call double @cs_to_double([2 x double] %500)
  %502 = call ptr @PyFloat_FromDouble(double noundef %501)
  store ptr %502, ptr %80, align 8
  br label %1082

503:                                              ; preds = %474
  %504 = load ptr, ptr %86, align 8
  %505 = call i32 @Py_IS_TYPE(ptr noundef %504, ptr noundef @PyFloat_Type)
  %506 = icmp ne i32 %505, 0
  br i1 %506, label %507, label %517

507:                                              ; preds = %503
  %508 = load ptr, ptr %86, align 8
  %509 = call double @PyFloat_AS_DOUBLE(ptr noundef %508)
  %510 = load [2 x double], ptr %93, align 8
  %511 = call %struct.CompensatedSum @cs_add([2 x double] %510, double noundef %509)
  %512 = getelementptr inbounds %struct.CompensatedSum, ptr %96, i32 0, i32 0
  %513 = extractvalue %struct.CompensatedSum %511, 0
  store double %513, ptr %512, align 8
  %514 = getelementptr inbounds %struct.CompensatedSum, ptr %96, i32 0, i32 1
  %515 = extractvalue %struct.CompensatedSum %511, 1
  store double %515, ptr %514, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %93, ptr align 8 %96, i64 16, i1 false)
  %516 = load ptr, ptr %86, align 8
  call void @_Py_DECREF_SPECIALIZED(ptr noundef %516, ptr noundef @_PyFloat_ExactDealloc)
  br label %471, !llvm.loop !24

517:                                              ; preds = %503
  %518 = load ptr, ptr %86, align 8
  %519 = call ptr @_Py_TYPE(ptr noundef %518)
  %520 = call i32 @PyType_HasFeature(ptr noundef %519, i64 noundef 16777216)
  %521 = icmp ne i32 %520, 0
  br i1 %521, label %522, label %587

522:                                              ; preds = %517
  %523 = load ptr, ptr %86, align 8
  %524 = call double @PyLong_AsDouble(ptr noundef %523)
  store double %524, ptr %97, align 8
  %525 = load double, ptr %97, align 8
  %526 = fcmp une double %525, -1.000000e+00
  br i1 %526, label %530, label %527

527:                                              ; preds = %522
  %528 = call ptr @PyErr_Occurred()
  %529 = icmp ne ptr %528, null
  br i1 %529, label %554, label %530

530:                                              ; preds = %527, %522
  %531 = load double, ptr %97, align 8
  %532 = load [2 x double], ptr %93, align 8
  %533 = call %struct.CompensatedSum @cs_add([2 x double] %532, double noundef %531)
  %534 = getelementptr inbounds %struct.CompensatedSum, ptr %98, i32 0, i32 0
  %535 = extractvalue %struct.CompensatedSum %533, 0
  store double %535, ptr %534, align 8
  %536 = getelementptr inbounds %struct.CompensatedSum, ptr %98, i32 0, i32 1
  %537 = extractvalue %struct.CompensatedSum %533, 1
  store double %537, ptr %536, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %93, ptr align 8 %98, i64 16, i1 false)
  %538 = load ptr, ptr %86, align 8
  store ptr %538, ptr %57, align 8
  %539 = load ptr, ptr %57, align 8
  store ptr %539, ptr %26, align 8
  %540 = load ptr, ptr %26, align 8
  %541 = load i32, ptr %540, align 8
  %542 = icmp slt i32 %541, 0
  %543 = zext i1 %542 to i32
  %544 = icmp ne i32 %543, 0
  br i1 %544, label %545, label %546

545:                                              ; preds = %530
  br label %553

546:                                              ; preds = %530
  %547 = load ptr, ptr %57, align 8
  %548 = load i32, ptr %547, align 8
  %549 = add i32 %548, -1
  store i32 %549, ptr %547, align 8
  %550 = icmp eq i32 %549, 0
  br i1 %550, label %551, label %553

551:                                              ; preds = %546
  %552 = load ptr, ptr %57, align 8
  call void @_Py_Dealloc(ptr noundef %552) #7
  br label %553

553:                                              ; preds = %545, %546, %551
  br label %471, !llvm.loop !24

554:                                              ; preds = %527
  %555 = load ptr, ptr %86, align 8
  store ptr %555, ptr %58, align 8
  %556 = load ptr, ptr %58, align 8
  store ptr %556, ptr %25, align 8
  %557 = load ptr, ptr %25, align 8
  %558 = load i32, ptr %557, align 8
  %559 = icmp slt i32 %558, 0
  %560 = zext i1 %559 to i32
  %561 = icmp ne i32 %560, 0
  br i1 %561, label %562, label %563

562:                                              ; preds = %554
  br label %570

563:                                              ; preds = %554
  %564 = load ptr, ptr %58, align 8
  %565 = load i32, ptr %564, align 8
  %566 = add i32 %565, -1
  store i32 %566, ptr %564, align 8
  %567 = icmp eq i32 %566, 0
  br i1 %567, label %568, label %570

568:                                              ; preds = %563
  %569 = load ptr, ptr %58, align 8
  call void @_Py_Dealloc(ptr noundef %569) #7
  br label %570

570:                                              ; preds = %562, %563, %568
  %571 = load ptr, ptr %87, align 8
  store ptr %571, ptr %59, align 8
  %572 = load ptr, ptr %59, align 8
  store ptr %572, ptr %24, align 8
  %573 = load ptr, ptr %24, align 8
  %574 = load i32, ptr %573, align 8
  %575 = icmp slt i32 %574, 0
  %576 = zext i1 %575 to i32
  %577 = icmp ne i32 %576, 0
  br i1 %577, label %578, label %579

578:                                              ; preds = %570
  br label %586

579:                                              ; preds = %570
  %580 = load ptr, ptr %59, align 8
  %581 = load i32, ptr %580, align 8
  %582 = add i32 %581, -1
  store i32 %582, ptr %580, align 8
  %583 = icmp eq i32 %582, 0
  br i1 %583, label %584, label %586

584:                                              ; preds = %579
  %585 = load ptr, ptr %59, align 8
  call void @_Py_Dealloc(ptr noundef %585) #7
  br label %586

586:                                              ; preds = %578, %579, %584
  store ptr null, ptr %80, align 8
  br label %1082

587:                                              ; preds = %517
  %588 = load [2 x double], ptr %93, align 8
  %589 = call double @cs_to_double([2 x double] %588)
  %590 = call ptr @PyFloat_FromDouble(double noundef %589)
  store ptr %590, ptr %84, align 8
  %591 = load ptr, ptr %84, align 8
  %592 = icmp eq ptr %591, null
  br i1 %592, label %593, label %626

593:                                              ; preds = %587
  %594 = load ptr, ptr %86, align 8
  store ptr %594, ptr %60, align 8
  %595 = load ptr, ptr %60, align 8
  store ptr %595, ptr %23, align 8
  %596 = load ptr, ptr %23, align 8
  %597 = load i32, ptr %596, align 8
  %598 = icmp slt i32 %597, 0
  %599 = zext i1 %598 to i32
  %600 = icmp ne i32 %599, 0
  br i1 %600, label %601, label %602

601:                                              ; preds = %593
  br label %609

602:                                              ; preds = %593
  %603 = load ptr, ptr %60, align 8
  %604 = load i32, ptr %603, align 8
  %605 = add i32 %604, -1
  store i32 %605, ptr %603, align 8
  %606 = icmp eq i32 %605, 0
  br i1 %606, label %607, label %609

607:                                              ; preds = %602
  %608 = load ptr, ptr %60, align 8
  call void @_Py_Dealloc(ptr noundef %608) #7
  br label %609

609:                                              ; preds = %601, %602, %607
  %610 = load ptr, ptr %87, align 8
  store ptr %610, ptr %61, align 8
  %611 = load ptr, ptr %61, align 8
  store ptr %611, ptr %22, align 8
  %612 = load ptr, ptr %22, align 8
  %613 = load i32, ptr %612, align 8
  %614 = icmp slt i32 %613, 0
  %615 = zext i1 %614 to i32
  %616 = icmp ne i32 %615, 0
  br i1 %616, label %617, label %618

617:                                              ; preds = %609
  br label %625

618:                                              ; preds = %609
  %619 = load ptr, ptr %61, align 8
  %620 = load i32, ptr %619, align 8
  %621 = add i32 %620, -1
  store i32 %621, ptr %619, align 8
  %622 = icmp eq i32 %621, 0
  br i1 %622, label %623, label %625

623:                                              ; preds = %618
  %624 = load ptr, ptr %61, align 8
  call void @_Py_Dealloc(ptr noundef %624) #7
  br label %625

625:                                              ; preds = %617, %618, %623
  store ptr null, ptr %80, align 8
  br label %1082

626:                                              ; preds = %587
  %627 = load ptr, ptr %84, align 8
  %628 = load ptr, ptr %86, align 8
  %629 = call ptr @PyNumber_Add(ptr noundef %627, ptr noundef %628)
  store ptr %629, ptr %85, align 8
  %630 = load ptr, ptr %84, align 8
  store ptr %630, ptr %62, align 8
  %631 = load ptr, ptr %62, align 8
  store ptr %631, ptr %21, align 8
  %632 = load ptr, ptr %21, align 8
  %633 = load i32, ptr %632, align 8
  %634 = icmp slt i32 %633, 0
  %635 = zext i1 %634 to i32
  %636 = icmp ne i32 %635, 0
  br i1 %636, label %637, label %638

637:                                              ; preds = %626
  br label %645

638:                                              ; preds = %626
  %639 = load ptr, ptr %62, align 8
  %640 = load i32, ptr %639, align 8
  %641 = add i32 %640, -1
  store i32 %641, ptr %639, align 8
  %642 = icmp eq i32 %641, 0
  br i1 %642, label %643, label %645

643:                                              ; preds = %638
  %644 = load ptr, ptr %62, align 8
  call void @_Py_Dealloc(ptr noundef %644) #7
  br label %645

645:                                              ; preds = %637, %638, %643
  %646 = load ptr, ptr %86, align 8
  store ptr %646, ptr %63, align 8
  %647 = load ptr, ptr %63, align 8
  store ptr %647, ptr %20, align 8
  %648 = load ptr, ptr %20, align 8
  %649 = load i32, ptr %648, align 8
  %650 = icmp slt i32 %649, 0
  %651 = zext i1 %650 to i32
  %652 = icmp ne i32 %651, 0
  br i1 %652, label %653, label %654

653:                                              ; preds = %645
  br label %661

654:                                              ; preds = %645
  %655 = load ptr, ptr %63, align 8
  %656 = load i32, ptr %655, align 8
  %657 = add i32 %656, -1
  store i32 %657, ptr %655, align 8
  %658 = icmp eq i32 %657, 0
  br i1 %658, label %659, label %661

659:                                              ; preds = %654
  %660 = load ptr, ptr %63, align 8
  call void @_Py_Dealloc(ptr noundef %660) #7
  br label %661

661:                                              ; preds = %653, %654, %659
  %662 = load ptr, ptr %85, align 8
  store ptr %662, ptr %84, align 8
  %663 = load ptr, ptr %84, align 8
  %664 = icmp eq ptr %663, null
  br i1 %664, label %665, label %682

665:                                              ; preds = %661
  %666 = load ptr, ptr %87, align 8
  store ptr %666, ptr %64, align 8
  %667 = load ptr, ptr %64, align 8
  store ptr %667, ptr %19, align 8
  %668 = load ptr, ptr %19, align 8
  %669 = load i32, ptr %668, align 8
  %670 = icmp slt i32 %669, 0
  %671 = zext i1 %670 to i32
  %672 = icmp ne i32 %671, 0
  br i1 %672, label %673, label %674

673:                                              ; preds = %665
  br label %681

674:                                              ; preds = %665
  %675 = load ptr, ptr %64, align 8
  %676 = load i32, ptr %675, align 8
  %677 = add i32 %676, -1
  store i32 %677, ptr %675, align 8
  %678 = icmp eq i32 %677, 0
  br i1 %678, label %679, label %681

679:                                              ; preds = %674
  %680 = load ptr, ptr %64, align 8
  call void @_Py_Dealloc(ptr noundef %680) #7
  br label %681

681:                                              ; preds = %673, %674, %679
  store ptr null, ptr %80, align 8
  br label %1082

682:                                              ; preds = %661
  br label %471, !llvm.loop !24

683:                                              ; preds = %471
  br label %684

684:                                              ; preds = %683, %438
  %685 = load ptr, ptr %84, align 8
  %686 = call i32 @Py_IS_TYPE(ptr noundef %685, ptr noundef @PyComplex_Type)
  %687 = icmp ne i32 %686, 0
  br i1 %687, label %688, label %991

688:                                              ; preds = %684
  %689 = load ptr, ptr %84, align 8
  %690 = call %struct.Py_complex @PyComplex_AsCComplex(ptr noundef %689)
  %691 = getelementptr inbounds %struct.Py_complex, ptr %99, i32 0, i32 0
  %692 = extractvalue %struct.Py_complex %690, 0
  store double %692, ptr %691, align 8
  %693 = getelementptr inbounds %struct.Py_complex, ptr %99, i32 0, i32 1
  %694 = extractvalue %struct.Py_complex %690, 1
  store double %694, ptr %693, align 8
  %695 = getelementptr inbounds %struct.Py_complex, ptr %99, i32 0, i32 0
  %696 = load double, ptr %695, align 8
  %697 = call %struct.CompensatedSum @cs_from_double(double noundef %696)
  %698 = getelementptr inbounds %struct.CompensatedSum, ptr %100, i32 0, i32 0
  %699 = extractvalue %struct.CompensatedSum %697, 0
  store double %699, ptr %698, align 8
  %700 = getelementptr inbounds %struct.CompensatedSum, ptr %100, i32 0, i32 1
  %701 = extractvalue %struct.CompensatedSum %697, 1
  store double %701, ptr %700, align 8
  %702 = getelementptr inbounds %struct.Py_complex, ptr %99, i32 0, i32 1
  %703 = load double, ptr %702, align 8
  %704 = call %struct.CompensatedSum @cs_from_double(double noundef %703)
  %705 = getelementptr inbounds %struct.CompensatedSum, ptr %101, i32 0, i32 0
  %706 = extractvalue %struct.CompensatedSum %704, 0
  store double %706, ptr %705, align 8
  %707 = getelementptr inbounds %struct.CompensatedSum, ptr %101, i32 0, i32 1
  %708 = extractvalue %struct.CompensatedSum %704, 1
  store double %708, ptr %707, align 8
  br label %709

709:                                              ; preds = %688
  store ptr %84, ptr %102, align 8
  %710 = load ptr, ptr %102, align 8
  %711 = load ptr, ptr %710, align 8
  store ptr %711, ptr %103, align 8
  %712 = load ptr, ptr %102, align 8
  store ptr null, ptr %712, align 8
  %713 = load ptr, ptr %103, align 8
  store ptr %713, ptr %65, align 8
  %714 = load ptr, ptr %65, align 8
  store ptr %714, ptr %18, align 8
  %715 = load ptr, ptr %18, align 8
  %716 = load i32, ptr %715, align 8
  %717 = icmp slt i32 %716, 0
  %718 = zext i1 %717 to i32
  %719 = icmp ne i32 %718, 0
  br i1 %719, label %720, label %721

720:                                              ; preds = %709
  br label %728

721:                                              ; preds = %709
  %722 = load ptr, ptr %65, align 8
  %723 = load i32, ptr %722, align 8
  %724 = add i32 %723, -1
  store i32 %724, ptr %722, align 8
  %725 = icmp eq i32 %724, 0
  br i1 %725, label %726, label %728

726:                                              ; preds = %721
  %727 = load ptr, ptr %65, align 8
  call void @_Py_Dealloc(ptr noundef %727) #7
  br label %728

728:                                              ; preds = %720, %721, %726
  br label %729

729:                                              ; preds = %728
  br label %730

730:                                              ; preds = %989, %881, %843, %806, %729
  %731 = load ptr, ptr %84, align 8
  %732 = icmp eq ptr %731, null
  br i1 %732, label %733, label %990

733:                                              ; preds = %730
  %734 = load ptr, ptr %87, align 8
  %735 = call ptr @PyIter_Next(ptr noundef %734)
  store ptr %735, ptr %86, align 8
  %736 = load ptr, ptr %86, align 8
  %737 = icmp eq ptr %736, null
  br i1 %737, label %738, label %764

738:                                              ; preds = %733
  %739 = load ptr, ptr %87, align 8
  store ptr %739, ptr %66, align 8
  %740 = load ptr, ptr %66, align 8
  store ptr %740, ptr %17, align 8
  %741 = load ptr, ptr %17, align 8
  %742 = load i32, ptr %741, align 8
  %743 = icmp slt i32 %742, 0
  %744 = zext i1 %743 to i32
  %745 = icmp ne i32 %744, 0
  br i1 %745, label %746, label %747

746:                                              ; preds = %738
  br label %754

747:                                              ; preds = %738
  %748 = load ptr, ptr %66, align 8
  %749 = load i32, ptr %748, align 8
  %750 = add i32 %749, -1
  store i32 %750, ptr %748, align 8
  %751 = icmp eq i32 %750, 0
  br i1 %751, label %752, label %754

752:                                              ; preds = %747
  %753 = load ptr, ptr %66, align 8
  call void @_Py_Dealloc(ptr noundef %753) #7
  br label %754

754:                                              ; preds = %746, %747, %752
  %755 = call ptr @PyErr_Occurred()
  %756 = icmp ne ptr %755, null
  br i1 %756, label %757, label %758

757:                                              ; preds = %754
  store ptr null, ptr %80, align 8
  br label %1082

758:                                              ; preds = %754
  %759 = load [2 x double], ptr %100, align 8
  %760 = call double @cs_to_double([2 x double] %759)
  %761 = load [2 x double], ptr %101, align 8
  %762 = call double @cs_to_double([2 x double] %761)
  %763 = call ptr @PyComplex_FromDoubles(double noundef %760, double noundef %762)
  store ptr %763, ptr %80, align 8
  br label %1082

764:                                              ; preds = %733
  %765 = load ptr, ptr %86, align 8
  %766 = call i32 @Py_IS_TYPE(ptr noundef %765, ptr noundef @PyComplex_Type)
  %767 = icmp ne i32 %766, 0
  br i1 %767, label %768, label %807

768:                                              ; preds = %764
  %769 = load ptr, ptr %86, align 8
  %770 = call %struct.Py_complex @PyComplex_AsCComplex(ptr noundef %769)
  %771 = getelementptr inbounds %struct.Py_complex, ptr %104, i32 0, i32 0
  %772 = extractvalue %struct.Py_complex %770, 0
  store double %772, ptr %771, align 8
  %773 = getelementptr inbounds %struct.Py_complex, ptr %104, i32 0, i32 1
  %774 = extractvalue %struct.Py_complex %770, 1
  store double %774, ptr %773, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %99, ptr align 8 %104, i64 16, i1 false)
  %775 = getelementptr inbounds %struct.Py_complex, ptr %99, i32 0, i32 0
  %776 = load double, ptr %775, align 8
  %777 = load [2 x double], ptr %100, align 8
  %778 = call %struct.CompensatedSum @cs_add([2 x double] %777, double noundef %776)
  %779 = getelementptr inbounds %struct.CompensatedSum, ptr %105, i32 0, i32 0
  %780 = extractvalue %struct.CompensatedSum %778, 0
  store double %780, ptr %779, align 8
  %781 = getelementptr inbounds %struct.CompensatedSum, ptr %105, i32 0, i32 1
  %782 = extractvalue %struct.CompensatedSum %778, 1
  store double %782, ptr %781, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %100, ptr align 8 %105, i64 16, i1 false)
  %783 = getelementptr inbounds %struct.Py_complex, ptr %99, i32 0, i32 1
  %784 = load double, ptr %783, align 8
  %785 = load [2 x double], ptr %101, align 8
  %786 = call %struct.CompensatedSum @cs_add([2 x double] %785, double noundef %784)
  %787 = getelementptr inbounds %struct.CompensatedSum, ptr %106, i32 0, i32 0
  %788 = extractvalue %struct.CompensatedSum %786, 0
  store double %788, ptr %787, align 8
  %789 = getelementptr inbounds %struct.CompensatedSum, ptr %106, i32 0, i32 1
  %790 = extractvalue %struct.CompensatedSum %786, 1
  store double %790, ptr %789, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %101, ptr align 8 %106, i64 16, i1 false)
  %791 = load ptr, ptr %86, align 8
  store ptr %791, ptr %67, align 8
  %792 = load ptr, ptr %67, align 8
  store ptr %792, ptr %16, align 8
  %793 = load ptr, ptr %16, align 8
  %794 = load i32, ptr %793, align 8
  %795 = icmp slt i32 %794, 0
  %796 = zext i1 %795 to i32
  %797 = icmp ne i32 %796, 0
  br i1 %797, label %798, label %799

798:                                              ; preds = %768
  br label %806

799:                                              ; preds = %768
  %800 = load ptr, ptr %67, align 8
  %801 = load i32, ptr %800, align 8
  %802 = add i32 %801, -1
  store i32 %802, ptr %800, align 8
  %803 = icmp eq i32 %802, 0
  br i1 %803, label %804, label %806

804:                                              ; preds = %799
  %805 = load ptr, ptr %67, align 8
  call void @_Py_Dealloc(ptr noundef %805) #7
  br label %806

806:                                              ; preds = %798, %799, %804
  br label %730, !llvm.loop !25

807:                                              ; preds = %764
  %808 = load ptr, ptr %86, align 8
  %809 = call ptr @_Py_TYPE(ptr noundef %808)
  %810 = call i32 @PyType_HasFeature(ptr noundef %809, i64 noundef 16777216)
  %811 = icmp ne i32 %810, 0
  br i1 %811, label %812, label %877

812:                                              ; preds = %807
  %813 = load ptr, ptr %86, align 8
  %814 = call double @PyLong_AsDouble(ptr noundef %813)
  store double %814, ptr %107, align 8
  %815 = load double, ptr %107, align 8
  %816 = fcmp une double %815, -1.000000e+00
  br i1 %816, label %820, label %817

817:                                              ; preds = %812
  %818 = call ptr @PyErr_Occurred()
  %819 = icmp ne ptr %818, null
  br i1 %819, label %844, label %820

820:                                              ; preds = %817, %812
  %821 = load double, ptr %107, align 8
  %822 = load [2 x double], ptr %100, align 8
  %823 = call %struct.CompensatedSum @cs_add([2 x double] %822, double noundef %821)
  %824 = getelementptr inbounds %struct.CompensatedSum, ptr %108, i32 0, i32 0
  %825 = extractvalue %struct.CompensatedSum %823, 0
  store double %825, ptr %824, align 8
  %826 = getelementptr inbounds %struct.CompensatedSum, ptr %108, i32 0, i32 1
  %827 = extractvalue %struct.CompensatedSum %823, 1
  store double %827, ptr %826, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %100, ptr align 8 %108, i64 16, i1 false)
  %828 = load ptr, ptr %86, align 8
  store ptr %828, ptr %68, align 8
  %829 = load ptr, ptr %68, align 8
  store ptr %829, ptr %15, align 8
  %830 = load ptr, ptr %15, align 8
  %831 = load i32, ptr %830, align 8
  %832 = icmp slt i32 %831, 0
  %833 = zext i1 %832 to i32
  %834 = icmp ne i32 %833, 0
  br i1 %834, label %835, label %836

835:                                              ; preds = %820
  br label %843

836:                                              ; preds = %820
  %837 = load ptr, ptr %68, align 8
  %838 = load i32, ptr %837, align 8
  %839 = add i32 %838, -1
  store i32 %839, ptr %837, align 8
  %840 = icmp eq i32 %839, 0
  br i1 %840, label %841, label %843

841:                                              ; preds = %836
  %842 = load ptr, ptr %68, align 8
  call void @_Py_Dealloc(ptr noundef %842) #7
  br label %843

843:                                              ; preds = %835, %836, %841
  br label %730, !llvm.loop !25

844:                                              ; preds = %817
  %845 = load ptr, ptr %86, align 8
  store ptr %845, ptr %69, align 8
  %846 = load ptr, ptr %69, align 8
  store ptr %846, ptr %14, align 8
  %847 = load ptr, ptr %14, align 8
  %848 = load i32, ptr %847, align 8
  %849 = icmp slt i32 %848, 0
  %850 = zext i1 %849 to i32
  %851 = icmp ne i32 %850, 0
  br i1 %851, label %852, label %853

852:                                              ; preds = %844
  br label %860

853:                                              ; preds = %844
  %854 = load ptr, ptr %69, align 8
  %855 = load i32, ptr %854, align 8
  %856 = add i32 %855, -1
  store i32 %856, ptr %854, align 8
  %857 = icmp eq i32 %856, 0
  br i1 %857, label %858, label %860

858:                                              ; preds = %853
  %859 = load ptr, ptr %69, align 8
  call void @_Py_Dealloc(ptr noundef %859) #7
  br label %860

860:                                              ; preds = %852, %853, %858
  %861 = load ptr, ptr %87, align 8
  store ptr %861, ptr %70, align 8
  %862 = load ptr, ptr %70, align 8
  store ptr %862, ptr %13, align 8
  %863 = load ptr, ptr %13, align 8
  %864 = load i32, ptr %863, align 8
  %865 = icmp slt i32 %864, 0
  %866 = zext i1 %865 to i32
  %867 = icmp ne i32 %866, 0
  br i1 %867, label %868, label %869

868:                                              ; preds = %860
  br label %876

869:                                              ; preds = %860
  %870 = load ptr, ptr %70, align 8
  %871 = load i32, ptr %870, align 8
  %872 = add i32 %871, -1
  store i32 %872, ptr %870, align 8
  %873 = icmp eq i32 %872, 0
  br i1 %873, label %874, label %876

874:                                              ; preds = %869
  %875 = load ptr, ptr %70, align 8
  call void @_Py_Dealloc(ptr noundef %875) #7
  br label %876

876:                                              ; preds = %868, %869, %874
  store ptr null, ptr %80, align 8
  br label %1082

877:                                              ; preds = %807
  %878 = load ptr, ptr %86, align 8
  %879 = call i32 @PyObject_TypeCheck(ptr noundef %878, ptr noundef @PyFloat_Type)
  %880 = icmp ne i32 %879, 0
  br i1 %880, label %881, label %892

881:                                              ; preds = %877
  %882 = load ptr, ptr %86, align 8
  %883 = call double @PyFloat_AS_DOUBLE(ptr noundef %882)
  store double %883, ptr %109, align 8
  %884 = load double, ptr %109, align 8
  %885 = load [2 x double], ptr %100, align 8
  %886 = call %struct.CompensatedSum @cs_add([2 x double] %885, double noundef %884)
  %887 = getelementptr inbounds %struct.CompensatedSum, ptr %110, i32 0, i32 0
  %888 = extractvalue %struct.CompensatedSum %886, 0
  store double %888, ptr %887, align 8
  %889 = getelementptr inbounds %struct.CompensatedSum, ptr %110, i32 0, i32 1
  %890 = extractvalue %struct.CompensatedSum %886, 1
  store double %890, ptr %889, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %100, ptr align 8 %110, i64 16, i1 false)
  %891 = load ptr, ptr %86, align 8
  call void @_Py_DECREF_SPECIALIZED(ptr noundef %891, ptr noundef @_PyFloat_ExactDealloc)
  br label %730, !llvm.loop !25

892:                                              ; preds = %877
  %893 = load [2 x double], ptr %100, align 8
  %894 = call double @cs_to_double([2 x double] %893)
  %895 = load [2 x double], ptr %101, align 8
  %896 = call double @cs_to_double([2 x double] %895)
  %897 = call ptr @PyComplex_FromDoubles(double noundef %894, double noundef %896)
  store ptr %897, ptr %84, align 8
  %898 = load ptr, ptr %84, align 8
  %899 = icmp eq ptr %898, null
  br i1 %899, label %900, label %933

900:                                              ; preds = %892
  %901 = load ptr, ptr %86, align 8
  store ptr %901, ptr %71, align 8
  %902 = load ptr, ptr %71, align 8
  store ptr %902, ptr %12, align 8
  %903 = load ptr, ptr %12, align 8
  %904 = load i32, ptr %903, align 8
  %905 = icmp slt i32 %904, 0
  %906 = zext i1 %905 to i32
  %907 = icmp ne i32 %906, 0
  br i1 %907, label %908, label %909

908:                                              ; preds = %900
  br label %916

909:                                              ; preds = %900
  %910 = load ptr, ptr %71, align 8
  %911 = load i32, ptr %910, align 8
  %912 = add i32 %911, -1
  store i32 %912, ptr %910, align 8
  %913 = icmp eq i32 %912, 0
  br i1 %913, label %914, label %916

914:                                              ; preds = %909
  %915 = load ptr, ptr %71, align 8
  call void @_Py_Dealloc(ptr noundef %915) #7
  br label %916

916:                                              ; preds = %908, %909, %914
  %917 = load ptr, ptr %87, align 8
  store ptr %917, ptr %72, align 8
  %918 = load ptr, ptr %72, align 8
  store ptr %918, ptr %11, align 8
  %919 = load ptr, ptr %11, align 8
  %920 = load i32, ptr %919, align 8
  %921 = icmp slt i32 %920, 0
  %922 = zext i1 %921 to i32
  %923 = icmp ne i32 %922, 0
  br i1 %923, label %924, label %925

924:                                              ; preds = %916
  br label %932

925:                                              ; preds = %916
  %926 = load ptr, ptr %72, align 8
  %927 = load i32, ptr %926, align 8
  %928 = add i32 %927, -1
  store i32 %928, ptr %926, align 8
  %929 = icmp eq i32 %928, 0
  br i1 %929, label %930, label %932

930:                                              ; preds = %925
  %931 = load ptr, ptr %72, align 8
  call void @_Py_Dealloc(ptr noundef %931) #7
  br label %932

932:                                              ; preds = %924, %925, %930
  store ptr null, ptr %80, align 8
  br label %1082

933:                                              ; preds = %892
  %934 = load ptr, ptr %84, align 8
  %935 = load ptr, ptr %86, align 8
  %936 = call ptr @PyNumber_Add(ptr noundef %934, ptr noundef %935)
  store ptr %936, ptr %85, align 8
  %937 = load ptr, ptr %84, align 8
  store ptr %937, ptr %73, align 8
  %938 = load ptr, ptr %73, align 8
  store ptr %938, ptr %10, align 8
  %939 = load ptr, ptr %10, align 8
  %940 = load i32, ptr %939, align 8
  %941 = icmp slt i32 %940, 0
  %942 = zext i1 %941 to i32
  %943 = icmp ne i32 %942, 0
  br i1 %943, label %944, label %945

944:                                              ; preds = %933
  br label %952

945:                                              ; preds = %933
  %946 = load ptr, ptr %73, align 8
  %947 = load i32, ptr %946, align 8
  %948 = add i32 %947, -1
  store i32 %948, ptr %946, align 8
  %949 = icmp eq i32 %948, 0
  br i1 %949, label %950, label %952

950:                                              ; preds = %945
  %951 = load ptr, ptr %73, align 8
  call void @_Py_Dealloc(ptr noundef %951) #7
  br label %952

952:                                              ; preds = %944, %945, %950
  %953 = load ptr, ptr %86, align 8
  store ptr %953, ptr %74, align 8
  %954 = load ptr, ptr %74, align 8
  store ptr %954, ptr %9, align 8
  %955 = load ptr, ptr %9, align 8
  %956 = load i32, ptr %955, align 8
  %957 = icmp slt i32 %956, 0
  %958 = zext i1 %957 to i32
  %959 = icmp ne i32 %958, 0
  br i1 %959, label %960, label %961

960:                                              ; preds = %952
  br label %968

961:                                              ; preds = %952
  %962 = load ptr, ptr %74, align 8
  %963 = load i32, ptr %962, align 8
  %964 = add i32 %963, -1
  store i32 %964, ptr %962, align 8
  %965 = icmp eq i32 %964, 0
  br i1 %965, label %966, label %968

966:                                              ; preds = %961
  %967 = load ptr, ptr %74, align 8
  call void @_Py_Dealloc(ptr noundef %967) #7
  br label %968

968:                                              ; preds = %960, %961, %966
  %969 = load ptr, ptr %85, align 8
  store ptr %969, ptr %84, align 8
  %970 = load ptr, ptr %84, align 8
  %971 = icmp eq ptr %970, null
  br i1 %971, label %972, label %989

972:                                              ; preds = %968
  %973 = load ptr, ptr %87, align 8
  store ptr %973, ptr %75, align 8
  %974 = load ptr, ptr %75, align 8
  store ptr %974, ptr %8, align 8
  %975 = load ptr, ptr %8, align 8
  %976 = load i32, ptr %975, align 8
  %977 = icmp slt i32 %976, 0
  %978 = zext i1 %977 to i32
  %979 = icmp ne i32 %978, 0
  br i1 %979, label %980, label %981

980:                                              ; preds = %972
  br label %988

981:                                              ; preds = %972
  %982 = load ptr, ptr %75, align 8
  %983 = load i32, ptr %982, align 8
  %984 = add i32 %983, -1
  store i32 %984, ptr %982, align 8
  %985 = icmp eq i32 %984, 0
  br i1 %985, label %986, label %988

986:                                              ; preds = %981
  %987 = load ptr, ptr %75, align 8
  call void @_Py_Dealloc(ptr noundef %987) #7
  br label %988

988:                                              ; preds = %980, %981, %986
  store ptr null, ptr %80, align 8
  br label %1082

989:                                              ; preds = %968
  br label %730, !llvm.loop !25

990:                                              ; preds = %730
  br label %991

991:                                              ; preds = %990, %684
  br label %992

992:                                              ; preds = %1063, %991
  %993 = load ptr, ptr %87, align 8
  %994 = call ptr @PyIter_Next(ptr noundef %993)
  store ptr %994, ptr %86, align 8
  %995 = load ptr, ptr %86, align 8
  %996 = icmp eq ptr %995, null
  br i1 %996, label %997, label %1023

997:                                              ; preds = %992
  %998 = call ptr @PyErr_Occurred()
  %999 = icmp ne ptr %998, null
  br i1 %999, label %1000, label %1022

1000:                                             ; preds = %997
  br label %1001

1001:                                             ; preds = %1000
  store ptr %84, ptr %111, align 8
  %1002 = load ptr, ptr %111, align 8
  %1003 = load ptr, ptr %1002, align 8
  store ptr %1003, ptr %112, align 8
  %1004 = load ptr, ptr %111, align 8
  store ptr null, ptr %1004, align 8
  %1005 = load ptr, ptr %112, align 8
  store ptr %1005, ptr %76, align 8
  %1006 = load ptr, ptr %76, align 8
  store ptr %1006, ptr %7, align 8
  %1007 = load ptr, ptr %7, align 8
  %1008 = load i32, ptr %1007, align 8
  %1009 = icmp slt i32 %1008, 0
  %1010 = zext i1 %1009 to i32
  %1011 = icmp ne i32 %1010, 0
  br i1 %1011, label %1012, label %1013

1012:                                             ; preds = %1001
  br label %1020

1013:                                             ; preds = %1001
  %1014 = load ptr, ptr %76, align 8
  %1015 = load i32, ptr %1014, align 8
  %1016 = add i32 %1015, -1
  store i32 %1016, ptr %1014, align 8
  %1017 = icmp eq i32 %1016, 0
  br i1 %1017, label %1018, label %1020

1018:                                             ; preds = %1013
  %1019 = load ptr, ptr %76, align 8
  call void @_Py_Dealloc(ptr noundef %1019) #7
  br label %1020

1020:                                             ; preds = %1012, %1013, %1018
  br label %1021

1021:                                             ; preds = %1020
  br label %1022

1022:                                             ; preds = %1021, %997
  br label %1064

1023:                                             ; preds = %992
  %1024 = load ptr, ptr %84, align 8
  %1025 = load ptr, ptr %86, align 8
  %1026 = call ptr @PyNumber_Add(ptr noundef %1024, ptr noundef %1025)
  store ptr %1026, ptr %85, align 8
  %1027 = load ptr, ptr %84, align 8
  store ptr %1027, ptr %77, align 8
  %1028 = load ptr, ptr %77, align 8
  store ptr %1028, ptr %6, align 8
  %1029 = load ptr, ptr %6, align 8
  %1030 = load i32, ptr %1029, align 8
  %1031 = icmp slt i32 %1030, 0
  %1032 = zext i1 %1031 to i32
  %1033 = icmp ne i32 %1032, 0
  br i1 %1033, label %1034, label %1035

1034:                                             ; preds = %1023
  br label %1042

1035:                                             ; preds = %1023
  %1036 = load ptr, ptr %77, align 8
  %1037 = load i32, ptr %1036, align 8
  %1038 = add i32 %1037, -1
  store i32 %1038, ptr %1036, align 8
  %1039 = icmp eq i32 %1038, 0
  br i1 %1039, label %1040, label %1042

1040:                                             ; preds = %1035
  %1041 = load ptr, ptr %77, align 8
  call void @_Py_Dealloc(ptr noundef %1041) #7
  br label %1042

1042:                                             ; preds = %1034, %1035, %1040
  %1043 = load ptr, ptr %86, align 8
  store ptr %1043, ptr %78, align 8
  %1044 = load ptr, ptr %78, align 8
  store ptr %1044, ptr %5, align 8
  %1045 = load ptr, ptr %5, align 8
  %1046 = load i32, ptr %1045, align 8
  %1047 = icmp slt i32 %1046, 0
  %1048 = zext i1 %1047 to i32
  %1049 = icmp ne i32 %1048, 0
  br i1 %1049, label %1050, label %1051

1050:                                             ; preds = %1042
  br label %1058

1051:                                             ; preds = %1042
  %1052 = load ptr, ptr %78, align 8
  %1053 = load i32, ptr %1052, align 8
  %1054 = add i32 %1053, -1
  store i32 %1054, ptr %1052, align 8
  %1055 = icmp eq i32 %1054, 0
  br i1 %1055, label %1056, label %1058

1056:                                             ; preds = %1051
  %1057 = load ptr, ptr %78, align 8
  call void @_Py_Dealloc(ptr noundef %1057) #7
  br label %1058

1058:                                             ; preds = %1050, %1051, %1056
  %1059 = load ptr, ptr %85, align 8
  store ptr %1059, ptr %84, align 8
  %1060 = load ptr, ptr %84, align 8
  %1061 = icmp eq ptr %1060, null
  br i1 %1061, label %1062, label %1063

1062:                                             ; preds = %1058
  br label %1064

1063:                                             ; preds = %1058
  br label %992

1064:                                             ; preds = %1062, %1022
  %1065 = load ptr, ptr %87, align 8
  store ptr %1065, ptr %79, align 8
  %1066 = load ptr, ptr %79, align 8
  store ptr %1066, ptr %4, align 8
  %1067 = load ptr, ptr %4, align 8
  %1068 = load i32, ptr %1067, align 8
  %1069 = icmp slt i32 %1068, 0
  %1070 = zext i1 %1069 to i32
  %1071 = icmp ne i32 %1070, 0
  br i1 %1071, label %1072, label %1073

1072:                                             ; preds = %1064
  br label %1080

1073:                                             ; preds = %1064
  %1074 = load ptr, ptr %79, align 8
  %1075 = load i32, ptr %1074, align 8
  %1076 = add i32 %1075, -1
  store i32 %1076, ptr %1074, align 8
  %1077 = icmp eq i32 %1076, 0
  br i1 %1077, label %1078, label %1080

1078:                                             ; preds = %1073
  %1079 = load ptr, ptr %79, align 8
  call void @_Py_Dealloc(ptr noundef %1079) #7
  br label %1080

1080:                                             ; preds = %1072, %1073, %1078
  %1081 = load ptr, ptr %84, align 8
  store ptr %1081, ptr %80, align 8
  br label %1082

1082:                                             ; preds = %1080, %988, %932, %876, %758, %757, %681, %625, %586, %499, %498, %435, %379, %284, %283, %211, %189, %166, %142, %118
  %1083 = load ptr, ptr %80, align 8
  ret ptr %1083
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i32 @_PyLong_IsCompact(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %struct._longobject, ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds %struct._object, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = call i32 @PyType_HasFeature(ptr noundef %6, i64 noundef 16777216)
  %8 = icmp ne i32 %7, 0
  %9 = xor i1 %8, true
  %10 = zext i1 %9 to i32
  %11 = sext i32 %10 to i64
  %12 = icmp ne i64 %11, 0
  br i1 %12, label %13, label %15

13:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__._PyLong_IsCompact, ptr noundef @.str.228, i32 noundef 123, ptr noundef @.str.229) #8
  unreachable

14:                                               ; No predecessors!
  br label %16

15:                                               ; preds = %1
  br label %16

16:                                               ; preds = %15, %14
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr inbounds %struct._longobject, ptr %17, i32 0, i32 1
  %19 = getelementptr inbounds %struct._PyLongValue, ptr %18, i32 0, i32 0
  %20 = load i64, ptr %19, align 8
  %21 = icmp ult i64 %20, 16
  %22 = zext i1 %21 to i32
  ret i32 %22
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal i64 @_PyLong_CompactValue(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds %struct._longobject, ptr %4, i32 0, i32 0
  %6 = getelementptr inbounds %struct._object, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = call i32 @PyType_HasFeature(ptr noundef %7, i64 noundef 16777216)
  %9 = icmp ne i32 %8, 0
  %10 = xor i1 %9, true
  %11 = zext i1 %10 to i32
  %12 = sext i32 %11 to i64
  %13 = icmp ne i64 %12, 0
  br i1 %13, label %14, label %16

14:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__._PyLong_CompactValue, ptr noundef @.str.228, i32 noundef 133, ptr noundef @.str.229) #8
  unreachable

15:                                               ; No predecessors!
  br label %17

16:                                               ; preds = %1
  br label %17

17:                                               ; preds = %16, %15
  %18 = load ptr, ptr %2, align 8
  %19 = call i32 @_PyLong_IsCompact(ptr noundef %18)
  %20 = icmp ne i32 %19, 0
  %21 = xor i1 %20, true
  %22 = zext i1 %21 to i32
  %23 = sext i32 %22 to i64
  %24 = icmp ne i64 %23, 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %17
  call void @__assert_rtn(ptr noundef @__func__._PyLong_CompactValue, ptr noundef @.str.228, i32 noundef 134, ptr noundef @.str.230) #8
  unreachable

26:                                               ; No predecessors!
  br label %28

27:                                               ; preds = %17
  br label %28

28:                                               ; preds = %27, %26
  %29 = load ptr, ptr %2, align 8
  %30 = getelementptr inbounds %struct._longobject, ptr %29, i32 0, i32 1
  %31 = getelementptr inbounds %struct._PyLongValue, ptr %30, i32 0, i32 0
  %32 = load i64, ptr %31, align 8
  %33 = and i64 %32, 3
  %34 = sub i64 1, %33
  store i64 %34, ptr %3, align 8
  %35 = load i64, ptr %3, align 8
  %36 = load ptr, ptr %2, align 8
  %37 = getelementptr inbounds %struct._longobject, ptr %36, i32 0, i32 1
  %38 = getelementptr inbounds %struct._PyLongValue, ptr %37, i32 0, i32 1
  %39 = getelementptr inbounds [1 x i32], ptr %38, i64 0, i64 0
  %40 = load i32, ptr %39, align 8
  %41 = zext i32 %40 to i64
  %42 = mul nsw i64 %35, %41
  ret i64 %42
}

declare ptr @PyNumber_Add(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal %struct.CompensatedSum @cs_from_double(double noundef %0) #0 {
  %2 = alloca %struct.CompensatedSum, align 8
  %3 = alloca double, align 8
  store double %0, ptr %3, align 8
  %4 = getelementptr inbounds %struct.CompensatedSum, ptr %2, i32 0, i32 0
  %5 = load double, ptr %3, align 8
  store double %5, ptr %4, align 8
  %6 = getelementptr inbounds %struct.CompensatedSum, ptr %2, i32 0, i32 1
  store double 0.000000e+00, ptr %6, align 8
  %7 = load %struct.CompensatedSum, ptr %2, align 8
  ret %struct.CompensatedSum %7
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal double @PyFloat_AS_DOUBLE(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call i32 @PyObject_TypeCheck(ptr noundef %3, ptr noundef @PyFloat_Type)
  %5 = icmp ne i32 %4, 0
  %6 = xor i1 %5, true
  %7 = zext i1 %6 to i32
  %8 = sext i32 %7 to i64
  %9 = icmp ne i64 %8, 0
  br i1 %9, label %10, label %12

10:                                               ; preds = %1
  call void @__assert_rtn(ptr noundef @__func__.PyFloat_AS_DOUBLE, ptr noundef @.str.231, i32 noundef 16, ptr noundef @.str.232) #8
  unreachable

11:                                               ; No predecessors!
  br label %13

12:                                               ; preds = %1
  br label %13

13:                                               ; preds = %12, %11
  %14 = load ptr, ptr %2, align 8
  %15 = getelementptr inbounds %struct.PyFloatObject, ptr %14, i32 0, i32 1
  %16 = load double, ptr %15, align 8
  ret double %16
}

declare ptr @PyFloat_FromDouble(double noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal double @cs_to_double([2 x double] %0) #0 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  %4 = alloca float, align 4
  %5 = alloca double, align 8
  %6 = alloca %struct.CompensatedSum, align 8
  store [2 x double] %0, ptr %6, align 8
  %7 = getelementptr inbounds %struct.CompensatedSum, ptr %6, i32 0, i32 1
  %8 = load double, ptr %7, align 8
  %9 = fcmp une double %8, 0.000000e+00
  br i1 %9, label %10, label %61

10:                                               ; preds = %1
  br i1 false, label %11, label %26

11:                                               ; preds = %10
  %12 = getelementptr inbounds %struct.CompensatedSum, ptr %6, i32 0, i32 1
  %13 = load double, ptr %12, align 8
  %14 = fptrunc double %13 to float
  store float %14, ptr %4, align 4
  %15 = load float, ptr %4, align 4
  %16 = load float, ptr %4, align 4
  %17 = fcmp oeq float %15, %16
  br i1 %17, label %18, label %22

18:                                               ; preds = %11
  %19 = load float, ptr %4, align 4
  %20 = call float @llvm.fabs.f32(float %19)
  %21 = fcmp une float %20, 0x7FF0000000000000
  br label %22

22:                                               ; preds = %11, %18
  %23 = phi i1 [ false, %11 ], [ %21, %18 ]
  %24 = zext i1 %23 to i32
  %25 = icmp ne i32 %24, 0
  br i1 %25, label %55, label %61

26:                                               ; preds = %10
  br i1 true, label %27, label %41

27:                                               ; preds = %26
  %28 = getelementptr inbounds %struct.CompensatedSum, ptr %6, i32 0, i32 1
  %29 = load double, ptr %28, align 8
  store double %29, ptr %3, align 8
  %30 = load double, ptr %3, align 8
  %31 = load double, ptr %3, align 8
  %32 = fcmp oeq double %30, %31
  br i1 %32, label %33, label %37

33:                                               ; preds = %27
  %34 = load double, ptr %3, align 8
  %35 = call double @llvm.fabs.f64(double %34)
  %36 = fcmp une double %35, 0x7FF0000000000000
  br label %37

37:                                               ; preds = %27, %33
  %38 = phi i1 [ false, %27 ], [ %36, %33 ]
  %39 = zext i1 %38 to i32
  %40 = icmp ne i32 %39, 0
  br i1 %40, label %55, label %61

41:                                               ; preds = %26
  %42 = getelementptr inbounds %struct.CompensatedSum, ptr %6, i32 0, i32 1
  %43 = load double, ptr %42, align 8
  store double %43, ptr %2, align 8
  %44 = load double, ptr %2, align 8
  %45 = load double, ptr %2, align 8
  %46 = fcmp oeq double %44, %45
  br i1 %46, label %47, label %51

47:                                               ; preds = %41
  %48 = load double, ptr %2, align 8
  %49 = call double @llvm.fabs.f64(double %48)
  %50 = fcmp une double %49, 0x7FF0000000000000
  br label %51

51:                                               ; preds = %41, %47
  %52 = phi i1 [ false, %41 ], [ %50, %47 ]
  %53 = zext i1 %52 to i32
  %54 = icmp ne i32 %53, 0
  br i1 %54, label %55, label %61

55:                                               ; preds = %51, %37, %22
  %56 = getelementptr inbounds %struct.CompensatedSum, ptr %6, i32 0, i32 0
  %57 = load double, ptr %56, align 8
  %58 = getelementptr inbounds %struct.CompensatedSum, ptr %6, i32 0, i32 1
  %59 = load double, ptr %58, align 8
  %60 = fadd double %57, %59
  store double %60, ptr %5, align 8
  br label %64

61:                                               ; preds = %51, %37, %22, %1
  %62 = getelementptr inbounds %struct.CompensatedSum, ptr %6, i32 0, i32 0
  %63 = load double, ptr %62, align 8
  store double %63, ptr %5, align 8
  br label %64

64:                                               ; preds = %61, %55
  %65 = load double, ptr %5, align 8
  ret double %65
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal %struct.CompensatedSum @cs_add([2 x double] %0, double noundef %1) #0 {
  %3 = alloca %struct.CompensatedSum, align 8
  %4 = alloca %struct.CompensatedSum, align 8
  %5 = alloca double, align 8
  %6 = alloca double, align 8
  store [2 x double] %0, ptr %4, align 8
  store double %1, ptr %5, align 8
  %7 = getelementptr inbounds %struct.CompensatedSum, ptr %4, i32 0, i32 0
  %8 = load double, ptr %7, align 8
  %9 = load double, ptr %5, align 8
  %10 = fadd double %8, %9
  store double %10, ptr %6, align 8
  %11 = getelementptr inbounds %struct.CompensatedSum, ptr %4, i32 0, i32 0
  %12 = load double, ptr %11, align 8
  %13 = call double @llvm.fabs.f64(double %12)
  %14 = load double, ptr %5, align 8
  %15 = call double @llvm.fabs.f64(double %14)
  %16 = fcmp oge double %13, %15
  br i1 %16, label %17, label %27

17:                                               ; preds = %2
  %18 = getelementptr inbounds %struct.CompensatedSum, ptr %4, i32 0, i32 0
  %19 = load double, ptr %18, align 8
  %20 = load double, ptr %6, align 8
  %21 = fsub double %19, %20
  %22 = load double, ptr %5, align 8
  %23 = fadd double %21, %22
  %24 = getelementptr inbounds %struct.CompensatedSum, ptr %4, i32 0, i32 1
  %25 = load double, ptr %24, align 8
  %26 = fadd double %25, %23
  store double %26, ptr %24, align 8
  br label %37

27:                                               ; preds = %2
  %28 = load double, ptr %5, align 8
  %29 = load double, ptr %6, align 8
  %30 = fsub double %28, %29
  %31 = getelementptr inbounds %struct.CompensatedSum, ptr %4, i32 0, i32 0
  %32 = load double, ptr %31, align 8
  %33 = fadd double %30, %32
  %34 = getelementptr inbounds %struct.CompensatedSum, ptr %4, i32 0, i32 1
  %35 = load double, ptr %34, align 8
  %36 = fadd double %35, %33
  store double %36, ptr %34, align 8
  br label %37

37:                                               ; preds = %27, %17
  %38 = getelementptr inbounds %struct.CompensatedSum, ptr %3, i32 0, i32 0
  %39 = load double, ptr %6, align 8
  store double %39, ptr %38, align 8
  %40 = getelementptr inbounds %struct.CompensatedSum, ptr %3, i32 0, i32 1
  %41 = getelementptr inbounds %struct.CompensatedSum, ptr %4, i32 0, i32 1
  %42 = load double, ptr %41, align 8
  store double %42, ptr %40, align 8
  %43 = load %struct.CompensatedSum, ptr %3, align 8
  ret %struct.CompensatedSum %43
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define internal void @_Py_DECREF_SPECIALIZED(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %8 = load ptr, ptr %4, align 8
  store ptr %8, ptr %3, align 8
  %9 = load ptr, ptr %3, align 8
  %10 = load i32, ptr %9, align 8
  %11 = icmp slt i32 %10, 0
  %12 = zext i1 %11 to i32
  %13 = icmp ne i32 %12, 0
  br i1 %13, label %14, label %15

14:                                               ; preds = %2
  br label %56

15:                                               ; preds = %2
  %16 = load ptr, ptr %4, align 8
  %17 = getelementptr inbounds %struct._object, ptr %16, i32 0, i32 0
  %18 = getelementptr inbounds %struct.anon, ptr %17, i32 0, i32 0
  %19 = load i32, ptr %18, align 8
  %20 = add i32 %19, -1
  store i32 %20, ptr %18, align 8
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %36

22:                                               ; preds = %15
  %23 = load ptr, ptr %4, align 8
  %24 = getelementptr inbounds %struct._object, ptr %23, i32 0, i32 0
  %25 = getelementptr inbounds %struct.anon, ptr %24, i32 0, i32 0
  %26 = load i32, ptr %25, align 8
  %27 = icmp ugt i32 %26, 0
  %28 = xor i1 %27, true
  %29 = zext i1 %28 to i32
  %30 = sext i32 %29 to i64
  %31 = icmp ne i64 %30, 0
  br i1 %31, label %32, label %34

32:                                               ; preds = %22
  call void @__assert_rtn(ptr noundef @__func__._Py_DECREF_SPECIALIZED, ptr noundef @.str.233, i32 noundef 233, ptr noundef @.str.234) #8
  unreachable

33:                                               ; No predecessors!
  br label %35

34:                                               ; preds = %22
  br label %35

35:                                               ; preds = %34, %33
  br label %56

36:                                               ; preds = %15
  br label %37

37:                                               ; preds = %36
  store ptr getelementptr inbounds (%struct.pyruntimestate, ptr @_PyRuntime, i32 0, i32 29), ptr %6, align 8
  %38 = load ptr, ptr %6, align 8
  %39 = getelementptr inbounds %struct._reftracer_runtime_state, ptr %38, i32 0, i32 0
  %40 = load ptr, ptr %39, align 8
  %41 = icmp ne ptr %40, null
  br i1 %41, label %42, label %52

42:                                               ; preds = %37
  %43 = load ptr, ptr %6, align 8
  %44 = getelementptr inbounds %struct._reftracer_runtime_state, ptr %43, i32 0, i32 1
  %45 = load ptr, ptr %44, align 8
  store ptr %45, ptr %7, align 8
  %46 = load ptr, ptr %6, align 8
  %47 = getelementptr inbounds %struct._reftracer_runtime_state, ptr %46, i32 0, i32 0
  %48 = load ptr, ptr %47, align 8
  %49 = load ptr, ptr %4, align 8
  %50 = load ptr, ptr %7, align 8
  %51 = call i32 %48(ptr noundef %49, i32 noundef 1, ptr noundef %50)
  br label %52

52:                                               ; preds = %42, %37
  br label %53

53:                                               ; preds = %52
  %54 = load ptr, ptr %5, align 8
  %55 = load ptr, ptr %4, align 8
  call void %54(ptr noundef %55)
  br label %56

56:                                               ; preds = %14, %53, %35
  ret void
}

declare void @_PyFloat_ExactDealloc(ptr noundef) #1

declare double @PyLong_AsDouble(ptr noundef) #1

declare %struct.Py_complex @PyComplex_AsCComplex(ptr noundef) #1

declare ptr @PyComplex_FromDoubles(double noundef, double noundef) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #3

declare void @_Py_Dealloc(ptr noundef) #1

attributes #0 = { noinline nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #2 = { cold noreturn "disable-tail-calls"="true" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #6 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #7 = { nounwind }
attributes #8 = { cold noreturn }
attributes #9 = { noreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 15, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Homebrew clang version 19.1.7"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
!15 = distinct !{!15, !7}
!16 = distinct !{!16, !7}
!17 = distinct !{!17, !7}
!18 = distinct !{!18, !7}
!19 = distinct !{!19, !7}
!20 = distinct !{!20, !7}
!21 = distinct !{!21, !7}
!22 = distinct !{!22, !7}
!23 = distinct !{!23, !7}
!24 = distinct !{!24, !7}
!25 = distinct !{!25, !7}
