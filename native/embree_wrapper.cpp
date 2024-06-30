#include <embree4/rtcore.h>
#include <embree4/rtcore_device.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_embree(py::module &m) {
    m.def("embree_version", []() { return RTC_VERSION_STRING; });

    py::enum_<RTCFormat>(m, "RTCFormat")
        .value("UNDEFINED", RTC_FORMAT_UNDEFINED)
        .value("UCHAR", RTC_FORMAT_UCHAR)
        .value("UCHAR2", RTC_FORMAT_UCHAR2)
        .value("UCHAR3", RTC_FORMAT_UCHAR3)
        .value("UCHAR4", RTC_FORMAT_UCHAR4)
        .value("CHAR", RTC_FORMAT_CHAR)
        .value("CHAR2", RTC_FORMAT_CHAR2)
        .value("CHAR3", RTC_FORMAT_CHAR3)
        .value("CHAR4", RTC_FORMAT_CHAR4)
        .value("USHORT", RTC_FORMAT_USHORT)
        .value("USHORT2", RTC_FORMAT_USHORT2)
        .value("USHORT3", RTC_FORMAT_USHORT3)
        .value("USHORT4", RTC_FORMAT_USHORT4)
        .value("SHORT", RTC_FORMAT_SHORT)
        .value("SHORT2", RTC_FORMAT_SHORT2)
        .value("SHORT3", RTC_FORMAT_SHORT3)
        .value("SHORT4", RTC_FORMAT_SHORT4)
        .value("UINT", RTC_FORMAT_UINT)
        .value("UINT2", RTC_FORMAT_UINT2)
        .value("UINT3", RTC_FORMAT_UINT3)
        .value("UINT4", RTC_FORMAT_UINT4)
        .value("INT", RTC_FORMAT_INT)
        .value("INT2", RTC_FORMAT_INT2)
        .value("INT3", RTC_FORMAT_INT3)
        .value("INT4", RTC_FORMAT_INT4)
        .value("ULLONG", RTC_FORMAT_ULLONG)
        .value("ULLONG2", RTC_FORMAT_ULLONG2)
        .value("ULLONG3", RTC_FORMAT_ULLONG3)
        .value("ULLONG4", RTC_FORMAT_ULLONG4)
        .value("LLONG", RTC_FORMAT_LLONG)
        .value("LLONG2", RTC_FORMAT_LLONG2)
        .value("LLONG3", RTC_FORMAT_LLONG3)
        .value("LLONG4", RTC_FORMAT_LLONG4)
        .value("FLOAT", RTC_FORMAT_FLOAT)
        .value("FLOAT2", RTC_FORMAT_FLOAT2)
        .value("FLOAT3", RTC_FORMAT_FLOAT3)
        .value("FLOAT4", RTC_FORMAT_FLOAT4)
        .value("FLOAT5", RTC_FORMAT_FLOAT5)
        .value("FLOAT6", RTC_FORMAT_FLOAT6)
        .value("FLOAT7", RTC_FORMAT_FLOAT7)
        .value("FLOAT8", RTC_FORMAT_FLOAT8)
        .value("FLOAT9", RTC_FORMAT_FLOAT9)
        .value("FLOAT10", RTC_FORMAT_FLOAT10)
        .value("FLOAT11", RTC_FORMAT_FLOAT11)
        .value("FLOAT12", RTC_FORMAT_FLOAT12)
        .value("FLOAT13", RTC_FORMAT_FLOAT13)
        .value("FLOAT14", RTC_FORMAT_FLOAT14)
        .value("FLOAT15", RTC_FORMAT_FLOAT15)
        .value("FLOAT16", RTC_FORMAT_FLOAT16)
        .value("FLOAT2X2_ROW_MAJOR", RTC_FORMAT_FLOAT2X2_ROW_MAJOR)
        .value("FLOAT2X3_ROW_MAJOR", RTC_FORMAT_FLOAT2X3_ROW_MAJOR)
        .value("FLOAT2X4_ROW_MAJOR", RTC_FORMAT_FLOAT2X4_ROW_MAJOR)
        .value("FLOAT3X2_ROW_MAJOR", RTC_FORMAT_FLOAT3X2_ROW_MAJOR)
        .value("FLOAT3X3_ROW_MAJOR", RTC_FORMAT_FLOAT3X3_ROW_MAJOR)
        .value("FLOAT3X4_ROW_MAJOR", RTC_FORMAT_FLOAT3X4_ROW_MAJOR)
        .value("FLOAT4X2_ROW_MAJOR", RTC_FORMAT_FLOAT4X2_ROW_MAJOR)
        .value("FLOAT4X3_ROW_MAJOR", RTC_FORMAT_FLOAT4X3_ROW_MAJOR)
        .value("FLOAT4X4_ROW_MAJOR", RTC_FORMAT_FLOAT4X4_ROW_MAJOR)
        .value("FLOAT2X2_COLUMN_MAJOR", RTC_FORMAT_FLOAT2X2_COLUMN_MAJOR)
        .value("FLOAT2X3_COLUMN_MAJOR", RTC_FORMAT_FLOAT2X3_COLUMN_MAJOR)
        .value("FLOAT2X4_COLUMN_MAJOR", RTC_FORMAT_FLOAT2X4_COLUMN_MAJOR)
        .value("FLOAT3X2_COLUMN_MAJOR", RTC_FORMAT_FLOAT3X2_COLUMN_MAJOR)
        .value("FLOAT3X3_COLUMN_MAJOR", RTC_FORMAT_FLOAT3X3_COLUMN_MAJOR)
        .value("FLOAT3X4_COLUMN_MAJOR", RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR)
        .value("FLOAT4X2_COLUMN_MAJOR", RTC_FORMAT_FLOAT4X2_COLUMN_MAJOR)
        .value("FLOAT4X3_COLUMN_MAJOR", RTC_FORMAT_FLOAT4X3_COLUMN_MAJOR)
        .value("FLOAT4X4_COLUMN_MAJOR", RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR)
        .value("GRID", RTC_FORMAT_GRID)
        .value("QUATERNION_DECOMPOSITION", RTC_FORMAT_QUATERNION_DECOMPOSITION)
        .export_values();

    py::enum_<RTCBuildQuality>(m, "RTCBuildQuality")
        .value("LOW", RTC_BUILD_QUALITY_LOW)
        .value("MEDIUM", RTC_BUILD_QUALITY_MEDIUM)
        .value("HIGH", RTC_BUILD_QUALITY_HIGH)
        .value("REFIT", RTC_BUILD_QUALITY_REFIT)
        .export_values();

    py::class_<RTCBounds, std::shared_ptr<RTCBounds>>(m, "RTCBounds")
        .def(py::init<>())
        .def_readwrite("lower_x", &RTCBounds::lower_x)
        .def_readwrite("lower_y", &RTCBounds::lower_y)
        .def_readwrite("lower_z", &RTCBounds::lower_z)
        .def_readwrite("upper_x", &RTCBounds::upper_x)
        .def_readwrite("upper_y", &RTCBounds::upper_y)
        .def_readwrite("upper_z", &RTCBounds::upper_z);

    py::class_<RTCLinearBounds, std::shared_ptr<RTCLinearBounds>>(
        m, "RTCLinearBounds")
        .def(py::init<>())
        .def_readwrite("bounds0", &RTCLinearBounds::bounds0)
        .def_readwrite("bounds1", &RTCLinearBounds::bounds1);

    py::enum_<RTCFeatureFlags>(m, "RTCFeatureFlags", py::arithmetic())
        .value("NONE", RTC_FEATURE_FLAG_NONE)
        .value("MOTION_BLUR", RTC_FEATURE_FLAG_MOTION_BLUR)
        .value("TRIANGLE", RTC_FEATURE_FLAG_TRIANGLE)
        .value("QUAD", RTC_FEATURE_FLAG_QUAD)
        .value("GRID", RTC_FEATURE_FLAG_GRID)
        .value("SUBDIVISION", RTC_FEATURE_FLAG_SUBDIVISION)
        .value("CONE_LINEAR_CURVE", RTC_FEATURE_FLAG_CONE_LINEAR_CURVE)
        .value("ROUND_LINEAR_CURVE", RTC_FEATURE_FLAG_ROUND_LINEAR_CURVE)
        .value("FLAT_LINEAR_CURVE", RTC_FEATURE_FLAG_FLAT_LINEAR_CURVE)
        .value("ROUND_BEZIER_CURVE", RTC_FEATURE_FLAG_ROUND_BEZIER_CURVE)
        .value("FLAT_BEZIER_CURVE", RTC_FEATURE_FLAG_FLAT_BEZIER_CURVE)
        .value("NORMAL_ORIENTED_BEZIER_CURVE",
               RTC_FEATURE_FLAG_NORMAL_ORIENTED_BEZIER_CURVE)
        .value("ROUND_BSPLINE_CURVE", RTC_FEATURE_FLAG_ROUND_BSPLINE_CURVE)
        .value("FLAT_BSPLINE_CURVE", RTC_FEATURE_FLAG_FLAT_BSPLINE_CURVE)
        .value("NORMAL_ORIENTED_BSPLINE_CURVE",
               RTC_FEATURE_FLAG_NORMAL_ORIENTED_BSPLINE_CURVE)
        .value("ROUND_HERMITE_CURVE", RTC_FEATURE_FLAG_ROUND_HERMITE_CURVE)
        .value("FLAT_HERMITE_CURVE", RTC_FEATURE_FLAG_FLAT_HERMITE_CURVE)
        .value("NORMAL_ORIENTED_HERMITE_CURVE",
               RTC_FEATURE_FLAG_NORMAL_ORIENTED_HERMITE_CURVE)
        .value("ROUND_CATMULL_ROM_CURVE",
               RTC_FEATURE_FLAG_ROUND_CATMULL_ROM_CURVE)
        .value("FLAT_CATMULL_ROM_CURVE",
               RTC_FEATURE_FLAG_FLAT_CATMULL_ROM_CURVE)
        .value("NORMAL_ORIENTED_CATMULL_ROM_CURVE",
               RTC_FEATURE_FLAG_NORMAL_ORIENTED_CATMULL_ROM_CURVE)
        .value("SPHERE_POINT", RTC_FEATURE_FLAG_SPHERE_POINT)
        .value("DISC_POINT", RTC_FEATURE_FLAG_DISC_POINT)
        .value("ORIENTED_DISC_POINT", RTC_FEATURE_FLAG_ORIENTED_DISC_POINT)
        .value("POINT", RTC_FEATURE_FLAG_POINT)
        .value("ROUND_CURVES", RTC_FEATURE_FLAG_ROUND_CURVES)
        .value("FLAT_CURVES", RTC_FEATURE_FLAG_FLAT_CURVES)
        .value("NORMAL_ORIENTED_CURVES",
               RTC_FEATURE_FLAG_NORMAL_ORIENTED_CURVES)
        .value("LINEAR_CURVES", RTC_FEATURE_FLAG_LINEAR_CURVES)
        .value("BEZIER_CURVES", RTC_FEATURE_FLAG_BEZIER_CURVES)
        .value("BSPLINE_CURVES", RTC_FEATURE_FLAG_BSPLINE_CURVES)
        .value("HERMITE_CURVES", RTC_FEATURE_FLAG_HERMITE_CURVES)
        .value("CURVES", RTC_FEATURE_FLAG_CURVES)
        .value("INSTANCE", RTC_FEATURE_FLAG_INSTANCE)
        .value("FILTER_FUNCTION_IN_ARGUMENTS",
               RTC_FEATURE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS)
        .value("FILTER_FUNCTION_IN_GEOMETRY",
               RTC_FEATURE_FLAG_FILTER_FUNCTION_IN_GEOMETRY)
        .value("FILTER_FUNCTION", RTC_FEATURE_FLAG_FILTER_FUNCTION)
        .value("USER_GEOMETRY_CALLBACK_IN_ARGUMENTS",
               RTC_FEATURE_FLAG_USER_GEOMETRY_CALLBACK_IN_ARGUMENTS)
        .value("USER_GEOMETRY_CALLBACK_IN_GEOMETRY",
               RTC_FEATURE_FLAG_USER_GEOMETRY_CALLBACK_IN_GEOMETRY)
        .value("USER_GEOMETRY", RTC_FEATURE_FLAG_USER_GEOMETRY)
        .value("32_BIT_RAY_MASK", RTC_FEATURE_FLAG_32_BIT_RAY_MASK)
        .value("INSTANCE_ARRAY", RTC_FEATURE_FLAG_INSTANCE_ARRAY)
        .value("ALL", RTC_FEATURE_FLAG_ALL)
        .export_values();

    py::enum_<RTCRayQueryFlags>(m, "RTCRayQueryFlags", py::arithmetic())
        .value("NONE", RTC_RAY_QUERY_FLAG_NONE)
        .value("INVOKE_ARGUMENT_FILTER",
               RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER)
        .value("INCOHERENT", RTC_RAY_QUERY_FLAG_INCOHERENT)
        .value("COHERENT", RTC_RAY_QUERY_FLAG_COHERENT)
        .export_values();

    m.attr("RTC_MAX_INSTANCE_LEVEL_COUNT") = RTC_MAX_INSTANCE_LEVEL_COUNT;

    py::class_<RTCRayQueryContext, std::shared_ptr<RTCRayQueryContext>>(
        m, "RTCRayQueryContext")
        .def(py::init<>())
        .def_property(
            "instPrimID",
            [](const RTCRayQueryContext &c) {
                py::capsule free_when_done(c.instPrimID, [](void *v) {});
                return py::array_t<unsigned int>({c.instPrimID},
                                                 {sizeof(unsigned int)},
                                                 c.instPrimID, free_when_done);
            },
            [](RTCRayQueryContext &c, py::array_t<unsigned int> v) {
                if (v.size() != RTC_MAX_INSTANCE_LEVEL_COUNT) {
                    throw std::runtime_error("instStackSize must be a scalar");
                }
                memcpy(c.instPrimID, v.data(), v.size() * sizeof(unsigned int));
            })
        .def_property(
            "instID",
            [](const RTCRayQueryContext &c) {
                py::capsule free_when_done(c.instID, [](void *v) {});
                return py::array_t<unsigned int>({c.instID},
                                                 {sizeof(unsigned int)},
                                                 c.instID, free_when_done);
            },
            [](RTCRayQueryContext &c, py::array_t<unsigned int> v) {
                if (v.size() > RTC_MAX_INSTANCE_LEVEL_COUNT) {
                    throw std::runtime_error("instID array size exceeds "
                                             "RTC_MAX_INSTANCE_LEVEL_COUNT");
                }
                std::memcpy(c.instID, v.data(),
                            v.size() * sizeof(unsigned int));
            });

    class RTCFilterFunctionNArgumentsWrapper {
        // This class is a wrapper around RTCFilterFunctionNArguments
      public:
        std::vector<int32_t> valid;
        std::vector<uint64_t> geometryUserPtr;
        std::shared_ptr<RTCRayQueryContext> context;
        uint64_t ray;
        uint64_t hit;
        uint32_t N;

        RTCFilterFunctionNArguments get_raw() {
            RTCFilterFunctionNArguments args;
            args.valid           = valid.data();
            args.geometryUserPtr = geometryUserPtr.data();
            args.context         = context.get();
            args.ray             = (RTCRayN*)ray;
            args.hit             = (RTCHitN *)hit;
            args.N               = N;
            return args;
        }
    };

    py::class_<RTCFilterFunctionNArgumentsWrapper,
               std::shared_ptr<RTCFilterFunctionNArgumentsWrapper>>(
        m, "RTCFilterFunctionNArguments")
        .def(py::init<>())
        .def_readwrite("valid", &RTCFilterFunctionNArgumentsWrapper::valid)
        .def_readwrite("geometryUserPtr",
                       &RTCFilterFunctionNArgumentsWrapper::geometryUserPtr)
        .def_readwrite("context", &RTCFilterFunctionNArgumentsWrapper::context)
        .def_readwrite("ray", &RTCFilterFunctionNArgumentsWrapper::ray)
        .def_readwrite("hit", &RTCFilterFunctionNArgumentsWrapper::hit)
        .def_readwrite("N", &RTCFilterFunctionNArgumentsWrapper::N);

    py::class_<RTCPointQuery, std::shared_ptr<RTCPointQuery>>(m,
                                                              "RTCPointQuery")
        .def(py::init<>())
        .def_readwrite("x", &RTCPointQuery::x)
        .def_readwrite("y", &RTCPointQuery::y)
        .def_readwrite("z", &RTCPointQuery::z)
        .def_readwrite("radius", &RTCPointQuery::radius)
        .def_readwrite("time", &RTCPointQuery::time);

    py::class_<RTCPointQuery4, std::shared_ptr<RTCPointQuery4>>(
        m, "RTCPointQuery4")
        .def(py::init<>())
        .def_property(
            "x",
            [](const RTCPointQuery4 &q) {
                py::capsule free_when_done(q.x, [](void *v) {});
                return py::array_t<float>({4}, {sizeof(float)}, q.x,
                                          free_when_done);
            },
            [](RTCPointQuery4 &q, py::array_t<float> v) {
                if (v.size() != 4) {
                    throw std::runtime_error("x must be a 4-element array");
                }
                std::memcpy(&q.x, v.data(), 4 * sizeof(float));
            })
        .def_property(
            "y",
            [](const RTCPointQuery4 &q) {
                py::capsule free_when_done(q.y, [](void *v) {});
                return py::array_t<float>({4}, {sizeof(float)}, q.y,
                                          free_when_done);
            },
            [](RTCPointQuery4 &q, py::array_t<float> v) {
                if (v.size() != 4) {
                    throw std::runtime_error("y must be a 4-element array");
                }
                std::memcpy(&q.y, v.data(), 4 * sizeof(float));
            })

        .def_property(
            "z",
            [](const RTCPointQuery4 &q) {
                py::capsule free_when_done(q.z, [](void *v) {});
                return py::array_t<float>({4}, {sizeof(float)}, q.z,
                                          free_when_done);
            },
            [](RTCPointQuery4 &q, py::array_t<float> v) {
                if (v.size() != 4) {
                    throw std::runtime_error("z must be a 4-element array");
                }
                std::memcpy(&q.z, v.data(), 4 * sizeof(float));
            })
        .def_property(
            "radius",
            [](const RTCPointQuery4 &q) {
                py::capsule free_when_done(q.radius, [](void *v) {});
                return py::array_t<float>({4}, {sizeof(float)}, q.radius,
                                          free_when_done);
            },
            [](RTCPointQuery4 &q, py::array_t<float> v) {
                if (v.size() != 4) {
                    throw std::runtime_error(
                        "radius must be a 4-element array");
                }
                std::memcpy(&q.radius, v.data(), 4 * sizeof(float));
            })
        .def_property(
            "time",
            [](const RTCPointQuery4 &q) {
                py::capsule free_when_done(q.time, [](void *v) {});
                return py::array_t<float>({4}, {sizeof(float)}, q.time,
                                          free_when_done);
            },
            [](RTCPointQuery4 &q, py::array_t<float> v) {
                if (v.size() != 4) {
                    throw std::runtime_error("time must be a 4-element array");
                }
                std::memcpy(&q.time, v.data(), 4 * sizeof(float));
            });

    m.def("rtcInitRayQueryContext",
          [](std::shared_ptr<RTCRayQueryContext> context) {
              rtcInitRayQueryContext(context.get());
          });
}

PYBIND11_MODULE(embree_wrapper, m) { init_embree(m); }