// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_CUDADEMDATA_KIRI_FLATBUFFERS_H_
#define FLATBUFFERS_GENERATED_CUDADEMDATA_KIRI_FLATBUFFERS_H_

#include "flatbuffers/flatbuffers.h"

#include "basic_types_generated.h"

namespace KIRI {
namespace FlatBuffers {

struct DemInitBoxVolume;
struct DemInitBoxVolumeBuilder;

struct DemShapeVolumes;
struct DemShapeVolumesBuilder;

struct CudaDemData;
struct CudaDemDataBuilder;

enum CudaDemType {
  CudaDemType_DEM = 0,
  CudaDemType_MIN = CudaDemType_DEM,
  CudaDemType_MAX = CudaDemType_DEM
};

inline const CudaDemType (&EnumValuesCudaDemType())[1] {
  static const CudaDemType values[] = {
    CudaDemType_DEM
  };
  return values;
}

inline const char * const *EnumNamesCudaDemType() {
  static const char * const names[2] = {
    "DEM",
    nullptr
  };
  return names;
}

inline const char *EnumNameCudaDemType(CudaDemType e) {
  if (flatbuffers::IsOutRange(e, CudaDemType_DEM, CudaDemType_DEM)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesCudaDemType()[index];
}

struct DemInitBoxVolume FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef DemInitBoxVolumeBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_BOX_LOWER = 4,
    VT_BOX_SIZE = 6,
    VT_BOX_COLOR = 8
  };
  const KIRI::FlatBuffers::float3 *box_lower() const {
    return GetStruct<const KIRI::FlatBuffers::float3 *>(VT_BOX_LOWER);
  }
  const KIRI::FlatBuffers::int3 *box_size() const {
    return GetStruct<const KIRI::FlatBuffers::int3 *>(VT_BOX_SIZE);
  }
  const KIRI::FlatBuffers::float3 *box_color() const {
    return GetStruct<const KIRI::FlatBuffers::float3 *>(VT_BOX_COLOR);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<KIRI::FlatBuffers::float3>(verifier, VT_BOX_LOWER) &&
           VerifyField<KIRI::FlatBuffers::int3>(verifier, VT_BOX_SIZE) &&
           VerifyField<KIRI::FlatBuffers::float3>(verifier, VT_BOX_COLOR) &&
           verifier.EndTable();
  }
};

struct DemInitBoxVolumeBuilder {
  typedef DemInitBoxVolume Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_box_lower(const KIRI::FlatBuffers::float3 *box_lower) {
    fbb_.AddStruct(DemInitBoxVolume::VT_BOX_LOWER, box_lower);
  }
  void add_box_size(const KIRI::FlatBuffers::int3 *box_size) {
    fbb_.AddStruct(DemInitBoxVolume::VT_BOX_SIZE, box_size);
  }
  void add_box_color(const KIRI::FlatBuffers::float3 *box_color) {
    fbb_.AddStruct(DemInitBoxVolume::VT_BOX_COLOR, box_color);
  }
  explicit DemInitBoxVolumeBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<DemInitBoxVolume> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DemInitBoxVolume>(end);
    return o;
  }
};

inline flatbuffers::Offset<DemInitBoxVolume> CreateDemInitBoxVolume(
    flatbuffers::FlatBufferBuilder &_fbb,
    const KIRI::FlatBuffers::float3 *box_lower = 0,
    const KIRI::FlatBuffers::int3 *box_size = 0,
    const KIRI::FlatBuffers::float3 *box_color = 0) {
  DemInitBoxVolumeBuilder builder_(_fbb);
  builder_.add_box_color(box_color);
  builder_.add_box_size(box_size);
  builder_.add_box_lower(box_lower);
  return builder_.Finish();
}

struct DemShapeVolumes FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef DemShapeVolumesBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_FOLDERS = 4,
    VT_FILES = 6,
    VT_OFFSET_GROUND = 8
  };
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *folders() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_FOLDERS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *files() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_FILES);
  }
  bool offset_ground() const {
    return GetField<uint8_t>(VT_OFFSET_GROUND, 0) != 0;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_FOLDERS) &&
           verifier.VerifyVector(folders()) &&
           verifier.VerifyVectorOfStrings(folders()) &&
           VerifyOffset(verifier, VT_FILES) &&
           verifier.VerifyVector(files()) &&
           verifier.VerifyVectorOfStrings(files()) &&
           VerifyField<uint8_t>(verifier, VT_OFFSET_GROUND) &&
           verifier.EndTable();
  }
};

struct DemShapeVolumesBuilder {
  typedef DemShapeVolumes Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_folders(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> folders) {
    fbb_.AddOffset(DemShapeVolumes::VT_FOLDERS, folders);
  }
  void add_files(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> files) {
    fbb_.AddOffset(DemShapeVolumes::VT_FILES, files);
  }
  void add_offset_ground(bool offset_ground) {
    fbb_.AddElement<uint8_t>(DemShapeVolumes::VT_OFFSET_GROUND, static_cast<uint8_t>(offset_ground), 0);
  }
  explicit DemShapeVolumesBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<DemShapeVolumes> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DemShapeVolumes>(end);
    return o;
  }
};

inline flatbuffers::Offset<DemShapeVolumes> CreateDemShapeVolumes(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> folders = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> files = 0,
    bool offset_ground = false) {
  DemShapeVolumesBuilder builder_(_fbb);
  builder_.add_files(files);
  builder_.add_folders(folders);
  builder_.add_offset_ground(offset_ground);
  return builder_.Finish();
}

inline flatbuffers::Offset<DemShapeVolumes> CreateDemShapeVolumesDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *folders = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *files = nullptr,
    bool offset_ground = false) {
  auto folders__ = folders ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*folders) : 0;
  auto files__ = files ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*files) : 0;
  return KIRI::FlatBuffers::CreateDemShapeVolumes(
      _fbb,
      folders__,
      files__,
      offset_ground);
}

struct CudaDemData FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef CudaDemDataBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_REST_DENSITY = 4,
    VT_REST_MASS = 6,
    VT_KERNEL_RADIUS = 8,
    VT_PARTICLE_RADIUS = 10,
    VT_GRAVITY = 12,
    VT_YOUNG = 14,
    VT_POISSON = 16,
    VT_FRICTION_ANGLE = 18,
    VT_DAMPING = 20
  };
  float rest_density() const {
    return GetField<float>(VT_REST_DENSITY, 0.0f);
  }
  float rest_mass() const {
    return GetField<float>(VT_REST_MASS, 0.0f);
  }
  float kernel_radius() const {
    return GetField<float>(VT_KERNEL_RADIUS, 0.0f);
  }
  float particle_radius() const {
    return GetField<float>(VT_PARTICLE_RADIUS, 0.0f);
  }
  const KIRI::FlatBuffers::float3 *gravity() const {
    return GetStruct<const KIRI::FlatBuffers::float3 *>(VT_GRAVITY);
  }
  float young() const {
    return GetField<float>(VT_YOUNG, 0.0f);
  }
  float poisson() const {
    return GetField<float>(VT_POISSON, 0.0f);
  }
  float friction_angle() const {
    return GetField<float>(VT_FRICTION_ANGLE, 0.0f);
  }
  float damping() const {
    return GetField<float>(VT_DAMPING, 0.0f);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<float>(verifier, VT_REST_DENSITY) &&
           VerifyField<float>(verifier, VT_REST_MASS) &&
           VerifyField<float>(verifier, VT_KERNEL_RADIUS) &&
           VerifyField<float>(verifier, VT_PARTICLE_RADIUS) &&
           VerifyField<KIRI::FlatBuffers::float3>(verifier, VT_GRAVITY) &&
           VerifyField<float>(verifier, VT_YOUNG) &&
           VerifyField<float>(verifier, VT_POISSON) &&
           VerifyField<float>(verifier, VT_FRICTION_ANGLE) &&
           VerifyField<float>(verifier, VT_DAMPING) &&
           verifier.EndTable();
  }
};

struct CudaDemDataBuilder {
  typedef CudaDemData Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_rest_density(float rest_density) {
    fbb_.AddElement<float>(CudaDemData::VT_REST_DENSITY, rest_density, 0.0f);
  }
  void add_rest_mass(float rest_mass) {
    fbb_.AddElement<float>(CudaDemData::VT_REST_MASS, rest_mass, 0.0f);
  }
  void add_kernel_radius(float kernel_radius) {
    fbb_.AddElement<float>(CudaDemData::VT_KERNEL_RADIUS, kernel_radius, 0.0f);
  }
  void add_particle_radius(float particle_radius) {
    fbb_.AddElement<float>(CudaDemData::VT_PARTICLE_RADIUS, particle_radius, 0.0f);
  }
  void add_gravity(const KIRI::FlatBuffers::float3 *gravity) {
    fbb_.AddStruct(CudaDemData::VT_GRAVITY, gravity);
  }
  void add_young(float young) {
    fbb_.AddElement<float>(CudaDemData::VT_YOUNG, young, 0.0f);
  }
  void add_poisson(float poisson) {
    fbb_.AddElement<float>(CudaDemData::VT_POISSON, poisson, 0.0f);
  }
  void add_friction_angle(float friction_angle) {
    fbb_.AddElement<float>(CudaDemData::VT_FRICTION_ANGLE, friction_angle, 0.0f);
  }
  void add_damping(float damping) {
    fbb_.AddElement<float>(CudaDemData::VT_DAMPING, damping, 0.0f);
  }
  explicit CudaDemDataBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<CudaDemData> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<CudaDemData>(end);
    return o;
  }
};

inline flatbuffers::Offset<CudaDemData> CreateCudaDemData(
    flatbuffers::FlatBufferBuilder &_fbb,
    float rest_density = 0.0f,
    float rest_mass = 0.0f,
    float kernel_radius = 0.0f,
    float particle_radius = 0.0f,
    const KIRI::FlatBuffers::float3 *gravity = 0,
    float young = 0.0f,
    float poisson = 0.0f,
    float friction_angle = 0.0f,
    float damping = 0.0f) {
  CudaDemDataBuilder builder_(_fbb);
  builder_.add_damping(damping);
  builder_.add_friction_angle(friction_angle);
  builder_.add_poisson(poisson);
  builder_.add_young(young);
  builder_.add_gravity(gravity);
  builder_.add_particle_radius(particle_radius);
  builder_.add_kernel_radius(kernel_radius);
  builder_.add_rest_mass(rest_mass);
  builder_.add_rest_density(rest_density);
  return builder_.Finish();
}

}  // namespace FlatBuffers
}  // namespace KIRI

#endif  // FLATBUFFERS_GENERATED_CUDADEMDATA_KIRI_FLATBUFFERS_H_
