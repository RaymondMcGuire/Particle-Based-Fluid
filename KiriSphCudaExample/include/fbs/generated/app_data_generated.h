// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_APPDATA_KIRI_FLATBUFFERS_H_
#define FLATBUFFERS_GENERATED_APPDATA_KIRI_FLATBUFFERS_H_

#include "flatbuffers/flatbuffers.h"

#include "basic_types_generated.h"

namespace KIRI {
namespace FlatBuffers {

struct CameraData;
struct CameraDataBuilder;

struct SceneData;
struct SceneDataBuilder;

struct AppData;
struct AppDataBuilder;

struct CameraData FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef CameraDataBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_POSITION = 4,
    VT_YAW = 6,
    VT_PITCH = 8
  };
  const KIRI::FlatBuffers::float3 *position() const {
    return GetStruct<const KIRI::FlatBuffers::float3 *>(VT_POSITION);
  }
  float yaw() const {
    return GetField<float>(VT_YAW, 0.0f);
  }
  float pitch() const {
    return GetField<float>(VT_PITCH, 0.0f);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<KIRI::FlatBuffers::float3>(verifier, VT_POSITION) &&
           VerifyField<float>(verifier, VT_YAW) &&
           VerifyField<float>(verifier, VT_PITCH) &&
           verifier.EndTable();
  }
};

struct CameraDataBuilder {
  typedef CameraData Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_position(const KIRI::FlatBuffers::float3 *position) {
    fbb_.AddStruct(CameraData::VT_POSITION, position);
  }
  void add_yaw(float yaw) {
    fbb_.AddElement<float>(CameraData::VT_YAW, yaw, 0.0f);
  }
  void add_pitch(float pitch) {
    fbb_.AddElement<float>(CameraData::VT_PITCH, pitch, 0.0f);
  }
  explicit CameraDataBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<CameraData> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<CameraData>(end);
    return o;
  }
};

inline flatbuffers::Offset<CameraData> CreateCameraData(
    flatbuffers::FlatBufferBuilder &_fbb,
    const KIRI::FlatBuffers::float3 *position = 0,
    float yaw = 0.0f,
    float pitch = 0.0f) {
  CameraDataBuilder builder_(_fbb);
  builder_.add_pitch(pitch);
  builder_.add_yaw(yaw);
  builder_.add_position(position);
  return builder_.Finish();
}

struct SceneData FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef SceneDataBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_WORLD_LOWER = 4,
    VT_WORLD_UPPER = 6,
    VT_WORLD_CENTER = 8,
    VT_WORLD_SIZE = 10,
    VT_CAMERA = 12
  };
  const KIRI::FlatBuffers::float3 *world_lower() const {
    return GetStruct<const KIRI::FlatBuffers::float3 *>(VT_WORLD_LOWER);
  }
  const KIRI::FlatBuffers::float3 *world_upper() const {
    return GetStruct<const KIRI::FlatBuffers::float3 *>(VT_WORLD_UPPER);
  }
  const KIRI::FlatBuffers::float3 *world_center() const {
    return GetStruct<const KIRI::FlatBuffers::float3 *>(VT_WORLD_CENTER);
  }
  const KIRI::FlatBuffers::float3 *world_size() const {
    return GetStruct<const KIRI::FlatBuffers::float3 *>(VT_WORLD_SIZE);
  }
  const KIRI::FlatBuffers::CameraData *camera() const {
    return GetPointer<const KIRI::FlatBuffers::CameraData *>(VT_CAMERA);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<KIRI::FlatBuffers::float3>(verifier, VT_WORLD_LOWER) &&
           VerifyField<KIRI::FlatBuffers::float3>(verifier, VT_WORLD_UPPER) &&
           VerifyField<KIRI::FlatBuffers::float3>(verifier, VT_WORLD_CENTER) &&
           VerifyField<KIRI::FlatBuffers::float3>(verifier, VT_WORLD_SIZE) &&
           VerifyOffset(verifier, VT_CAMERA) &&
           verifier.VerifyTable(camera()) &&
           verifier.EndTable();
  }
};

struct SceneDataBuilder {
  typedef SceneData Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_world_lower(const KIRI::FlatBuffers::float3 *world_lower) {
    fbb_.AddStruct(SceneData::VT_WORLD_LOWER, world_lower);
  }
  void add_world_upper(const KIRI::FlatBuffers::float3 *world_upper) {
    fbb_.AddStruct(SceneData::VT_WORLD_UPPER, world_upper);
  }
  void add_world_center(const KIRI::FlatBuffers::float3 *world_center) {
    fbb_.AddStruct(SceneData::VT_WORLD_CENTER, world_center);
  }
  void add_world_size(const KIRI::FlatBuffers::float3 *world_size) {
    fbb_.AddStruct(SceneData::VT_WORLD_SIZE, world_size);
  }
  void add_camera(flatbuffers::Offset<KIRI::FlatBuffers::CameraData> camera) {
    fbb_.AddOffset(SceneData::VT_CAMERA, camera);
  }
  explicit SceneDataBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<SceneData> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<SceneData>(end);
    return o;
  }
};

inline flatbuffers::Offset<SceneData> CreateSceneData(
    flatbuffers::FlatBufferBuilder &_fbb,
    const KIRI::FlatBuffers::float3 *world_lower = 0,
    const KIRI::FlatBuffers::float3 *world_upper = 0,
    const KIRI::FlatBuffers::float3 *world_center = 0,
    const KIRI::FlatBuffers::float3 *world_size = 0,
    flatbuffers::Offset<KIRI::FlatBuffers::CameraData> camera = 0) {
  SceneDataBuilder builder_(_fbb);
  builder_.add_camera(camera);
  builder_.add_world_size(world_size);
  builder_.add_world_center(world_center);
  builder_.add_world_upper(world_upper);
  builder_.add_world_lower(world_lower);
  return builder_.Finish();
}

struct AppData FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef AppDataBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_BGEO_EXPORT_MODE_ENABLE = 4,
    VT_RENDER_MODE_ENABLE = 6,
    VT_RENDER_MODE_FPS = 8,
    VT_SCENE = 10
  };
  bool bgeo_export_mode_enable() const {
    return GetField<uint8_t>(VT_BGEO_EXPORT_MODE_ENABLE, 0) != 0;
  }
  bool render_mode_enable() const {
    return GetField<uint8_t>(VT_RENDER_MODE_ENABLE, 0) != 0;
  }
  float render_mode_fps() const {
    return GetField<float>(VT_RENDER_MODE_FPS, 0.0f);
  }
  const KIRI::FlatBuffers::SceneData *scene() const {
    return GetPointer<const KIRI::FlatBuffers::SceneData *>(VT_SCENE);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint8_t>(verifier, VT_BGEO_EXPORT_MODE_ENABLE) &&
           VerifyField<uint8_t>(verifier, VT_RENDER_MODE_ENABLE) &&
           VerifyField<float>(verifier, VT_RENDER_MODE_FPS) &&
           VerifyOffset(verifier, VT_SCENE) &&
           verifier.VerifyTable(scene()) &&
           verifier.EndTable();
  }
};

struct AppDataBuilder {
  typedef AppData Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_bgeo_export_mode_enable(bool bgeo_export_mode_enable) {
    fbb_.AddElement<uint8_t>(AppData::VT_BGEO_EXPORT_MODE_ENABLE, static_cast<uint8_t>(bgeo_export_mode_enable), 0);
  }
  void add_render_mode_enable(bool render_mode_enable) {
    fbb_.AddElement<uint8_t>(AppData::VT_RENDER_MODE_ENABLE, static_cast<uint8_t>(render_mode_enable), 0);
  }
  void add_render_mode_fps(float render_mode_fps) {
    fbb_.AddElement<float>(AppData::VT_RENDER_MODE_FPS, render_mode_fps, 0.0f);
  }
  void add_scene(flatbuffers::Offset<KIRI::FlatBuffers::SceneData> scene) {
    fbb_.AddOffset(AppData::VT_SCENE, scene);
  }
  explicit AppDataBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<AppData> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<AppData>(end);
    return o;
  }
};

inline flatbuffers::Offset<AppData> CreateAppData(
    flatbuffers::FlatBufferBuilder &_fbb,
    bool bgeo_export_mode_enable = false,
    bool render_mode_enable = false,
    float render_mode_fps = 0.0f,
    flatbuffers::Offset<KIRI::FlatBuffers::SceneData> scene = 0) {
  AppDataBuilder builder_(_fbb);
  builder_.add_scene(scene);
  builder_.add_render_mode_fps(render_mode_fps);
  builder_.add_render_mode_enable(render_mode_enable);
  builder_.add_bgeo_export_mode_enable(bgeo_export_mode_enable);
  return builder_.Finish();
}

}  // namespace FlatBuffers
}  // namespace KIRI

#endif  // FLATBUFFERS_GENERATED_APPDATA_KIRI_FLATBUFFERS_H_