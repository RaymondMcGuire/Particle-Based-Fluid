/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:16:02
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_constants.h
 */

#ifndef _KIRI_MATERIAL_CONSTANTS_H_
#define _KIRI_MATERIAL_CONSTANTS_H_
#pragma once
enum KIRI_MATERIAL_CONSTANT_TYPE
{
    T_EMERALD = 1,
    T_JADE = 2,
    T_OBSIDIAN = 3,
    T_PEARL = 4,
    T_RUBY = 5,
    T_TURQUOISE = 6,
    T_BRASS = 7,
    T_BRONZE = 8,
    T_CHROME = 9,
    T_COPPER = 10,
    T_GOLD = 11,
    T_SILVER = 12,
    T_BLACK_PLASTIC = 13,
    T_CYAN_PLASTIC = 14,
    T_GREEN_PLASTIC = 15,
    T_RED_PLASTIC = 16,
    T_WHITE_PLASTIC = 17,
    T_YELLOW_PLASTIC = 18,
    T_BLACK_RUBBER = 19,
    T_CYAN_RUBBER = 20,
    T_GREEN_RUBBER = 21,
    T_RED_RUBBER = 22,
    T_WHITE_RUBBER = 23,
    T_YELLOW_RUBBER = 24
};

struct STRUCT_MATERIAL
{
    constexpr STRUCT_MATERIAL(Vector3F _ambi, Vector3F _diff, Vector3F _spec, float _shin) : ambient(_ambi), diffuse(_diff), specular(_spec), shininess(_shin) {}
    Vector3F ambient;
    Vector3F diffuse;
    Vector3F specular;
    float shininess;
};

constexpr STRUCT_MATERIAL M_EMERALD(Vector3F(0.0215f, 0.1745f, 0.0215f), Vector3F(0.07568f, 0.61424f, 0.07568f), Vector3F(0.633f, 0.727811f, 0.633f), 0.6f);
constexpr STRUCT_MATERIAL M_JADE(Vector3F(0.135f, 0.2225f, 0.1575f), Vector3F(0.54f, 0.89f, 0.63f), Vector3F(0.316228f, 0.316228f, 0.316228f), 0.1f);
constexpr STRUCT_MATERIAL M_OBSIDIAN(Vector3F(0.05375f, 0.05f, 0.06625f), Vector3F(0.18275f, 0.17f, 0.22525f), Vector3F(0.332741f, 0.328634f, 0.346435f), 0.3f);
constexpr STRUCT_MATERIAL M_PEARL(Vector3F(0.25f, 0.20725f, 0.20725f), Vector3F(1.0f, 0.829f, 0.829f), Vector3F(0.296648f, 0.296648f, 0.296648f), 0.088f);
constexpr STRUCT_MATERIAL M_RUBY(Vector3F(0.1745f, 0.01175f, 0.01175f), Vector3F(0.61424f, 0.04136f, 0.04136f), Vector3F(0.727811f, 0.626959f, 0.626959f), 0.6f);
constexpr STRUCT_MATERIAL M_TURQUOISE(Vector3F(0.1f, 0.18725f, 0.1745f), Vector3F(0.396f, 0.74151f, 0.69102f), Vector3F(0.297254f, 0.30829f, 0.306678f), 0.1f);
constexpr STRUCT_MATERIAL M_BRASS(Vector3F(0.329412f, 0.223529f, 0.027451f), Vector3F(0.780392f, 0.568627f, 0.113725f), Vector3F(0.992157f, 0.941176f, 0.807843f), 0.21794872f);
constexpr STRUCT_MATERIAL M_BRONZE(Vector3F(0.2125f, 0.1275f, 0.054f), Vector3F(0.714f, 0.4284f, 0.18144f), Vector3F(0.393548f, 0.271906f, 0.166721f), 0.2f);
constexpr STRUCT_MATERIAL M_CHROME(Vector3F(0.25f, 0.25f, 0.25f), Vector3F(0.4f, 0.4f, 0.4f), Vector3F(0.774597f, 0.774597f, 0.774597f), 0.6f);
constexpr STRUCT_MATERIAL M_COPPER(Vector3F(0.19125f, 0.0735f, 0.0225f), Vector3F(0.7038f, 0.27048f, 0.0828f), Vector3F(0.256777f, 0.137622f, 0.086014f), 0.1f);
constexpr STRUCT_MATERIAL M_GOLD(Vector3F(0.24725f, 0.1995f, 0.0745f), Vector3F(0.75164f, 0.60648f, 0.22648f), Vector3F(0.628281f, 0.555802f, 0.366065f), 0.4f);
constexpr STRUCT_MATERIAL M_SILVER(Vector3F(0.19225f, 0.19225f, 0.19225f), Vector3F(0.50754f, 0.50754f, 0.50754f), Vector3F(0.508273f, 0.508273f, 0.508273f), 0.4f);
constexpr STRUCT_MATERIAL M_BLACK_PLASTIC(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.01f, 0.01f, 0.01f), Vector3F(0.5f, 0.5f, 0.5f), 0.25f);
constexpr STRUCT_MATERIAL M_CYAN_PLASTIC(Vector3F(0.0f, 0.1f, 0.06f), Vector3F(0.0f, 0.50980392f, 0.50980392f), Vector3F(0.50196078f, 0.50196078f, 0.50196078f), 0.25f);
constexpr STRUCT_MATERIAL M_GREEN_PLASTIC(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.1f, 0.35f, 0.1f), Vector3F(0.45f, 0.55f, 0.45f), 0.25f);
constexpr STRUCT_MATERIAL M_RED_PLASTIC(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.5f, 0.0f, 0.0f), Vector3F(0.7f, 0.6f, 0.6f), 0.25f);
constexpr STRUCT_MATERIAL M_WHITE_PLASTIC(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.55f, 0.55f, 0.55f), Vector3F(0.7f, 0.7f, 0.7f), 0.25f);
constexpr STRUCT_MATERIAL M_YELLOW_PLASTIC(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.5f, 0.5f, 0.0f), Vector3F(0.6f, 0.6f, 0.5f), 0.25f);
constexpr STRUCT_MATERIAL M_BLACK_RUBBER(Vector3F(0.02f, 0.02f, 0.02f), Vector3F(0.01f, 0.01f, 0.01f), Vector3F(0.4f, 0.4f, 0.4f), 0.078125f);
constexpr STRUCT_MATERIAL M_CYAN_RUBBER(Vector3F(0.0f, 0.05f, 0.05f), Vector3F(0.4f, 0.5f, 0.5f), Vector3F(0.04f, 0.7f, 0.7f), 0.078125f);
constexpr STRUCT_MATERIAL M_GREEN_RUBBER(Vector3F(0.0f, 0.05f, 0.0f), Vector3F(0.4f, 0.5f, 0.4f), Vector3F(0.04f, 0.7f, 0.04f), 0.078125f);
constexpr STRUCT_MATERIAL M_RED_RUBBER(Vector3F(0.05f, 0.0f, 0.0f), Vector3F(0.5f, 0.4f, 0.4f), Vector3F(0.7f, 0.04f, 0.04f), 0.078125f);
constexpr STRUCT_MATERIAL M_WHITE_RUBBER(Vector3F(0.05f, 0.05f, 0.05f), Vector3F(0.5f, 0.5f, 0.5f), Vector3F(0.7f, 0.7f, 0.7f), 0.078125f);
constexpr STRUCT_MATERIAL M_YELLOW_RUBBER(Vector3F(0.05f, 0.05f, 0.0f), Vector3F(0.5f, 0.5f, 0.4f), Vector3F(0.7f, 0.7f, 0.04f), 0.078125f);

#endif