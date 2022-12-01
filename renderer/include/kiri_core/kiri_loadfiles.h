/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:54:54
 * @FilePath: \core\include\kiri_core\kiri_loadfiles.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_LOAD_FILES_H_
#define _KIRI_LOAD_FILES_H_
#pragma once
#include <kiri_pch.h>
#include <kiri_define.h>
#include <root_directory.h>

class KiriLoadFiles
{
private:
    typedef String (*Builder)(const String &path);

public:
    static String GetPath(const String &path)
    {
        static String (*pathBuilder)(String const &) = GetPathBuilder();
        return (*pathBuilder)(path);
    }

    static String GetBuildPath(const String &path)
    {
        static String (*pathBuilder)(String const &) = GetBuildPathBuilder();
        return (*pathBuilder)(path);
    }

private:
    static String const &GetRoot()
    {
#ifdef KIRI_WINDOWS
        static char *envRoot = nullptr;
        static char const *givenRoot = nullptr;
        size_t sz = 0;
        if (_dupenv_s(&envRoot, &sz, "ROOT_PATH_PATH") == 0 && envRoot != nullptr)
        {
            givenRoot = envRoot;
            free(envRoot);
        }
        else
        {
            givenRoot = ROOT_PATH;
        }
        static String root = (givenRoot != nullptr ? givenRoot : "");
        return root;
#endif

#ifdef KIRI_APPLE
        static char const *envRoot = getenv("ROOT_PATH_PATH");
        static char const *givenRoot = (envRoot != nullptr ? envRoot : ROOT_PATH);
        static String root = (givenRoot != nullptr ? givenRoot : "");
        return root;
#endif
    }

    static String const &GetBuildRoot()
    {
#ifdef KIRI_WINDOWS
        static String root = ".";
        return root;
#endif

#ifdef KIRI_APPLE
        static char const *envRoot = getenv("LOGL_BUILD_ROOT_PATH");
        static char const *givenRoot = (envRoot != nullptr ? envRoot : MSWIN_BUILD_PATH);
        static String root = (givenRoot != nullptr ? givenRoot : "");
        return root;
#endif
    }

    static Builder GetPathBuilder()
    {
        return &KiriLoadFiles::GetPathRelativeRoot;
    }

    static Builder GetBuildPathBuilder()
    {
        return &KiriLoadFiles::GetBuildPathRelativeRoot;
    }

    static String GetPathRelativeRoot(const String &path)
    {
        return GetRoot() + String("/") + path;
    }

    static String GetBuildPathRelativeRoot(const String &path)
    {
        return GetBuildRoot() + String("/") + path;
    }
};

#endif
